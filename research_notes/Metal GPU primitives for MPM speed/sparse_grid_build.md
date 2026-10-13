# Sparse-grid (NanoVDB) construction for Newton implicit MPM: the CUDA path, its primitives, and a Metal port

Sources were read locally and checked against upstream on 2026-10-10:
- **Warp.** The innate-inc Warp fork at `upstream/warp-innate-mpm` (1.18.0.dev20260917). These files are byte-identical (md5) to NVIDIA/warp tag `v1.18.0`: `volume_builder.cu`, `volume_builder.cpp`, `volume_builder.h`, `fem/geometry/nanogrid.py`, `fem/utils.py`, `fem/space/partition.py` and `native/sort.cpp`. `volume.cpp` differs, because it contains the fork's Metal bridge.
- **Newton.** Newton at `upstream/newton-1.5.2` (tag v1.5.2, commit cf5378db).
- **Short names.** "W118" below means `https://github.com/NVIDIA/warp/blob/v1.18.0/` and "N152" means `https://github.com/newton-physics/newton/blob/v1.5.2/`.

## Q1. How Warp (1.16-1.18) builds a NanoVDB grid from points on the GPU, what is CUDA-only, and the CPU path

### Takeaway
Warp 1.16 replaced NanoVDB's own `nanovdb::tools::cuda::PointsToGrid` builder with a key-based builder that Warp wrote itself. The build is:
1. Emit 64-bit keys, then sort them and run-length encode them with CUB. Only two sorts are needed.
2. Derive the leaf and lower keys by bit-shifting the sorted keys, which keeps them sorted.
3. Fill each node level with one thread per node, using binary search into the parent key array and 32/64-bit atomics for masks and bboxes.

The only CUDA-specific parts are the CUB calls (`DeviceRadixSort::SortKeys`, `DeviceRunLengthEncode::Encode`, `DeviceScan::ExclusiveSum`), the `atomicOr/Min/Max/Add` intrinsics and the stream plumbing. The per-node logic sits in a header shared with a CPU implementation (`volume_builder.cpp`, also new in 1.16). The CPU implementation does the same thing with `std::sort` + `std::unique`, so it produces the same tree layout and voxel order.

### Cited Findings
**History: 1.15 vs 1.16+**
- **Warp ≤1.15 used NanoVDB's builder.** `volume_builder.cu` included `<nanovdb/tools/cuda/PointsToGrid.cuh>` and built grids via `nanovdb::tools::cuda::PointsToGrid<BuildT, Resource> p2g(params.map, stream)` → `p2g.getHandle(...)`. — [warp v1.15.0 volume_builder.cu](https://github.com/NVIDIA/warp/blob/v1.15.0/warp/native/volume_builder.cu)
- **Warp ≤1.15 had no CPU path.** `allocate_by_tiles` raised "Only CUDA devices are supported for allocate_by_tiles", and `allocate_by_voxels` required CUDA. — [warp v1.15.0 types.py](https://github.com/NVIDIA/warp/blob/v1.15.0/warp/_src/types.py)
- **Warp 1.16 added rebuildable volumes, including on the CPU.** The changelog (1.16.0, 2026-08-03) reads: "Add rebuildable NanoVDB volumes through `wp.Volume.allocate_by_tiles(..., rebuildable=True)`, `wp.Volume.allocate_by_voxels(..., rebuildable=True)`, and `wp.Volume.rebuild()`. Support fixed capacities, optional point masks, CPU execution, CUDA graph-capturable allocation and rebuilding, and in-place refreshes of rebuildable `warp.fem.Nanogrid` topologies" (GH-1606). — [W118 CHANGELOG.md](https://github.com/NVIDIA/warp/blob/v1.18.0/CHANGELOG.md)
- **The 1.16+ `.cu` builder no longer includes PointsToGrid.** Its only external dependency is `<cub/cub.cuh>`. 1.16.0 and 1.18.0 have the same kernel list. — [W118 warp/native/volume_builder.cu](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/volume_builder.cu)

**GPU build, step by step (`rebuild_count_points` + `rebuild_populate_grid_from_scratch`)** — source for all items: [W118 volume_builder.cu](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/volume_builder.cu), [W118 volume_builder.h](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/volume_builder.h)
1. **Allocate scratch.** `rebuild_alloc_key_scratch` allocates `keys_a`, `keys_b`, `leaf/lower/upper/voxel_keys`, `parent_keys`, `run_counts` and `counts`, each N entries of 64 bits. This runs on every build and every rebuild (`wp_alloc_device`, then free).
2. **Emit upper keys.** `rebuild_emit_upper_keys` gives one thread per point. It converts world→index space with `map.applyInverseMapF(p).round()`, or takes integer coords directly. The key is `rebuild_upper_key_from_coord`, built from (x,y,z)+2^31, each shifted >>12 and packed 21 bits per axis. The comment says it matches "NanoVDB's upper-root-tile key ordering". Masked points get `REBUILD_INVALID_KEY = ~0ull`.
3. **Sort and deduplicate the upper keys.** `cub::DeviceRadixSort::SortKeys(... count, 0, 64, stream)` sorts all 64 bits, keys only. `cub::DeviceRunLengthEncode::Encode` then produces the unique upper keys and their count.
4. **Emit hierarchy keys.** `rebuild_emit_hierarchy_keys` gives one thread per point. It binary-searches the sorted unique upper keys for `upper_id`, then builds `key = upper_id<<36 | upper_offset<<21 | lower_offset<<9 | leaf_offset`, with offsets from `pnanovdb_{upper,lower,leaf}_coord_to_offset`.
5. **Sort the hierarchy keys.** This is the second full 64-bit `SortKeys`. For an index grid of active voxels (`allocate_by_voxels`), an RLE of the sorted keys gives the unique voxel keys.
6. **Derive leaf and lower keys without sorting.** `rebuild_shift_keys<9>` followed by an RLE gives the unique leaf keys. `rebuild_shift_keys<21>` followed by an RLE gives the unique lower keys. A right shift is monotone, so the shifted array stays sorted.
7. **Finalize counts.** `rebuild_finalize_counts<<<1,1>>>` trims the trailing INVALID key and checks capacities against status flags.
8. **Exact (non-rebuildable) path: one host sync.** `allocate_exact_grid_from_points_impl` copies the counts device→host (`wp_memcpy_d2h` + `wp_cuda_stream_synchronize`) to size the grid buffer. The docstring says this is why "Exact, non-rebuildable volume allocation is not graph-capture safe". The rebuildable path uses the reserved capacities instead and never syncs. — [W118 types.py `Volume.allocate_by_voxels`](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/_src/types.py)
9. **Populate the tree, one launch per level.**
   - `rebuild_init_grid_tree_root<<<1,1>>>` sets up the grid, tree and root.
   - `rebuild_build_upper_nodes` runs one thread per upper node; node i is the i-th sorted unique upper key. It is followed by `rebuild_set_upper_background_values`, with 32768 table entries per node.
   - `rebuild_build_lower_nodes` runs one thread per lower key. It finds its parent upper node and sets the parent child-mask bit with a 64-bit `atomicOr`, then sets the background values (4096 per node).
   - `rebuild_build_leaf_nodes` binary-searches `lower_keys` for its parent and sets the parent child-mask bit with an atomic OR.
   - For OnIndex (active-voxel) grids:
     - `rebuild_set_active_voxels` runs one thread per unique voxel key. It binary-searches `leaf_keys`, sets the value-mask bit with `atomicOr`, and increments `leaf_active_counts[leaf]` with `atomicAdd`.
     - `cub::DeviceScan::ExclusiveSum` (u32→u64) turns the per-leaf counts into `leaf_active_prefix`.
     - `rebuild_finalize_onindex_leaves` writes each leaf's first-index offset `1 + prefix`.
   - Bounding boxes: reset, then propagate leaf → lower → upper with `atomicMin/atomicMax` on int32 coords, then `rebuild_finalize_world_bbox`.
10. **Node layout is a pure function of the sorted unique key arrays.** Node i at each level is the i-th unique key. Voxel linear index = leaf prefix + bit rank within the leaf. The atomics (OR on mask words, min/max on bboxes, add on per-leaf counts) are order-independent, so the result is deterministic whatever the thread scheduling. [Inference from the code above; the atomic operations listed are commutative and associative on integers.]

**CPU path (`volume_builder.cpp`, used for `device="cpu"`)** — [W118 volume_builder.cpp](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/volume_builder.cpp)
- **Same algorithm, serial.** It uses the same key helpers from `volume_builder.h` (`rebuild_upper_key_from_coord`, `rebuild_hierarchy_key`):
  - upper keys: `std::sort` + `std::unique` (`rebuild_sort_unique`);
  - hierarchy keys: `std::lower_bound` for upper_id, then `std::sort`, dropping INVALID keys;
  - leaf and lower keys: `>>9` and `>>21`, each followed by `rebuild_sort_unique`;
  - masks and bboxes: non-atomic read-modify-write.
- **Same output.** The sorted unique key sets are identical to the CUDA path's, so node ordering and voxel indices match.

**Metal in the innate-inc fork (not upstream)** — source: the local `warp/native/volume.cpp` in the fork. This is not in NVIDIA/warp, so there is no public URL.
- **Every Metal volume entry point wraps the host builder.** This covers `wp_volume_from_tiles_device`, `wp_volume_from_active_voxels_device`, `wp_volume_rebuild_from_active_voxels_device` and `get_voxels`. The wrapper is `MetalVolumeScope`, which calls `wp_metal_synchronize(ordinal)` ("inputs may still be written by kernels"). It then calls the `_host` builder on unified memory and re-keys the descriptor under the Metal GPU address (`metal_volume_rekey`).
- **Cost.** Every grid build drains the GPU and runs serially on the CPU. In the granular example the "Allocate grid" timer measured 11.7-13.4 ms per step on metal:0. — local files `runs/mpm/profile/sections_mpm_granular_now.json`, `granular_branch.json`, `granular_oracle.json`

### Inferences
- **Porting the 1.16+ builder to Metal is mostly mechanical.**
  - It does not depend on NanoVDB's CUDA tools (PointsToGrid, `nanovdb::cuda::DeviceBuffer`).
  - Its kernels are thin wrappers around `CUDA_CALLABLE` helpers in `volume_builder.h` (30 of them) and the PNanoVDB C accessors. The fork's `PNanoVDB.h` already has a `PNANOVDB_METAL_MODE` for `__METAL_VERSION__`.
  - The real work is the three device primitives: 64-bit keys-only radix sort, RLE and exclusive scan. These are the same primitives covered in the sibling note `metal_sort_scan.md`.
- **64-bit `atomicOr` on mask words.** MSL has no general 64-bit atomic OR. Because OR is bitwise, it can be replaced exactly by a 32-bit `atomic_fetch_or_explicit` on the low or high half that contains the bit. `atomicMin/Max` on int32 and `atomicAdd` on u32 map directly to MSL atomics.
- **Parity.**
  - Any correct sort plus RLE gives the same sorted unique key set. Keys-only sorting with deduplication makes stability irrelevant inside the volume builder.
  - So a Metal GPU builder would produce bit-identical NanoVDB layout and voxel order to CUDA and to the current host path. Parity is not at risk from the builder itself.

### Gaps
- I did not find NVIDIA documentation explaining the 1.16 decision to drop PointsToGrid. The GH-1606 issue text was not read.
- I did not check whether the `nanovdb::Map::applyInverseMapF(...).round()` world→index rounding is bit-identical between CUDA (FMA contraction on) and Metal. If FMA contraction differs, a point exactly on a voxel boundary could land in a different voxel. See `research_notes/CUDA and Metal float parity/cuda_fma_contraction.md`.

## Q2. What Newton 1.5.2's SolverImplicitMPM rebuilds each step, which Warp utilities it calls, and whether results depend on sort stability

### Takeaway
`Config.grid_type` has three values:
- **`"sparse"`, the default.** With `max_active_cell_count = -1` (also the default, and what the granular example uses), it calls `wp.Volume.allocate_by_voxels(particle_q)` every step, an exact build with a host sync. It then constructs `fem.Nanogrid(volume)`, which builds a second volume for the nodes from the 8 corners of every cell.
- **Rebuildable sparse.** With `max_active_cell_count > 0`, padding 0 and a Q1 velocity basis, the grid is allocated once with capacities and then refreshed in place via `grid.rebuild(...)` → `Volume.rebuild` + `_refresh_rebuildable_topology`. This path is graph-capturable on CUDA.
- **`"dense"` / `"fixed"`.** These use a dense `fem.Grid3D` over the particle bounding box, which "fixed" allocates once.

Downstream, the particle binning (`PicQuadrature` → `compress_node_indices`) and the space partitions (`_finalize_node_indices` → `compress_node_indices`) use `radix_sort_pairs` + `runlength_encode` + `array_scan`. These depend on sort stability: CUB's sort is stable, and the CPU LSD radix sort is too.

### Cited Findings
**Grid types and configuration**
- **Options and defaults.** `grid_type: Literal["sparse","dense","fixed"] = "sparse"`; `max_active_cell_count: int = -1` ("``-1`` retains per-step sparse-grid allocation"); node capacities `max_leaf/lower/upper_node_count = -1`. — [N152 solver_implicit_mpm.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)
- **When a sparse grid is rebuildable.** `self._sparse_rebuildable = grid_type == "sparse" and max_active_cell_count > 0 and grid_padding == 0 and velocity_basis == "Q1" and strain_rebuild_safe and collider_rebuild_safe`. Rebuildable grids force particle-mode warm-start. — same file
- **Step-time branch (`_particles_to_cells`).**
  - Fixed or rebuildable grids are reused. For rebuildable ones, `grid.rebuild(positions, point_envs=..., status=self._grid_status, point_mask=point_mask)` runs, followed by a `record_volume_rebuild_status` kernel, which "retain[s] and report[s] rebuild failures without a host synchronization".
  - Otherwise `_allocate_grid` runs under `self._timer("Allocate grid")`. — same file and [N152 implicit_mpm_solver_kernels.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py)
- **The "Allocate grid" section in plain sparse mode.**
  - Newton's helper `allocate_by_voxels(positions, voxel_size, padding_voxels)` calls `wp.Volume.allocate_by_voxels(voxel_points=particle_q.flatten(), voxel_size)`.
  - With padding, it loops: `get_voxels` → `pad_voxels` kernel (3×3×3) → `allocate_by_voxels` again.
  - It then builds `fem.Nanogrid(volume, temporary_store)`.
  - For separate worlds, it goes through `fem.Nanogrid.from_environment_voxels(...)`. — same files
- **Capacity sizing (`_rebuild_capacity`).** It does one throwaway `allocate_by_voxels` and reads `get_voxels().numpy()` on the host. Lower and upper counts come from `np.unique(floor_divide(ijk, 128))` and `np.unique(floor_divide(ijk, 4096))`, times a ratio of 16. This happens once, at allocation. — [N152 implicit_mpm_solver_kernels.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py)
- **Newton requires `warp-lang>=1.16.0`,** so the PointsToGrid-free builder is always in use. — [N152 pyproject.toml](https://github.com/newton-physics/newton/blob/v1.5.2/pyproject.toml)
- **The Q1 node grid is a second volume.**
  - Non-rebuildable: `fem.Nanogrid.__init__` calls `grid.get_voxels`, then `_build_node_grid`. That launches `_cell_node_indices` (8 corners per cell) and calls `wp.Volume.allocate_by_voxels(cell_nodes.flatten())`, a second exact build over 8×cells points, then `node_grid.get_voxels`.
  - Rebuildable: `_build_rebuildable_node_grid` keeps candidate and mask buffers, and `_refresh_rebuildable_topology` calls `self._node_grid.rebuild(...)` (and the edge and face grids if used) each step. — [W118 fem/geometry/nanogrid.py](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/_src/fem/geometry/nanogrid.py)
- **The `"dense"` / `"fixed"` grid.**
  - `compute_bounds` on CUDA, or `np.min/max(positions.numpy())` otherwise.
  - Then `fem.Grid3D(bounds_lo, bounds_hi, res)`.
  - The doc says "Dense grids read dynamic bounds on the host". "fixed" keeps the first grid; the active subset is chosen by `_create_geometry_partition` → `fem.ExplicitGeometryPartition(max_cell_count=...)`.
  - The granular example only graph-captures on CUDA when `grid_type == "fixed"`. — [N152 solver_implicit_mpm.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py), [N152 example_mpm_granular.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/examples/mpm/example_mpm_granular.py)

**Warp utilities on the per-step path**
- **`compress_node_indices`** — [W118 fem/utils.py](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/_src/fem/utils.py)
  1. `_prepare_node_sort_kernel` writes `keys[i] = node` (or NULL) and `values[i] = i // divisor`.
  2. `radix_sort_pairs(keys, values, count)`.
  3. `runlength_encode(sorted keys → unique_node_indices, counts, run_count)`.
  4. `_scatter_node_counts` kernel.
  5. `array_scan(node_offsets, inclusive=True)`.
- **Its callers.**
  - `PicQuadrature` calls it to map particles to cells (`_cell_particle_offsets`, `_cell_particle_indices`, `_active_cell_count`). — [W118 fem/quadrature/pic_quadrature.py](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/_src/fem/quadrature/pic_quadrature.py)
  - `NodePartition._finalize_node_indices` calls `compress_node_indices(NodeCategory.COUNT, node_category)` and then scatters partition indices. Without `max_node_count`, it copies `category_offsets` to the host and waits on an event. — [W118 fem/space/partition.py](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/_src/fem/space/partition.py)
- **Newton's rheology solver also calls `wp.utils.array_scan`** (`batch_counts` → offsets). — [N152 solve_rheology.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solve_rheology.py)

**Stability**
- **CUB's sort is stable.** `radix_sort_pairs` on CUDA uses `cub::DeviceRadixSort::SortPairs` on `cub::DoubleBuffer`s. — [W118 native/sort.cu](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/sort.cu). CCCL documents "DeviceRadixSort is stable." — [CCCL DeviceRadixSort](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html)
- **The CPU sort is stable too.** `radix_sort_pairs_host` is an LSD counting sort with 16-bit digits per pass, scattering with `histogram[b]++` in input order. — [W118 native/sort.cpp](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/sort.cpp)
- **The Metal fork runs these utilities on the host.** `warp.utils` sorts, RLE and scans run host-side on Metal (`_runs_on_host`). During capture they are recorded as host calls (`metal_record_host_call`; native `wp_metal_capture_host_op`, "the Python utilities such as array_scan and radix_sort_pairs run on the host for Metal"). — local fork `warp/_src/utils.py`, `warp/native/metal.mm`

### Inferences
- **Results depend on stability.**
  - In `compress_node_indices`, the values are the original array indices. A stable sort lists the particles of each cell in ascending particle index; an unstable sort permutes them.
  - That order feeds quadrature-point order in `PicQuadrature`, and so the summation order of particle→grid integrals and assembled matrices. That changes floating-point results.
  - Likewise, `_finalize_node_indices` orders the nodes inside each category by ascending space-node index only if the sort is stable. This sets partition node numbering, and so matrix row order.
  - A Metal GPU `radix_sort_pairs` must therefore be stable, or must sort a packed `(key<<32 | index)` 64-bit key, to keep CUDA parity.
- **Two exact volume builds per plain-sparse step:** the cell grid over N particles and the node grid over 8×cells corners. Each costs two 64-bit sorts plus RLEs.
  - The exact path's device→host count copy is an inherent sync even on CUDA.
  - On Metal today, both builds also drain the GPU and run on the CPU. That is likely where most of the 12-13 ms "Allocate grid" goes. The "Scratchpad" (~7.9 ms) and "Bin particles" (~2.7 ms) sections contain the `compress_node_indices` sorts.
- **The rebuildable mode is the CUDA-graph-friendly configuration.** A Metal port should target it first, because it removes all host syncs except the explicit `check_sparse_grid_rebuild_status`. But it changes behavior (warm-start mode forced to "particles"), so CUDA parity runs must use the same config.

### Gaps
- **No per-sub-step timing split.** I did not measure on Metal how "Allocate grid" divides between the cell-grid build, the node-grid build and the synchronizations.
- **No check of Newton's own sorts.** I did not check whether Newton's rheology/coloring code (`_compute_coloring`, `node_color`) calls `radix_sort_pairs` directly. Only the `fem` paths were traced.

## Q3. Prior art for building sparse/hash grids for MPM on GPUs, and on Metal

### Takeaway
The GPU-MPM literature (Gao 2018 GSPGrid, Wang 2020 multi-GPU, Fei 2021) rebuilds a block-sparse grid every step:
1. Compute 64-bit interleaved particle codes.
2. Insert the block codes into a GPU spatial hash table that assigns consecutive block indices.
3. Run a histogram (counting) sort of particles by (block index, cell), not a full radix sort.

This is fast but gives a block ordering that depends on hash-insertion order, so it is not parity-compatible with NanoVDB's sorted-key order. On Metal there is essentially no prior art: Taichi removed sparse SNodes from its Metal backend, Genesis MPM is dense-only, and I found no open Metal sparse-MPM implementation.

### Cited Findings
- **Gao et al. 2018, GSPGrid** (ACM TOG 37(6) Art. 254). "SPGrid translates the cell index of each particle to a 64-bit integer offset, which is used as the keyword for sorting... We therefore use spatial hashing to transform these sparse offsets of GSPGrid blocks into consecutive numbers. Whenever a new block is touched, the index assigned to the block increments by one." They also write: "In GPU, this is not a concern anymore; thus we use histogram sort instead of the conventional radix sort... The observed performance speedup approaches an order of magnitude." — [Gao et al. 2018, GPU Optimization of Material Point Methods](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf)
- **Gao 2018 hash table.** The table's key is 64-bit, and its capacity is 64× the number of sparse GSPGrid blocks. It "is only used for particle ordering and page topology construction which are the two beginning tasks of each time step". Neighbor-block topology comes from spatial hashing "in O(1)". The cube example has 7M particles, 900K cells and 17K blocks. — same source
- **Wang et al. 2020, multi-GPU MPM** (ACM TOG 39(4) Art. 30).
  - Their "Partition Update" step maintains "the active-block array and the mapping from block coordinates to array indices".
  - GPU-SPGrid "explicitly manages grid blocks with spatial hashing, which maps spatial block coordinates to block indices in an array".
  - Particle-bin offsets come from "a lightweight exclusive scan".
  - The new particle indices are "uniquely determined by a spatial hash with a 32-bit integer", using the previous block ID plus a −1/0/+1 advection vector, since particles move at most one cell per step. — [Wang et al. 2020, A Massively Parallel and Scalable Multi-GPU Material Point Method](https://dl.acm.org/doi/10.1145/3386569.3392442)
- **Fei, Huang, Gao 2021.** They interleave 3D indices into a 64-bit offset (block code in the high bits, cell code in the low bits). A hash table records the "gblocks" and then their 3×3×3 neighbors, giving the "pblocks", which are "consecutively stored and indexed in memory". They choose histogram sort over radix sort and sort to cell granularity. Reported speed: 1.7×-8.6× faster than prior work on one GPU. — [Fei et al. 2021, Principles towards Real-Time Simulation of MPM on Modern GPUs (arXiv:2111.00699)](https://arxiv.org/abs/2111.00699)
- **Taichi.** "Using sparse data structures on the Metal backend is now deprecated. The support for Dynamic SNode has been removed in v1.3.0, and the support for Pointer/Bitmasked SNode will be removed in v1.4.0"; "The LLVM-based backends (CPU/CUDA) offer the full functionality". — [Taichi docs: Spatially Sparse Data Structures](https://docs.taichi-lang.org/docs/sparse); removal listed in [Taichi v1.4.0 release](https://github.com/taichi-dev/taichi/releases/tag/v1.4.0)
- **Genesis MPM.** It uses a dense grid (`use_sparse_grid` and `leaf_block_size` are deprecated no-ops that only warn) and float-atomic P2G, so it is not bitwise deterministic. Read from the Genesis `main` branch on 2026-10-09. — [Genesis mpm_solver.py](https://raw.githubusercontent.com/Genesis-Embodied-AI/Genesis/main/genesis/engine/solvers/mpm_solver.py)
- **Metal 64-bit atomics are limited.** Per Dawn's MSL backend, only u64 min/max exist, without a return value, on Apple8 (M2) / Apple9 and not M1. There is no 64-bit OR or add. — [Dawn commit: Atomic 64 bit for MSL](https://dawn.googlesource.com/dawn/+/b620b2fe514a5ee2e6a2d924fd998c514019da87); [ue5-nanite-macos AtomicsWorkaround](https://github.com/philipturner/ue5-nanite-macos/blob/main/AtomicsWorkaround)
- **NanoVDB background.** NanoVDB is the GPU-friendly, linearized, pointer-free VDB used by Warp volumes (Museth 2021, SIGGRAPH Talks). — [NanoVDB: A GPU-Friendly and Portable VDB Data Structure](https://dl.acm.org/doi/10.1145/3450623.3464653)
- **A web search for Metal sparse-grid MPM found nothing:** no open-source Metal compute MPM with a sparse grid or hash table. The results were unrelated (Gaussian splatting, lattice QCD, mumax3 Metal ports). — [web search results, e.g. msplat](https://pypi.org/project/msplat/)

### Inferences
- **Hash-based block allocation is not parity-compatible on its own.** Atomic insertion order sets the block indices, so they vary run to run. It can only be used for parity if a sort of the unique block keys follows it. The number of unique keys is small (blocks, or NanoVDB leaves/lowers, are far fewer than particles), so that sort is cheap. The result would reproduce NanoVDB's order.
- **Reuse the histogram-sort idea under a NanoVDB key.** Particles move less than one cell per step under CFL, so leaf membership changes slowly. A per-step counting sort keyed by (leaf index, offset in leaf) could replace the full 64-bit radix sort of N hierarchy keys. That means about two passes over N instead of eight 8-bit digit passes. It still produces exactly the same sorted unique key set.

### Gaps
- I could not read the Wang 2020 PDF text beyond the excerpts above. No timing of its partition update was extracted.
- I did not find any published Metal or Apple-GPU sparse-MPM benchmark.
- I did not check Taichi's or Genesis's successor "Quadrants" for a Metal sparse revival after Taichi 1.4.

## Q4. Options for metalsim: port the builder, use a fixed dense grid, or a GPU hash structure; tradeoffs for parity

### Takeaway
The parity-safe, highest-value option is to port Warp 1.16+'s own key-based builder (`volume_builder.cu`) to Metal. It needs:
- a keys-only 64-bit radix sort;
- RLE;
- an exclusive scan;
- about 19 one-thread-per-item kernels over PNanoVDB.

It produces byte-identical grids and voxel order by construction. Pair it with a stable Metal `radix_sort_pairs` for `compress_node_indices`, and target Newton's rebuildable-sparse mode so nothing syncs to the host.

A fixed or dense `Grid3D` avoids building any grid, but it is a different discretization: cell/node numbering and the partition differ. It only gives parity if the CUDA reference also runs `grid_type="fixed"`/`"dense"`. A pure hash grid is fastest to write but not parity-safe unless followed by a sort of the unique keys.

### Cited Findings
- **The portable part is already isolated.**
  - `RebuildGridDataBase` (buffer, offsets, capacities, `nanovdb::Map`) and 30 `CUDA_CALLABLE` helpers live in `volume_builder.h` and are compiled by both `.cu` and `.cpp`. The `.cu` adds only CUB calls, atomics and launch code (1266 lines in total). — [W118 volume_builder.h](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/volume_builder.h), [W118 volume_builder.cu](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/volume_builder.cu)
  - `volume_builder.h` still uses `nanovdb::Coord` and `nanovdb::Map` (53 `nanovdb::` references). These are C++ NanoVDB types that MSL would need replaced by PNanoVDB/plain structs. — local count in the same file
- **Primitive counts per build.** Sorts are 64-bit keys-only over N; RLEs run over N; there is one exclusive scan over the leaf count (u32→u64). Scratch is about 7×N×8 bytes, allocated and freed on each call. — [W118 volume_builder.cu](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/native/volume_builder.cu)

  | Build | 64-bit sorts | RLEs |
  |---|---|---|
  | Tiles build | 2 | 3 |
  | Active-voxel build | 2 | 4 |

- **Atomics used:** 64-bit `atomicOr` (mask words), int32 `atomicMin/Max` (bboxes), u32 `atomicAdd` (leaf counts). — same file. Metal has no 64-bit OR. — [Dawn MSL 64-bit atomics commit](https://dawn.googlesource.com/dawn/+/b620b2fe514a5ee2e6a2d924fd998c514019da87)
- **The rebuildable mode needs no host round trip.** Rebuildable volumes reserve capacity and report overflow via a status word. Newton's rebuildable mode calls `grid.rebuild(...)` with status recorded on the device. — [W118 types.py](https://github.com/NVIDIA/warp/blob/v1.18.0/warp/_src/types.py), [N152 solver_implicit_mpm.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)
- **Dense/fixed is a different discretization.** It is `fem.Grid3D` over a voxel-rounded bounding box, built from host-read bounds, rather than `fem.Nanogrid`. — [N152 solver_implicit_mpm.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)
- **The sort/scan primitives are covered in the sibling note.** A reduce-then-scan LSD radix sort is recommended because Apple GPUs lack forward-progress guarantees (Onesweep and decoupled look-back are unsafe). The estimate there is a few ms for about 2M 64-bit pairs on an M4 Max. — `research_notes/Metal GPU primitives for MPM speed/metal_sort_scan.md` (Apple Developer Forums thread 672532 cited there: https://developer.apple.com/forums/thread/672532)

### Inferences
**Option A: port Warp's key-based builder to Metal (recommended for parity)**
- *Work items.*
  1. A Metal keys-only u64 LSD radix sort, an RLE (flag, scan, scatter) and an exclusive scan. These are shared with the `warp.utils` port.
  2. Translate the roughly 19 kernels in `volume_builder.cu`, using `volume_builder.h` helpers rewritten without `nanovdb::Coord`/`Map`. The fork's `PNanoVDB.h` already compiles under `__METAL_VERSION__`.
  3. Replace 64-bit `atomicOr` with a 32-bit `atomic_fetch_or` on the word half containing the bit. This is exact because OR is bitwise.
  4. Keep the exact path's one count readback. For the rebuildable path, use capacities plus the status flag.
  5. Pool the scratch buffers. The CUDA code allocates and frees about 7N×8 bytes per call, which is costly on Metal if done through MTLBuffer creation.
- *Effort.* Roughly the size of the LBVH port already done in the fork, since it is a similar sort-plus-per-node-kernel structure. The primitives dominate the effort.
- *Parity.* Exact by construction: the output depends only on the set of unique keys. The CPU builder is a ready-made oracle for bitwise testing (compare `get_voxels()` and the raw grid buffer).
- *Cost reduction.* Keys-only sorts over N particles, plus about 8N for the node grid. With 16-bit or 8-bit digits, the sorts can skip the top bits that are always zero:
  - Upper keys use 63 bits.
  - Hierarchy keys use 36 bits plus log2(#upper). A hierarchy-key sort with a small upper count needs only about 40 significant bits, i.e. 5 × 8-bit passes instead of 8.

**Option B: fixed/dense `Grid3D`**
- *Speed.* Zero build cost and graph-friendly. Newton's example already captures it on CUDA.
- *Parity.* Only valid against a CUDA run that also uses `grid_type="fixed"`. Results are not comparable to sparse runs, because node numbering, partition and possibly boundary behavior differ.
- *Memory.* Bounded-domain only. The memory and active-subset partition (`ExplicitGeometryPartition`) still uses `compress_node_indices` sorts.

**Option C: GPU hash grid in the GSPGrid style**
- *Pros and cons.* Simplest O(N) build. But block or voxel indices depend on atomic insertion order, so it breaks parity and run-to-run determinism.
- *Fix.* Usable only as a deduplication front end: hash-insert the voxel/leaf keys, compact the unique keys, sort the small unique set, then build the NanoVDB levels. That reproduces NanoVDB's order while sorting U ≪ N keys. It is a reasonable second-stage optimization of Option A, not a replacement.

**Option D: keep the host builder but stop draining the GPU**
- *Change.* Record it as a capture host op, like the sorts.
- *Limit.* The CPU work remains (std::sort of about 9N keys per step), so this cannot get far below the current ~12 ms.

**Stability (separate from the volume builder)**
- The Metal `radix_sort_pairs` used by `compress_node_indices` and `_finalize_node_indices` must be stable, i.e. LSD with in-order scatter, to match CUB's documented stability. Otherwise per-cell particle order and partition node order diverge from CUDA, and the float summation order changes with them.

### Gaps
- I did not prototype or time any Metal sort/RLE for the specific key widths here (36-40 significant bits for hierarchy keys).
- No measurement splits the current 12-13 ms between host CPU work and GPU drains or synchronizations.
- I did not verify whether Apple GPUs' 32-bit `atomic_fetch_or` on device memory has any ordering caveats when two threads OR the two halves of the same 64-bit word. They are separate 32-bit locations, so it should be safe, but this was not tested.
