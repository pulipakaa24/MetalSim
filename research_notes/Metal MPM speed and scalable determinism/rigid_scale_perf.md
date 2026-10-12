# Rigid side at scale: Newton CollisionPipeline + MuJoCo Warp on Metal vs CUDA (Franka-Pour, 2,048 worlds)

Scope note: local code paths are relative to the worktree `_forks/` (`N` = `_forks/newton/newton/_src`, `W` = `_forks/warp/warp`, `MJW` = `_forks/mjw/mujoco_warp/_src`). The local Newton is the 1.5.x line (CHANGELOG top: `[1.5.2] - 2026-09-09`, `_forks/newton/CHANGELOG.md:7`) plus MetalSim patches (e.g. the Metal hashtable claim path). Several upstream collision-performance PRs merged into Newton `main` after 1.5.0 (2026-08-11) are **not** in this tree; this is flagged where it matters. Numbers marked "est." are my own arithmetic, not measurements.

Measured anchors for this scene (L4, Newton 1.5.2, Warp 1.16.0, captured, `max_triangle_pairs=8,388,608`) — `runs/mpm/scale_l4/rigid_off_2048.json`:
- 2,048 worlds, 188,417 shapes (92 per world + 1 global), 26,624 bodies, 43,008 dofs.
- `rigid_contact_max` = 14,053,376 (6,862 per world); actual `rigid_contacts` = 415,582 (~203 per world, 3.0% of capacity).
- `collide_ms` median 104.75; `substep_ms` median 10.99 (so one collide ≈ 9.5 MuJoCo substeps); 452 `wp.launch` per eager tick (collide + 3 substeps).
- With the default `max_triangle_pairs` (1e6) only 279,013 contacts were produced (`rigid_off_2048_tpdefault.json`): the triangle-pair / reducer buffer overflowed, i.e. this scene needs >1M triangle pairs at 2,048 worlds (~>500 per world).
- `nefc_mean` = 95.8 rows per world against `njmax` = 2,560 (3.7% of capacity); `nconmax` = 400 per world (`metalsim/physics/mpm/probes/scale_pour.py:661`).
- Collision meshes (MuJoCo Menagerie panda.xml): 11 `class="collision"` mesh geoms (`upstream/mujoco_menagerie/franka_emika_panda/panda.xml:134-217`); the STL ones have 200-300 triangles each (hand/link0/6/7: 200; link1-4: 300; counted from the binary STL headers). Newton's MJCF importer keeps them as triangle `GeoType.MESH` (`N/utils/import_mjcf.py:1227-1260`), whereas MuJoCo itself collides meshes as convex hulls.

## Q1. How Newton's CollisionPipeline runs on the GPU (local 1.5.2 code)

### Takeaway
One `collide()` is ~20 kernels plus 4 GPU scans, in fixed order. Almost all heavy kernels are **grid-stride loops with a fixed thread count** (32,768 threads; mesh tile kernels a fixed block count) that read the live count from a device counter, so they are *not* sized by capacity. The capacity-sized pieces are buffers (memory, and a few clears/scans), not the main compute launches. The mesh-vs-convex path turns each mesh-box pair into one candidate per BVH-overlapping triangle, runs GJK/MPR on each, and funnels every resulting contact through a global append counter and a global hashtable.

### Stage by stage (what launches, how sized)
- Construction sizing.
  - `rigid_contact_max` default: neighbor-budget heuristic, 40 contacts per mesh pair, 5 per primitive pair, 20 neighbors per shape, plane pairs per world (`N/sim/collide.py:296-401`). Here 6,862 per world.
  - Broad phase default is `"explicit"` (precomputed `model.shape_contact_pairs`) when none is given (`N/sim/collide.py:1009-1021`); `shape_pairs_max = len(shape_pairs_filtered)`. NxN/SAP use a per-world `n(n-1)/2` bound (`N/sim/collide.py:404-463`).
  - `max_triangle_pairs` default 1,000,000 (`N/sim/collide.py:768`) sizes both `triangle_pairs` (`vec3i`) and the `GlobalContactReducer` (`N/geometry/narrow_phase.py:1656-1662, 1701-1703`).
- Launch geometry (`N/geometry/narrow_phase.py:1739-1789`).
  - `block_dim = 128` on any GPU (`:1596`). `num_blocks = max(256, min(ceil(pairs/128), max_blocks_limit))`, with `max_blocks_limit = sm_count*4` on CUDA and **256 on everything else** → `total_num_threads = 32,768` on both L4 (58 SMs → 232, clamped up to 256) and Metal (est.).
  - Mesh-mesh / mesh-plane tile kernels: `target_blocks = sm_count*4 if device.is_cuda else 64` (`:1765`). On the L4 that is 232 blocks; on Metal (`is_cuda` False) it is **64 blocks** (×256 threads mesh-mesh, `MESH_SDF_BLOCK_DIM`, `N/geometry/sdf_contact.py:52`; ×512 mesh-plane, `narrow_phase.py:1587`). The Warp fork does report a Metal `sm_count` (= GPU core count, `W/_src/context.py:6105`) but Newton never reads it off CUDA.
- `collide()` sequence (`N/sim/collide.py:1296-1537`, `N/geometry/narrow_phase.py:1791-2240`):
  1. `compute_shape_aabbs`, `dim=shape_count` (188K); also zeros counters and bumps the contact generation (`collide.py:1357-1385`). `contacts.clear()` is only called when `clear_buffers=True` (default False, `N/sim/contacts.py:152`), so the 14M-slot contact arrays are **not** cleared per step.
  2. Broad phase. Explicit: one thread per precomputed pair, AABB test, atomic append to `broad_phase_shape_pairs` (`N/geometry/broad_phase_nxn.py:30-74, 504-506`). Pair order is therefore nondeterministic.
  3. `_counter_array.zero_()` (one small memset, `narrow_phase.py:1852`).
  4. Primitive kernel (32,768 threads, grid-stride over `candidate_pair_count`): analytic pairs, routes the rest into GJK / mesh / mesh-plane / mesh-mesh / SDF lists with atomic appends (`narrow_phase.py:1857-1891`, loop `:204`).
  5. GJK/MPR kernel for convex pairs (box-box cups/table here), same 32,768 threads (`:1896-1916`); lean support map when no capsule/cylinder/cone/ellipsoid (`:1602-1612`).
  6. Midphase `narrow_phase_find_mesh_triangle_overlaps_kernel`, `dim=[256, 128]`, `block_dim=128`: one threadgroup per mesh-convex pair (grid-stride over pairs), cooperative `wp.tile_mesh_query_aabb` of the convex shape's tight AABB (+contact threshold) against the mesh BVH; per-tile scan and **one atomic per tile** on `triangle_pairs_count` (`narrow_phase.py:800-913`; `N/geometry/collision_core.py:1003-1052`). Every triangle whose BVH leaf box overlaps the convex's AABB becomes a candidate. When a large box (table, cup wall) overlaps a link's AABB, that can be all 200-300 triangles of the link (est.).
  7. Reducer `clear_active()`: 2 small launches, 1,024 threads, clear only the previously active hashtable entries (`N/geometry/contact_reduction_global.py:990-1043`, kernel `:720-798`).
  8. Mesh-plane: `compute_mesh_plane_block_offsets_scan` = 2 kernels with `dim = max_candidate_pairs+1` plus 2 `array_scan`s over that capacity (`narrow_phase.py:1054-1098`), then a tiled kernel over 64 (Metal) / 232 (L4) blocks of 512 that loops over **all vertices** of each mesh in a mesh-plane pair (`:1101-1360`).
  9. `mesh_triangle_contacts_to_reducer_kernel`, 32,768 threads, grid-stride over `triangle_pairs_count`: per triangle, GJK/MPR (triangle vs convex) and an append into the reducer buffer (`narrow_phase.py:2015-2035`; non-reduce variant `:916-1025`). Each appended contact takes one `wp.atomic_add(reducer_data.contact_count, 0, 1)` on a **single global counter** (`contact_reduction_global.py:1102-1104`).
  10. `reduce_buffered_contacts_kernel`, 32,768 threads, grid-stride over buffered contacts: hashtable insert per (shape pair, normal bin / voxel) key and `atomic_max` of packed 64-bit values into `NUM_SPATIAL_DIRECTIONS+1 = 7` slots (`contact_reduction_global.py:1507-1542, 156-211`; `N/geometry/contact_reduction.py:73-94`).
  11. Mesh-mesh (SDF, or BVH fallback when the mesh has no SDF): same capacity-sized offset scan, then a tiled kernel over 64/232 blocks of 256 with cooperative edge culling (`narrow_phase.py:2086-2161`; `N/geometry/sdf_contact.py:886-928, 960-1110`). Self-collisions are on by default in the MJCF importer (`import_mjcf.py:203`). Whether any Franka link pairs pass the AABB test here is **unverified**.
  12. `exported_flags.zero_()` over the full reducer capacity (`narrow_phase.py:2166`; array sized `capacity` = `max_triangle_pairs`, `contact_reduction_global.py:932`), then `export_reduced_contacts_kernel` (32,768 threads, grid-stride over active hashtable slots, `atomic_add` dedup on `exported_flags[contact_id]`, writes final contacts with another global append) (`contact_reduction_global.py:1617-1713`).
  13. `verify_narrow_phase_buffers`, `dim=[1]` (default on, `narrow_phase.py:2205-2240`).
  14. Deterministic mode only: `ContactSorter.sort_full` = prepare kernel over `n = rigid_contact_max`, `radix_sort_pairs` of int64 keys over **the full capacity**, then backup and gather kernels over `n` (`N/geometry/contact_sort.py:436-445`, `:392-393`). The error text says this is deliberate for CUDA-graph capture (`collide.py:1461-1468`). Here n = 14,053,376 against ~415K live contacts.

### Capacity vs actual-count summary
| Item | Sized by | Per step at 2,048 worlds (est.) |
|---|---|---|
| Primitive / GJK / tri-contact / reduce / export kernels | fixed 32,768 threads, loop to device count | proportional to live work |
| Midphase | 256 tiles × 128, loop over live mesh pairs | live work |
| Mesh-plane / mesh-mesh tile kernels | fixed 64 (Metal) / 232 (L4) blocks | live work, but fewer resident threads on Metal |
| Mesh-plane + mesh-mesh offset scans | `max_candidate_pairs+1` (4 kernels + 4 scans) | capacity |
| `exported_flags.zero_()` | `max_triangle_pairs` int32 = 33.5 MB | capacity |
| Hashtable clear | active entries only | live |
| Deterministic sort | `rigid_contact_max` = 14.05M int64 keys + 13 arrays gathered | capacity |
| Memory | `triangle_pairs` 8.39M × 12 B = 101 MB; reducer buffers ~40 B × 8.39M = 336 MB; hashtable 2,097,152 keys (16.8 MB) + values 2,097,152 × 7 × 8 B = 117 MB | — |

### Upstream status (Newton main after 1.5.x; from the GitHub API)
- **Merged on main, absent here:**
  - [#3918 "Improve collision kernel performance"](https://github.com/newton-physics/newton/pull/3918) (merged 2026-08-18): fewer launches/syncs, "sparse candidate routing for large CUDA workloads", optional mesh work-buffer bounds.
  - [#4063 "Accelerate collision detection"](https://github.com/newton-physics/newton/pull/4063) (2026-08-31): bounded tile sorting; fewer radix passes for deterministic/persistent contact sort by "packing only the shape-index and contact-family bits required".
  - [#4258](https://github.com/newton-physics/newton/pull/4258) (2026-09-18): `num_blocks` was sized from `block_dim`, so the narrow block launched "a quarter of the threads"; ~46k-body / 195k-contact scene collide ~10.4 ms (pre-#3918) → ~2.2 ms.
  - [#4142 "Improve hydroelastic contact performance"](https://github.com/newton-physics/newton/pull/4142) (2026-09-04): mesh-SDF kernels scan "only active records" instead of capacity.
  - [#3801 "Improve sdf collision performance"](https://github.com/newton-physics/newton/pull/3801) (2026-08-10): nut-and-bolt at 2,000 worlds, 475 → 291 ms. Collide grows ~linearly with worlds.
- **Sizing/correctness:**
  - [#4604](https://github.com/newton-physics/newton/issues/4604): the fixed 1M `max_triangle_pairs` overflows at ~1,250 worlds and drops contacts by atomic allocation order. [#4605](https://github.com/newton-physics/newton/pull/4605) (merged 2026-10-09) sizes it from the model (`participants*6/2` pairs per world × 128 candidates, min 1M, warns at ≥256 MiB).
  - [#4626](https://github.com/newton-physics/newton/issues/4626) (open): per-world partitioning of the reducer and triangle-pair buffer. One global atomic counter lets worlds starve each other, and the deterministic packing caps a batch at 2^20 = 1,048,575 candidates.
  - [#4472](https://github.com/newton-physics/newton/pull/4472) reports reducer overflow. [#4476](https://github.com/newton-physics/newton/pull/4476) skips mesh stages for shapes in no pair.
  - [#4365](https://github.com/newton-physics/newton/issues/4365): 65,536 humanoid worlds, hashtable capacity 33.5M with 14.6% used.
- **Memory:** [#2384](https://github.com/newton-physics/newton/pull/2384) per-world `shape_pairs_max` (~975× smaller); [#3402](https://github.com/newton-physics/newton/issues/3402)/[#4095](https://github.com/newton-physics/newton/pull/4095) ~64-96 B per `rigid_contact_max` slot, warning ≥256 MiB.
- **Determinism:** [#2409](https://github.com/newton-physics/newton/pull/2409) adds the fingerprint tiebreak in the `atomic_max` packing plus one radix sort + gather per frame. The docs call the overhead "small" with no timing.
- **Docs ([collisions.rst](https://github.com/newton-physics/newton/blob/main/docs/concepts/collisions.rst)):** EXPLICIT for few pairs, SAP for >100 shapes. Mesh-mesh BVH fallback is "significantly slower… scales with triangle count". A coarse SDF (`max_resolution=64`) is "dramatically faster". Graph-capture `collide`; colliding once per frame "can significantly improve performance".
- **Warp:** [NVIDIA/warp#742](https://github.com/NVIDIA/warp/issues/742) (open, by Newton's collision author) asks for device-array launch dims ("indirect dispatch") because contact counts vary per frame. Upstream Warp has none, so Newton uses grid-stride loops. [#1270](https://github.com/NVIDIA/warp/issues/1270): Warp kernels already grid-stride by default.

## Q2. GPU performance characteristics that apply

- **Single-address append counters.** `triangle_pairs_count` is tile-aggregated (one atomic per 128-lane tile, `collision_core.py:1041-1047`). The reducer's `contact_count` (`contact_reduction_global.py:1103`), the final contact writer, and the broad-phase/primitive routing counters take **one atomic per item**. NVIDIA's compiler automatically warp-aggregates such same-address atomics in many cases since CUDA 9 ([NVIDIA blog, warp-aggregated atomics](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)). I found no statement that Apple's Metal compiler does the same. Apple recommends SIMD-group reduce → threadgroup atomic → one device atomic per threadgroup, and says global atomics are "coherent across the whole GPU" with contention growing with core count ([Apple Tech Talk 10580](https://developer.apple.com/videos/play/tech-talks/10580/); [WWDC22 10159](https://developer.apple.com/videos/play/wwdc2022/10159/)). No public Apple atomic-throughput numbers exist ([metal-benchmarks](https://github.com/philipturner/metal-benchmarks) has none). **Speculation:** non-aggregated hot counters cost relatively more on Metal than on the L4.
- **Hashtable `atomic_max` on 64-bit values.** MSL allows only non-returning `atomic_max/min_explicit` on `device atomic_ulong` ([MSL spec 4.1 §6.16.4.6](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)). The Feature Set Tables say "64-bit atomics" from Apple9 ([Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)), but Apple DTS confirms compare-exchange on `atomic_ulong` does not compile ([forum 848400](https://developer.apple.com/forums/thread/848400)). metal-benchmarks: M2 added a single non-returning UInt64 min/max instruction; emulating it costs 2.5× bandwidth and 5× latency.
  - Local consequence: the Metal hashtable insert is a 32-bit CAS claim on the key's high word, then a 64-bit `atomic_min` to publish, and readers spin on "claimed" slots (`N/geometry/hashtable.py:106-227`). That is ≥2 device atomics plus extra loads per new key, vs one 64-bit CAS on CUDA (`:255-288`). The Warp fork makes 64-bit `atomic_cas` raise on Metal rather than silently racing (`_forks/warp/METALSIM_CHANGES.md`, "integrated" row).
- **Clearing capacity-sized buffers.** Warp's Metal `memset` is a compute kernel, one thread per 16 bytes (`W/native/metal.mm:723-745, 2077-2096`), not a blit `fillBuffer`. 33.5 MB `exported_flags` ≈ 2.1M threads. At 300-546 GB/s that is ~60-110 µs per collide (est.): small next to 105-195 ms. Apple publishes no blit-fill vs kernel-clear comparison. Encoder switches (blit ↔ compute) cost extra ([Metal Performance HUD docs](https://developer.apple.com/documentation/xcode/gaining-performance-insights-with-metal-performance-hud); [MPS tuning hints](https://developer.apple.com/documentation/metalperformanceshaders/tuning-hints)), so the kernel clear avoids that.
- **Grid-stride vs one-thread-per-capacity.** A grid-stride loop costs about the same instructions as the bounds check ([NVIDIA grid-stride pro tip](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)), and it avoids the per-threadgroup ramp of large idle grids. MuJoCo Warp measured this: [#1464](https://github.com/google-deepmind/mujoco_warp/pull/1464) convex CCD from one thread per `naconmax` slot to grid-stride gave 1.57-1.88× on the kernel. The flip side is that a fixed grid can **under-fill** the device. Newton's fixed 32,768 threads assume an NVIDIA SM count, and its 64-block mesh fallback is a CPU-oriented number (`narrow_phase.py:1759-1765`). Apple's rule of thumb is 1-2K concurrent threads per GPU core ([WWDC22 10159](https://developer.apple.com/videos/play/wwdc2022/10159/)), so 40-80K threads for a 40-core M4 Max. Mesh-plane at 64×512 = 32K and mesh-mesh at 64×256 = 16K are below that (est.).
- **BVH traversal divergence.** In the midphase, a threadgroup traverses one mesh BVH cooperatively (`wp.tile_mesh_query_aabb`), so divergence is within a tile, not across unrelated queries. Per-thread `mesh_query_aabb` is the non-tile fallback (`collision_core.py:1053-1062`). Warp's [#1840](https://github.com/NVIDIA/warp/issues/1840) gained 5-9% on mesh AABB queries (up to 21% for packed-leaf BVH queries) from candidate-bound loads, while the sphere/capsule variants regressed from register pressure. [#1529](https://github.com/NVIDIA/warp/issues/1529) adds nearer-child-first ray traversal. The Metal fork builds the same LBVH as CUDA (`METALSIM_CHANGES.md`, `67f3407a`), so tree shape matches across devices.
- **The heavy part is per-triangle GJK/MPR.** Each candidate triangle runs a full GJK/MPR plus triangle-prism refinement (`narrow_phase.py:1004-1023`). With >1M (likely several million, est.) candidates per collide, this kernel and the reducer insert are the probable top two. **Unverified:** no per-kernel profile of `collide` exists in the repo yet (`runs/gpu_queue/*rig_kprof_smoke64*` was queued, no output).

## Q3. Metal specifics

- **Idle grids.**
  - Fork measurement: 16.8M early-exit threads ≈ 400 µs per launch on the M4 Max; the bounded indirect dispatch of the 160K live threads ≈ 10 µs (`METALSIM_CHANGES.md`, `62526460`). A forum report on an unknown Apple GPU agrees: ~16.8M mostly-dead threads ≈ 600 µs vs 25 µs ([Apple forum 750363](https://developer.apple.com/forums/thread/750363)).
  - Scaling, est.: ~24 ns per 1K dead threads. A `(2048, 2560)` = 5.24M-thread MJW launch ≈ 125 µs of pure overhead; a `naconmax` = 819,200 launch ≈ 20 µs.
- **Indirect dispatch.**
  - `dispatchThreadgroups(indirectBuffer:...)` reads `MTLDispatchThreadgroupsIndirectArguments{threadgroupsPerGrid[3]}` on the GPU; threads per threadgroup is fixed by the CPU ([Apple docs](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups(indirectbuffer:indirectbufferoffset:threadsperthreadgroup:))). "Indirect draw and dispatch arguments" is supported from Apple3 ([Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)).
  - Metal 4 adds an indirect `dispatchThreads` ([MTL4ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtl4computecommandencoder)).
  - The fork exposes this as `wp.launch(dim_limit=arr)`: a 1-thread prep kernel writes the threadgroup count, recorded into graphs and replayed outside the ICB, splitting indirect execution ranges (`W/_src/context.py:11459-11540`; `METALSIM_CHANGES.md`, `62526460`). Other devices run the full `dim`.
- **ICBs.**
  - Compute ICBs and in-ICB barriers are supported from Apple3, and "Maximum indirect command buffer length: No limit" ([Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)).
  - In an ICB "dispatches are always concurrent", so ordering needs `set_barrier`. The GPU can encode `concurrent_dispatch_threadgroups` itself ([MSL spec 4.1 §6.17.2](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)). `executeCommandsInBuffer(_:indirectBuffer:offset:)` takes a GPU-supplied range.
  - Apple publishes no replay-cost figure ([WWDC19 601](https://developer.apple.com/videos/play/wwdc2019/601/)). Fork measurements:
    - command-buffer boundary ≈ 26 µs of GPU time, now batched at 1,024 dispatches per command buffer (`ea6dcfd`);
    - zero-work dispatch with 19 array arguments 7.1 → 2.4 µs with device-pointer args (`08084e0`), not on by default (`metal_device_args`, "integrated" row).
  - At ~30 dispatches per collide that is <0.1 ms (est.), so dispatch count is not the problem for collide.
- **Atomics.**
  - `atomic_int/uint` (Metal 1), `atomic_ulong` (2.4, max/min only, device only, void-returning), `atomic_float` (Metal 3; add/sub device-only until MSL 4.1 adds threadgroup). "Floating-point atomics" from Apple7. Fetch-modify ops were relaxed-only; MSL 4.1 / Apple7+ add acquire/release ([MSL spec 4.1 §2.6, §6.16.4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf); [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)).
  - No device vs threadgroup atomic throughput numbers are published. Apple's guidance is to aggregate before going to device memory (see Q2).
- **Bandwidth, compute and caches.**
  - M4 Max: 546 GB/s on a 512-bit LPDDR5X bus ([Notebookcheck](https://www.notebookcheck.net/Apple-M4-Max-official-with-incremental-upgrades-over-the-M3-Max.911098.0.html)); ~16-18 TFLOPS FP32 (third-party, clock-dependent; [flopper.io](https://flopper.io/gpu/apple-m4-max)). M4 Max SLC size is not published; M1/M2 Max were 48 MB, GPU L2 ~0.5-1 MB, 8 KB L1D per core ([metal-benchmarks](https://github.com/philipturner/metal-benchmarks)).
  - L4: 30.3 TFLOPS FP32, 300 GB/s ([NVIDIA L4](https://www.nvidia.com/en-us/data-center/l4/)); 48 MB L2 (AD104, secondary: [TweakTown](https://www.tweaktown.com/news/88604/nvidia-details-ad102-ad103-ad104-gpu-specs-transistors-rop-counts/)).
  - Net: the L4 has ~1.8× the FP32 and the M4 Max ~1.8× the DRAM bandwidth, so GJK/MPR-bound collide favours the L4. The observed ratio is 195/105 = 1.86×, the same as the FP32 ratio (est.; consistent with collide being ALU-bound, **not proven**).
- **Buffer clears.** Warp-Metal clears are kernels (above). Metal 4 folds fill/copy/dispatch into one compute encoder ([WWDC25 254](https://developer.apple.com/videos/play/wwdc2025/254/)). No public fill-vs-kernel benchmark exists.

## Q4. MuJoCo Warp: capacity-sized launches per step

- **Capacities.** `nconmax`/`njmax` are per world; `naconmax = nconmax*nworld` (`MJW/io.py:1537-1542, 1708`). efc arrays are `(nworld, njmax)` (`MJW/types.py:2006-2038`). Newton passes `nconmax=400, njmax=2560` → `naconmax = 819,200`, efc grid 5,242,880 (`N/solvers/mujoco/solver_mujoco.py:7463-7470`).
- **With Newton contacts (`use_mujoco_contacts=False`)** the MJW collision driver is off (`solver_mujoco.py:3800`). Each step runs:
  - `convert_newton_contacts_to_mjwarp_kernel` with `dim = min(rigid_contact_max, naconmax)` = 819,200, early exit past `rigid_contact_count`; a fast path when the contact generation is unchanged (`solver_mujoco.py:4261-4398`; `N/solvers/mujoco/kernels.py:387-720`);
  - `mjw_data.nacon.zero_()` (`:4305`).
- **make_constraint.** Contact rows are launched at `dim=naconmax` or `(naconmax, nmaxdim)` with `if conid >= nacon_in[0]: return` (`MJW/constraint.py:5325-5630`; exits at `:2678, 2810, 3129, 3298, 4227, 4383`). Dense contact Jacobian loops to `min(nefc, njmax)` (`:3785, 3929`). `d.efc.Jqvel.zero_()` over the full `(nworld, njmax)` on the dense path (`:5493`).
- **Solver.** Many kernels at `(nworld, njmax)` or `(nworld, njmax, threads_per_efc)` guarded by `efcid >= nefc_in[worldid]` (`MJW/solver.py:1500-1509, 2037-2058, 3904-3927, 4352-4362`). Off CUDA the elliptic JTCJ uses `dim_block = naconmax` (`:3388`; [issue #1704](https://github.com/google-deepmind/mujoco_warp/issues/1704): 33.6M threads per Newton iteration for Go2 at 4,096 worlds, >99% exit immediately; fix [PR #1715](https://github.com/google-deepmind/mujoco_warp/pull/1715)). The iteration loop is `capture_while` (`:3984-3992`), so launch dims stay at capacity.
- **Upstream work bounding launches.**
  - [#1464](https://github.com/google-deepmind/mujoco_warp/pull/1464) (merged): CCD grid-stride, 1.18-1.37× overall on Aloha.
  - [#1722](https://github.com/google-deepmind/mujoco_warp/pull/1722) (open draft, consolidates #1638/#1642/#1643): occupancy-sized launches over active efc rows; constraint construction 4.7-32% faster at 512-8,192 worlds, solver 1-2% slower at ≥2,048.
  - [#1769](https://github.com/google-deepmind/mujoco_warp/pull/1769) (open): NxN broadphase capped at 8 occupancy waves.
  - [#1549](https://github.com/google-deepmind/mujoco_warp/pull/1549) (merged): Warp "lean" launches, Aloha +9.7%.
  - [#1385](https://github.com/google-deepmind/mujoco_warp/pull/1385) (open) / [#926](https://github.com/google-deepmind/mujoco_warp/issues/926): per-step `wp.empty(naconmax)` scratch causes fragmentation and OOM.
  - No upstream MJW use of indirect dispatch. The upstream strategy is grid-stride over device counts.
- **This scene.** nefc ≈ 96 of 2,560 (3.7%) and ~203 contacts of 400. Each `(nworld, njmax)` launch is ~96% dead threads. At ~125 µs of overhead per 5.24M-thread launch (est., from the fork's 400 µs / 16.8M figure), a handful of such launches per solver iteration × iterations × 3 substeps can reach several ms per tick on Metal (**speculation**; count the launches with the fork's `wp_metal_counters` before acting).

## Implications for our scene (ranked by expected payoff; all results-preserving unless noted)

1. **Profile `collide` per kernel on both devices first.** The 1.86× Metal/L4 ratio matches the FP32 ratio, which suggests the per-triangle GJK/MPR (`mesh_triangle_contacts_to_reducer_kernel`) and the reducer insert dominate (speculation). Read the device counters `triangle_pairs_count`, `shape_pairs_mesh_mesh_count`, `shape_pairs_mesh_plane_count` and `reducer.contact_count` (`narrow_phase.py:1684-1693`) once at 2,048 worlds to learn the real work sizes. Everything below should be re-ranked by that profile.
2. **Give Metal the CUDA launch geometry for the mesh tile kernels and the fixed grid.** Use the fork's `device.sm_count` (40) instead of the non-CUDA constants in `narrow_phase.py:1745-1765`: mesh-plane/mesh-mesh would get 160 blocks instead of 64, and `total_num_threads` could scale to ~1-2K threads per core (40-80K) instead of 32,768 ([WWDC22 10159](https://developer.apple.com/videos/play/wwdc2022/10159/)). Pure scheduling change.
   - Bitwise: contacts flow through atomics already. Fast mode is run-to-run nondeterministic either way, and deterministic mode is invariant to block count.
   - Expected payoff: large if the mesh-plane/mesh-mesh kernels are significant, otherwise small (speculation).
3. **Aggregate the hot append counters per SIMD group / threadgroup on Metal.** Targets: reducer `contact_count` (`contact_reduction_global.py:1103`), final contact writer, primitive-kernel routing. Use the tile-scan pattern Newton already uses for `triangle_pairs_count` (`collision_core.py:1041-1047`), or `simd_prefix_exclusive_sum` + one atomic per SIMD group as Apple recommends ([Tech Talk 10580](https://developer.apple.com/videos/play/tech-talks/10580/)).
   - Results: contact IDs change order. That is already nondeterministic in fast mode, and deterministic mode resolves by fingerprint and then sorts ([#2409](https://github.com/newton-physics/newton/pull/2409)).
   - Payoff unknown, potentially large given millions of appends per collide on one address (speculation; measure with an atomic microbenchmark like `scale_pour.py --part ordered`).
4. **Backport (or rebase onto) Newton main's collision-performance PRs.** [#3918](https://github.com/newton-physics/newton/pull/3918) (sparse candidate routing, fewer launches), [#4258](https://github.com/newton-physics/newton/pull/4258) (block-count fix; ~10.4 → 2.2 ms on a 195K-contact scene), [#4063](https://github.com/newton-physics/newton/pull/4063) (fewer deterministic sort passes), [#4142](https://github.com/newton-physics/newton/pull/4142) (active-record scans), [#4605](https://github.com/newton-physics/newton/pull/4605) (model-derived `max_triangle_pairs`). They are intended to be results-preserving. Check the "CUDA-only" branches (#3918 says sparse routing is for CUDA, atomic fallback elsewhere), which may again fall back on Metal.
5. **Deterministic mode only: bound the contact sort by the live count.** `sort_full` radix-sorts 14.05M int64 keys and gathers 13 arrays over full capacity every collide (`contact_sort.py:436-445`) for ~415K live contacts (3%). The fork already has `count_limit` on `radix_sort_pairs` (`W/_src/metal_primitives.py:944-998`) and `dim_limit` launches. Using them with `dim_limit = rigid_contact_count` gives identical output (the sentinel keys of the unused tail sort last either way). Expected payoff: large in `--pipeline-det` runs (est. tens of ms of memory traffic at 14M × (8+4) B × passes), none otherwise.
6. **Bound the remaining capacity-sized collide work with `dim_limit`.**
   - `exported_flags.zero_()`: only `[0, reducer.contact_count)` is ever read (`contact_reduction_global.py:1713`), so a bounded clear-kernel is equivalent; ~0.1 ms est.
   - The 4 offset-kernels + 4 scans over `max_candidate_pairs+1` (`narrow_phase.py:1054-1098`, `sdf_contact.py:886-928`): bound to the pair counts.
   - Small payoff each (≤ ~1 ms total, est.).
7. **MuJoCo side: `dim_limit` on the `(nworld, njmax)` / `naconmax` launches**, as the fork already did for `warp.fem`. Gates are `nacon`, and per-world `nefc`, which needs a world-major bound (`max nefc` over worlds via `atomic_max`, then `dim=(njmax_bound, nworld)` reordering), or the grid-stride approach of [#1722](https://github.com/google-deepmind/mujoco_warp/pull/1722). Also use #1715 for the off-CUDA JTCJ `dim_block`. Bitwise-safe when the per-thread work is unchanged. Payoff: several ms per tick (speculation, see Q4); this is about the substep, not `collide`.
8. **Algorithmic, not results-preserving (flag only).**
   - (a) Convex-hull or SDF collision for the Franka links: `ModelBuilder.approximate_meshes("convex_hull")` (`N/sim/builder.py:7270`) matches what MuJoCo itself does with mesh geoms, and replaces ~hundreds of triangle MPRs per pair with one GJK/MPR.
   - (b) Collide once per policy step rather than per coupled step, if the parity target allows (docs; [#4281](https://github.com/newton-physics/newton/pull/4281)).
   - Both change contacts, so they need a parity decision; they are likely the biggest wins of all.
9. **Not worth pursuing for collide:** dispatch count/ICB tuning (~30 dispatches ≈ <0.1 ms) and blit-vs-kernel clears (no evidence either way; the kernel clear avoids encoder switches).

## Measured (2026-10-11): what the rigid tick actually spends, and what was fixed

Per-kernel GPU time from `scale_pour.py --kprof` (Metal: `WP_METAL_PROFILE=1`, one command buffer per dispatch; L4: nsys `cuda_gpu_kern_sum`). Franka-Pour, 2,048 worlds, `max_triangle_pairs` 8,388,608, det off. Raw data: `runs/rigid_perf/` in the agent worktree (`l4/`, `out/`, `out/pre_crash_measurements.txt`).

Early point (3 warm-up ticks, 81,920 contacts), ms per collide:

| kernel | L4 | Metal before | Metal after |
|---|---|---|---|
| mesh_sdf_collision_global_reduce (mesh-mesh SDF, BVH fallback) | 87.7 | 109.4 | 63.7 |
| narrow_phase_find_mesh_triangle_overlaps (mesh-convex midphase) | 0.40 | 65.0 | 0.25 |
| mesh_triangle_contacts_to_reducer | 1.25 | 1.74 | 1.20 |
| mesh_plane_contacts_reduce | 0.77 | 1.63 | 0.92 |
| everything else in collide | ~1.0 | ~1.6 | ~1.4 |
| collide wall | 90.5 | 178.9 | 67-68 |

Late point (70 warm-up ticks, ~418K contacts), eager tick (1 collide + 3 substeps):

| kernel | L4 | Metal before | Metal after |
|---|---|---|---|
| midphase | 3.28 | 333.6 | 1.05 |
| mesh-mesh SDF | 86.9 | 110.7 | 67.7 |
| mesh_triangle_contacts_to_reducer | 23.0 | 25.9 | 20.6 |
| MuJoCo Warp `_update_constraint_efc` (19.5 launches) | 6.85 | 5.2 | 5.2 |
| eager tick wall / captured tick | 150.6 / 128.0 | 523.5 / 507.2 | 129.6 / 112.1 |

Root causes:
- **Midphase on Metal was serial per block.** `wp.tile_mesh_query_aabb` has a cooperative implementation only under `__CUDA_ARCH__` (`_forks/warp/warp/native/tile_mesh.h:13-330`). On Metal it falls back to the CPU branch: every thread of the 128-thread block repeats the whole BVH traversal, and only lane 0's single triangle is emitted per block-wide step. Each step costs a tile scan and barriers. The cost grows with the number of overlapping triangles, from 65 ms early in the trajectory to 334 ms late. Fix (Newton): on Metal, one thread per pair calling `wp.mesh_query_aabb`, which finds the same triangles.
- **Mesh tile kernels were under-launched off CUDA.** The block count is `sm_count*4` only `if device.is_cuda`, else 64 (`narrow_phase.py`). Fix: Metal launches `sm_count*32` threadgroups with the same work split. Collide by threadgroups per core: 1.6: 175 ms, 4: 80, 8: 71, 16: 71, 32: 67, 64: 66.
- **Barrier race in the mesh-SDF kernels (hangs).** The edge-culling loop's exit read of the shared `progress` counter has no barrier before thread 0 resets it for the next mode or combo. A lagging thread re-enters the loop alone and its barriers pair with the wrong ones. On Metal it hung the GPU: 1,280 threadgroups at 2,048 worlds hung reproducibly within 20 collides, and a hang at 160 threadgroups ran unguarded for about 2 h before the Mac panicked (WindowServer watchdog). Fix: a `tile_stack_clear` (a pure barrier at that point) before each reset. The same configuration then ran 40 collides cleanly. Killing a hung process does free the GPU.
- **Deterministic contact sort over capacity.** `ContactSorter` radix-sorts and gathers all `rigid_contact_max` = 14.05M slots: 12.4 ms per collide on Metal. Fix: on Metal Warp builds with `dim_limit`, stop the prepare, sort (`metal_primitives.radix_sort_pairs(count_limit=)`) and gather at `contact_count`: 0.2 ms.
- **"Captured tick > eager" was trajectory drift, not a capture cost.** `scale_pour.py --mode capture` times collide (ticks ~6-26), then tick (~46-66), then the graph (~68-88). Collide cost on the old code grows sharply with contact count, because of the midphase above. At one trajectory point, captured ≈ eager or less: base 507 vs 524 ms, final 112 vs 130 ms per tick after 70 warm-up ticks. Graph replay of this tick has `host_ops` (one `capture_while` per substep) and takes the non-ICB path with about 11 host waits per tick. That costs a few ms, not hundreds.
