# Coupled MPM scenes on Metal vs the L4: research before building (2026-10-13)

Context: `metalsim/physics/mpm/MPM.md` 8j. Captured two-way is 1.76x the L4's time and captured ANYmal 1.29x. This
note covers four questions and records what is verified from code or measurement versus what is inferred. Batches and
logs are in `runs/mpm/coupled_speed/`.

## A. The two-way growth: the rigidity product IJtm

**Verified (code).**
- Newton's coupled solve applies the rigidity operator twice in every iteration:
  `delta_body_qd = -IJtm @ delta_impulse`, then `collider_velocity += J @ delta_body_qd`.
  - `apply_rigidity_operator`: `solve_rheology.py:321`. The matrices come from `build_rigidity_operator`,
    `rasterized_collisions.py:792`.
  - J has one block row per grid node, with at most 3 blocks (`max_blocks_per_node = 3`).
  - IJtm has one block row per rigid-body block column (2 per body: linear and angular). Each of its rows holds one
    block per collider node of that body.
- Warp's `bsr_mv` chooses between two kernels (`warp/_src/sparse.py`, `bsr_mv`). The tiled kernel is used when
  `nnz * block_size > 2 * tile_size * rows`, with tile_size 64, but only for `device.is_cuda`. Everything else gets one
  thread per (row, sub-row), summing the row's blocks serially.
  - The L4's nsys profile shows exactly one tiled and one untiled product per iteration:
    `runs/mpm/profile/kernels_l4/twoway_coupling*_cuda_gpu_kern_sum.csv` (204 launches each over 4 frames).
- On Metal, Newton's `_ListedRowsMv` (fork commit `80e79684`) uses the untiled body for both products.
- Warp's Metal `tile_sum` (`warp/native/tile_reduce.h`, `metal_reduce::block_reduce`) uses the same per-SIMD-group
  shuffle-down tree as CUDA's `warp_reduce`. It then folds the group partials in index order, as CUDA's
  `block_combine_thread_results` does with its warp partials.
  - So with the same kernel choice, Metal sums the long rows in the L4's order: 64 strided lane sums, then the tree.
  - FMA placement follows `fuse_fp` on both, as in every other kernel.

**Measured** (batch_02, `rigidity_mv_probe.py`: eager, synchronized, two-way frames 120-149):
- IJtm has 12 rows. The longest row grows to about 1,170 blocks as the boxes sink into the sand.
- Per 50 calls: 29-40 ms untiled vs 12.1-12.9 ms tiled. The figures include a host synchronization per call.
- J (655,360 rows, at most 2 blocks each) is unchanged, at about 9.3-11.8 ms.

**Consequence.** The fix is to make CUDA's kernel choice on Metal, in Warp (`warp.config.metal_bsr_mv_tiles`). Newton's
listed-rows path then defers to `bsr_mv` for matrices that would be tiled.
- Two-way changes rounding: it now follows the L4's order.
- Pure-MPM scenes and ANYmal do not use `bsr_mv` and stay bitwise (batch_03: floors bitwise equal; two-way repeatable
  and still inside the L4 ensemble at 0.83).

## B. test_coupled_contact_filter_keeps_source_order: a test-harness artifact, not a Warp or Newton bug

**Cause (verified, batch_05 logs).**
- Every pytest run in batches 02-05 failed, alone or not, with any kernel cache: `contact_filter/*.log` shows
  "1 failed" in all 19 runs.
- The same code run as a script passed (`coupled_det_diag_*.log`).
- Those pytest runs were `python -m pytest` launched by `scripts/gpu_run.sh`, which starts every job in the MAIN
  checkout. `python -m` puts the current directory ahead of `PYTHONPATH` on `sys.path`, so they imported the main
  checkout's `metalsim`.
  - Its `newton_compat.py` is an older copy (other sessions' working tree) without `_set_coupled_determinism`
    (`diff` shows the whole RUN_TO_RUN coupled-module block missing).
- The docs-audit run that passed ran pytest from inside its own worktree.
- Script jobs (`python path/to/script.py`) are not affected: `sys.path[0]` is the script's directory.
- Fix in the batches: `python -P -m pytest` (Python 3.11+: do not prepend the cwd). batch_06 verifies.
- **Every agent launching `python -m ...` through gpu_run.sh from a worktree hits this.**

**Kept from the reading (Warp fork 95f6f121, `warp/_src/context.py`)**, since the module-option machinery is shared,
e.g. by `BatchSimOptions(deterministic=True)`:
- `set_module_options` updates the options and calls `mark_modified()` (clears hashers, keeps execs).
- `Module.load` re-hashes and recompiles on a mismatch. The deterministic mode is part of the hash and salts it.
- Kernel mangled names follow `kernel.hash`.
- Deterministic launch metadata is snapshotted at load: from the compile's adjoints after a fresh build, rebuilt with
  the Metal lowering after a cache hit.
- So changing the mode after a module was loaded selects a new binary.
- Warp's docs only say to set the option before the module compiles. Changing it afterwards is undocumented, and no
  upstream issue was found.

## C. The ordered body-force sum in two-way (3.0 ms per frame)

**Verified (code).**
- `example_mpm_twoway_coupling.py:59-110` launches (body_count, 256) threads. Each thread scans
  `node_count / 256` nodes, filtered by body.
- `collider_node_count` is the maximum length of the collected impulse arrays: the fixed grid's node count, up to the
  1M-entry buffers (lines 215 and 309), not the active collider nodes.
- The cost is therefore O(bodies x grid nodes) of reads per substep, with 4 substeps per frame. Each thread runs a
  serial loop of about 4,000 iterations, so the kernel is latency-bound like the IJtm rows.
- The L4 uses `wp.atomic_add`, whose order is not repeatable, so there is no CUDA order to match. The requirement is
  repeatability.

**Options.**
- (a) **Same bits, same latency.** One pass over the nodes, with lane `l` accumulating every body into
  `partials[body, l]`. This gives the identical per-(body, lane) order but the same serial loop length, so no gain.
  Rejected.
- (b) **Compact first.** A stable prefix-scan compaction of the nodes whose collider maps to a body. Order is
  preserved; this is the same machinery as `_compact_triplet_slots`. Then a fixed-lane sum over the compacted list
  (thousands of nodes instead of about 1M), then a two-level fixed-order lane fold.
  - It changes the per-lane order, so it is not bitwise equal to the current Metal output. It stays repeatable.
- (c) **More lanes.** For example 4,096 lanes per body plus a fixed tree fold of the lanes (Warp `tile_sum`, an explicit
  shuffle tree). This cuts the loop length 16x.
  - It still reads O(bodies x nodes), and the order changes too.
- Preferred: (b). It lowers both the work and the latency, and the reductions use Warp's explicit-tree `tile_sum`.
  Metal's `simd_sum` does not document its internal order, so it is avoided for repeatable results.
  - Pitfalls from MLX and llama.cpp reduction kernels: partial SIMD groups and inactive lanes give wrong two-level sums
    unless the group count is rounded up; MLX's `ops.h` shuffle reduction is reported wrong on M1/M2 with inactive
    lanes. Warp's `metal_reduce` handles inactive lanes with an explicit mask.
- **Implemented: option (b)** (`runs/mpm/coupled_speed/patches/newton_body_force_compact.diff`, Newton example).
  - Three steps: `flag_body_nodes`, then an integer `array_scan`, then `compact_body_nodes`. The kept node indices stay
    in increasing order and their count is read on the device. `compute_body_forces_compact_lanes` then runs 256 lanes
    over the compacted list, and the existing `sum_body_force_lanes` folds the lanes in lane order.
  - Capture-safe: no host reads and no allocations per step.
  - `NEWTON_METAL_BODY_FORCE_COMPACT=0` restores the full scan.
  - CPU check (`probes/body_force_compact_check.py`, 65,536 nodes, 12 bodies, 2,416 body nodes):
    - the compaction is exact;
    - the forces are 2.0e-7 from the float64 reference (the full scan: 2.4e-7) and 2.7e-7 from the full scan;
    - two calls are bitwise equal;
    - 0.98 -> 0.25 ms per call.
  - The Metal check and timing are in `batch_bf1.sh`.

## D. ANYmal captured (1.29x): Gauss-Seidel and MuJoCo Warp small kernels

**From the existing notes** (`research_notes/Metal MPM speed and scalable determinism/latency_bound_solvers.md`
Q1-Q4, `apple_vs_nvidia_gpu.md` Q2 and Q4):
- Fusing colors or iterations into one dispatch needs a grid-wide barrier.
- Metal has no inter-threadgroup forward-progress guarantee, and Apple GPUs have shown forward-progress failures
  (Sorensen et al.; Levien 2021). On this machine a non-terminating kernel wedges the GPU and leads to a panic
  (memory: metal-gpu-hang-hazard). So persistent and grid-barrier designs are excluded.
- Bitwise-safe levers left:
  - per-dispatch cost: already cut by command-buffer batching, device arguments and indirect gating (MPM.md 8, 8c);
  - data layout (Newton's `_ReorderedGaussSeidelSolver`: entry-major SoA, unrolled gather; its bitwise status must be
    checked);
  - occupancy and threadgroup sizing per color.

**Measured** (MPM.md 8j, isolated-dispatch profile):
- The Gauss-Seidel kernel takes 8.5 vs 3.3 ms per frame over 400 launches.
- MuJoCo Warp tree passes (`_crb_accumulate`, `_cfrc_backward`, 20 launches each) take 3-11x the L4's time per
  launch. Isolated-dispatch times overstate small kernels inside a replayed graph.

**Next.**
- Measure ANYmal captured by subtraction: replay variants with groups of kernels removed or made empty, so the real
  in-graph share of each group is known before choosing.
- The MuJoCo Warp side belongs to the rigid stack's own performance work. Report it, don't fork it here.

Sources: Warp "Deterministic Execution" guide (nvidia.github.io/warp/latest/user_guide/execution_and_performance/deterministic_execution.html);
`warp.set_module_options` API page; PyTorch PR #150566 (MPS simd reductions, MLX `ops.h` caveat); llama.cpp PR #16559
(two-level simd_sum); MLX issue #3533 (two-level reduction timing on M2); local code and notes cited inline.

## Outcome (2026-10-13; MPM.md 8k)

- A: Warp `31cf79d7` + Newton `addd10cb`. Captured two-way 54.8 -> 40.0 ms/frame.
- C: compaction in the two-way example (40.0 -> 37.6 ms/frame captured, 1.21x the L4) and in the solver's proxy harvest.
  MPMCoupling: harvest 15.6 -> 1.5 ms, coupled step 49.3 -> 34.2 ms at 16 bodies. Franka-Pour unchanged.
- D: the colour-launch threadgroup count (1/2/4/8 per core) and `gs-soa` are bitwise equal to the current solver, and
  neither is faster. ANYmal's remaining 1.21x is mostly the MuJoCo Warp robot graph (8.3 ms vs about 1.4 ms of L4
  kernel time).
- B: a harness artifact, fixed in `scripts/gpu_run.sh` (PR "gpu_run.sh: run the job in the caller's directory").
