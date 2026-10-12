# Franka-Pour MPM step at 2,048 worlds on Metal: where the time goes, and what can go without changing a bit

Date: 2026-10-12. Scene: Isaac Lab's Franka-Pour MPM as `metalsim/physics/mpm/probes/scale_pour.py --part mpm` /
`pour_bitwise.py` mirror it:
- 2,048 worlds x 735 particles in one `SolverImplicitMPM`, `separate_worlds`;
- rebuildable sparse grid, 1,024 cells reserved per world (about 48 strain and 104 velocity nodes used per world);
- voxel 0.015; Jacobi capped at 24 iterations (25 run); P0 strain; pic27 colliders;
- a 50 kg proxy cup, so the solve has a rigidity operator.

Machine: M4 Max, `metal_fast_math` off. Base trees: Warp fork `metalsim` 43ffc6dd (then 41b7bc45), Newton fork
`metalsim-1.5.2` c0427fc7.

## 1. What was read

**Newton 1.5.2:**
- `solver_implicit_mpm.py`: the step pipeline, grid / scratchpad rebuild, collider rasterization, the rigidity operator,
  `_solve_rheology`.
- `solve_rheology.py`: the Jacobi and sub-grid contact solvers, `apply_rigidity_operator`, the loop and the batched
  condition.
- `rheology_solver_kernels.py`, `contact_solver_kernels.py`, `rasterized_collisions.py`.

**Warp:**
- `warp.fem`: the generic and dispatched integrate kernels, interpolate-to-BSR, `compress_node_indices`,
  `SpaceRestriction`, `EnvironmentSpacePartition`.
- `warp.sparse`: `bsr_mv`, `bsr_set_transpose`, `bsr_set_from_triplets`.
- The fork's `metal_sparse.py` (Metal BSR builder and transpose).
- `metal_primitives.radix_sort_pairs`, including its `count_limit` contract.
- The Metal launch path:
  - `wp_bounded_prep` plus an indirect dispatch;
  - threadgroup size `min(block_dim, pipeline.maxTotalThreadsPerThreadgroup)`;
  - entry points that return when `task_index >= dim.size`.

**Literature (background survey):**
- GPU MPM grid construction: Gao et al. 2018; Wang et al. 2020 / Claymore; Fei et al. 2021; Taichi / AsyncTaichi.
- Batched one-workgroup-per-system iterative solvers (Ginkgo, arXiv 2308.08417).
- Temporal blocking for Jacobi.
- Apple GPUs:
  - no forward-progress guarantee between threadgroups (Sorensen et al. OOPSLA 2021; Apple developer forum 672532);
  - 32 KB of threadgroup memory;
  - Dynamic Caching occupancy.

Two points from the survey decided the design space:
- Cell-sorted particles and warp-level P2G reductions change summation order, so they are rounding-level only.
- A per-system workgroup solver is exact when the stopping rule stays global.

## 2. Profile

**How it was measured.** `probes/section_kprof.py` swaps Newton's section timers for a context manager. At each section
boundary it synchronizes and collects Warp's `WP_METAL_PROFILE` per-kernel GPU times.
- Stepping is eager, one command buffer per dispatch. The empty-kernel command buffer cost (about 2.5 us) is subtracted.
- These numbers attribute time to kernels. The figure of merit is the captured step: one graph replay.
- A profile taken in gpuexit / captured mode misses every kernel replayed inside an indirect command buffer (contact
  solve, rigidity products, residual), so attribution must be done eager.

Base, 2,048 worlds, ms of GPU time per step (total about 108):

| section | ms | main kernels |
|---|---|---|
| Strain solve | 34.1 | rigidity `bsr_mv` x52: 8.8; strain + collider transposes: 9.1; Jacobi x25: 3.7; sub-grid friction x25: 3.6; impulse / stress application: 3.4 |
| Scratchpad (function spaces) | 16.1 | `_prepare_node_sort_kernel` x6: 5.9; `_split_vertex_element_index` x3: 3.5; `memtile`: 1.6 |
| grid rebuild etc. (outside sections) | 15.3 | NanoVDB builder (`vb_*` keys, sorts, leaves), memsets |
| Rasterize collider | 14.4 | collider-matrix triplet sort (64-bit keys): about 6; `collision_weight_field`: 2.2; `rasterize_collider_kernel`: 2.0 |
| Unconstrained velocity | 8.9 | APIC `integrate` (generic assembly, `at_node`): 5.5 |
| Strain matrix | 6.5 | bilinear dispatch: 1.4; triplet keys / segments / heads: 2.8 |
| Collider compliance | 6.4 | J / IJtm assembly: two sorts of 3 x node-count triplets |

Captured step at 2,048 worlds: 90.5-92.7 ms.

## 3. Candidates

**Grid / scratchpad reuse on unchanged occupancy: never applies.**
- `probes/pour_occupancy.py` compared the (world, cell) key sets of consecutive steps over 200 steps.
- At 2,048 worlds no step was unchanged. On average 46 % of the worlds change cells every step, and at least 45 worlds
  change in every step. At 64 worlds also 0 of 199 steps were unchanged.
- An incremental rebuild with output identical to a full rebuild is not possible here. NanoVDB voxel indices and the
  environment-first node numbers depend on every world's occupancy.

**Per-world fused Jacobi (one threadgroup per environment, barriers between the six passes): bitwise, slower at scale.**

| worlds | 64 | 256 | 1,024 | 2,048 |
|---|---|---|---|---|
| dispatch per pass (ms) | 7.71 | 15.45 | 47.2 | 92.1 |
| fused per world (ms) | 7.05 | 16.85 | 54.2 | 105.2 |

- It is bitwise at 1, 64 and 2,048 worlds, eager and captured.
- It loses at scale because a world (about 48 strain and 104 velocity nodes on 256 lanes) runs 125 barrier-separated
  passes per solve, latency-bound. The register footprint of the inlined local solve limits how many threadgroups are
  resident.
- Not adopted. The code is on Newton branch `fused-jacobi-attempt` (local).
- Its kernel was reviewed after the 10:31 crash; the timeline in section 5 places that crash before any of its launches.
  - Every loop is bounded and uniform per threadgroup.
  - It has no cross-threadgroup dependency.
  - It now refuses to run unless the pipeline holds the full threadgroup (`threadgroup_size_ok`). A smaller threadgroup
    would split an environment and leave a partial last group whose idle threads return before the barriers.

**Rigidity products on non-empty rows only: bitwise, adopted (Newton).**
- An empty row of `y = alpha A x + beta y` writes a fixed point: `0.0 + y` for J, `-0.0` for IJtm. So after the initial
  guess's full product, only the rows with blocks need recomputing.
- Implementation: device-side row list (flag, scan, scatter), then `bsr_mv`'s kernel body over the list in a launch
  bounded by its length.
- 8.8 -> 2.6 ms.
- Lesson: a bounded launch may run the rest of its last threadgroup.
  - The first version lacked an in-kernel guard, so those threads re-applied row 0. `debug_listed.py` showed row 0
    receiving twice the delta.
  - Every bounded launch needs an in-kernel bound check.
- Used from 8,192 rows. At one world the extra dispatches cost 0.5 ms.

**Rigidity triplets compacted before assembly: bitwise, adopted (Newton).** The kept slots move to the front in slot order
and their count goes to `bsr_set_from_triplets`; the stable sort orders them as before. Collider compliance 6.4 -> 3.1 ms.

**Compact transposes bounded to the stored blocks: bitwise, adopted (Warp `metal_sparse`).**
- The offsets search only the sorted prefix and skip the columns past the last kept key.
- The column pass does not read the keys past the prefix.
- The value pass is a bounded launch.
- 9.1 -> 4.9 ms.

**Node-index compression skips the ignored tail: bitwise, adopted (Warp `warp.fem`).** No key writes past the device
bound. For `SpaceRestriction`, no writes at all past the bound. `_prepare_node_sort_kernel` 5.9 -> 1.4 ms.

**Generic linear integrate with all of a node's dofs in one thread (APIC): bitwise, no gain.** 92.1 vs 92.6 ms
(Warp branch `dofloop-attempt`).

**Row-ordered triplet presort instead of the radix sort: bitwise, slower.**
- Method: sort each run of equal rows in place, stable partition, order check, radix count 0.
- It matched the radix sort on every unit case and on the trajectories.
- Rasterize 13.9 -> 18.1 ms, strain matrix 6.2 -> 10.3 ms, step 79.2 -> 85.8 ms.
- Not adopted.

**Fusing the small per-step kernels (rasterize, compliance, warm start, unconstrained velocity): not pursued.** Those
sections are dominated by sorts and generic assembly, not dispatch counts.

**Particle sorting by cell: not done.** It changes P2G summation order, so it is rounding-level and needs the L4 floor
ensemble. Left as a labelled option.

## 4. Result

Captured ms per MPM step, base vs final, two interleaved rounds (Warp 43ffc6dd + 2 commits, Newton c0427fc7 + 2):

| worlds | base | final |
|---|---|---|
| 1 | 3.96 / 3.84 | 4.00 / 3.99 |
| 64 | 8.00 / 7.75 | 7.87 / 7.46 |
| 256 | 15.40 / 15.53 | 14.19 / 14.16 |
| 1,024 | 45.97 / 46.77 | 41.38 / 41.07 |
| 2,048 | 91.16 / 90.50 | 78.95 / 78.88 |

**Correctness checks:**
- Every recording is bitwise equal to the base: particle q, qd, APIC gradients and all `state.mpm` arrays, hashed every
  10 steps over 320 steps, at 1, 64 and 2,048 worlds, eager and captured.
- `tests/test_mpm.py`: 14 passed, 1 xfailed.
- Warp tests pass: `test_metal_sparse`, `test_metal_bounded_launch`, `test_metal_primitives`, and fem `multi_env` /
  `integrate` / `quadrature`.
- `test_sparse.test_bsr_copy_scale` errors identically on the base: float64 on Metal.

GPU time per step at 2,048 worlds went from 108 to 91 ms (eager attribution).

Franka-Pour policy step (`franka_pour_ppo bench --mode step`, 2,048 envs, M4 Max, 20 steps after 3, two rounds):
2.059 / 2.064 s -> 1.810 / 1.813 s.

Pushed: Newton `metalsim-1.5.2` `27415f95`, `80e79684` (+ docs `3fd1f052`); Warp `metalsim` `8631b6b0`, `fddebdbd` (+ docs
`3342e5d8`), on top of `41b7bc45`. That upstream commit faults on a one-world captured Franka-Pour step (base tree too),
so MetalSim pins Newton `3fd1f052` and keeps Warp at `43ffc6dd` for now. Newton's two commits alone on Warp `43ffc6dd`
(what a fresh setup gets): bitwise; 2,048 worlds 90.6 / 90.5 -> 84.4 / 85.0 ms per captured step. The rejected
attempts' diffs are in `runs/mpm/step_at_scale/rejected/`.

Occupancy re-run (committed log `runs/mpm/step_at_scale/occupancy_2048.log`): 0 of 199 steps unchanged, 54.5 % of
worlds unchanged per step on average, 931 worlds changed per step on average, at least 45.

## 5. The 10:31 hang

Every WindowServer watchdog report from 10:20 to 10:33 shows the kernel's `IOGPU Completion` thread last running at
about 10:20:13. That is during another job (10:17:04-10:28:33), 8.5 minutes before the fused-Jacobi job was granted at
10:28:37. The GPU completed nothing after that time, so neither the fused kernel nor the dof-loop kernel can have
executed on a working GPU.

The review and the tiny-launch protocol were applied anyway:
- `probes/first_launch.py` runs 1, then 64, then 2,048 worlds.
- Each step's completion is awaited from a worker thread with a 5-10 s limit. Host-side module builds are excluded from
  the limit, and the process exits on timeout without waiting for the GPU.
- Every new kernel passed this protocol before it ran at scale.

## 6. Remaining (final profile, 2,048 worlds)

Remaining costs:
- grid rebuild, about 15 ms (NanoVDB key sorts and leaf builds);
- collider-matrix assembly in rasterize, about 14 ms (a 64-bit radix sort over 1.5 M x 8 triplets);
- scratchpad tables, about 11 ms (`_split_vertex_element_index` and `memtile` over capacity-sized tables whose tails the
  dispatched assembly reads);
- APIC generic integrate, 5.5 ms;
- Jacobi / friction / impulse passes, about 10 ms.

Bitwise candidates:
- narrower radix keys (the column count is the 8,192-node-per-world velocity reservation, about 104 used);
- restriction tables whose readers stop at the active offsets;
- NanoVDB passes bounded by the active voxel count.
