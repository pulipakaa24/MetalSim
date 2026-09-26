# Draft pull request for google-deepmind/mujoco_warp (NOT opened; the owner's CLA and approval are needed)

Branch: `elliptic-jtcj-offcuda` in the fork (github.com/pulipakaa24/mujoco_warp), one commit on top of upstream
`main` `cc97eea`, worktree `upstream/mujoco_warp-pr`. No co-author trailer (upstream's CLA check rejects it).
Measurements below: 2026-09-25, M4 Max (40-core GPU, 64 GB), Warp fork `metalsim` (Metal backend), MuJoCo Warp
fork `metalsim` `07a51a6` (v3.14.0 + Metal device patch); the same change applies unmodified to `main`.

---

## Title

solver: world-major elliptic-cone Hessian term off CUDA (replaces the `dim_block = naconmax` fallback)

## Summary

`_update_gradient` launches the elliptic-cone Hessian term `_update_gradient_JTCJ_dense` as
`dim=(dim_block, ndof_tri)`. On CUDA `dim_block` is sized from the SM count and each thread loops over contact
slots in strides of `dim_block`. On every other device the fallback ("fall back for CPU") is
`dim_block = d.naconmax`: one thread per contact slot per lower-triangle Hessian entry, 11 launches per step
(one per Newton iteration). The launch scales with the contact *capacity*, not with the contacts that exist:

- Go2 (Menagerie, `cone="elliptic"`), 4096 worlds, `naconmax` 196,608 (48 per world), 171 Hessian entries:
  33.6 M threads per launch, of which > 99 % read `nacon` and return (~5 contacts per world exist).
- Every active thread accumulates into `ctx_h[world, i, j]` with `+=`, which Warp lowers to an atomic add;
  all contacts of a world contend on the same entries.

This PR adds `_update_gradient_JTCJ_dense_world`, launched `dim=(nworld, ndof_tri)` off CUDA: each thread owns
one (world, Hessian entry), scans its world's constraint rows (`nefc[world]`, so the launch is sized by the
world count) for elliptic contacts in the CONE state, and writes its entry once. Same arithmetic per contact
as `_update_gradient_JTCJ_dense` (the same `_elliptic_hessian_entry_from_projections` contraction), no atomics,
and the contacts of a world are visited in constraint-row order, so the sum is deterministic. The CUDA path is
unchanged.

## Measurements

Go2 scene from MuJoCo Menagerie with its own `cone="elliptic" impratio="100"`, position-servo PD to random
targets, 4 ms step, Newton 10 iterations / 20 line-search iterations, 1000 timed steps after 100 warm-up
(script: MetalSim `scripts/diagnostics/competitors/metalsim_step.py`; profile: `metalsim_profile.py`).

| device | before (`dim_block = naconmax`) | after (world-major) |
|---|---|---|
| Metal (M4 Max), 4096 worlds, steps/s | 240,281 | 553,510 (2.30×) |
| Metal, per-kernel GPU time of the cone term, ms/step (11 launches) | 9.71 (72 % of the step; other session's profile, same machine class) | 1.54 (20 %) |
| Warp CPU device, 32 worlds, ms/step | 12.8 | 5.8 (2.2×) |

Pyramidal cones on the same scene: 923,850 steps/s on Metal, so elliptic cones cost 1.67× after the change
instead of 3.84×. The G1 (Menagerie, nv 35, MuJoCo Warp's `auto` Jacobian → sparse) does not take this kernel;
its elliptic slowdown off CUDA has a different cause (one lane per constraint group in `_JTDACJ_sparse`, a
separate change in the MetalSim fork, not part of this PR).

## Correctness

- `mujoco_warp/_src/solver_test.py` and `forward_test.py` on the CPU device: see the test count in the
  commit message (run with `pytest --cpu`).
- Go2 and G1 with elliptic cones vs MuJoCo C (`mj_step`), 2 worlds, PD to random targets on the CPU device:
  max |dq| 1.8e-7 → 2.4e-6 rad over 40 steps, identical to the previous kernel's envelope
  (MetalSim `scripts/diagnostics/competitors/elliptic_cpu_check.py`).
- On Metal, 512 worlds × 200 steps under graph replay against the previous kernel: differences within the
  run-to-run floor of two instances of the same configuration (MuJoCo Warp orders contacts nondeterministically);
  the MuJoCo C oracle envelope (4 worlds, 100 steps) unchanged
  (`scripts/diagnostics/competitors/elliptic_check.py`; numbers in MetalSim `docs/research/elliptic_cones_2026-09-25.md`).

## Notes for reviewers

- The CPU backend also takes the new kernel (measured above; the old fallback was written for it).
- `wp.tid()` for `dim=(nworld, ndof_tri)` puts consecutive threads on consecutive Hessian entries of one world,
  so the `efc_J[world, row, dof]` reads of a contact row are contiguous across the triangle's column index.
- If a backend exposes `Device.sm_count` (Warp's Metal backend reports its GPU core count there), the contact-major
  kernel with `dim_block = ceil(sm_count * 6 * 256 / ndof_tri)` measures the same as the world-major one on Metal
  (538,057 vs 553,510 steps/s at factor 6; 483,040 at factor 2; 551,979 at factor 16), so the SM-sized form is
  not specific to CUDA; the world-major kernel was preferred for the PR because it has no tuning factor, no
  atomics, and a deterministic sum.
- The MetalSim fork additionally carries a two-pass form (`MJW_JTCJ_MODE=world2`: one thread per world lists its
  cone contacts and curvature terms, then the entry threads apply the list; Go2 580,420 and G1-dense 144,348 vs
  553,510 / 126,568 for the one-pass kernel). It needs three workspace arrays on the solver context, so it is
  left out of this PR; it can follow if the one-pass form is accepted.
- The kernel uses `main`'s scale-invariant cone contraction (`bc8ed60`, #1660), so
  `test_elliptic_hessian_scale_invariance` passes on the CPU for both Jacobian layouts.
