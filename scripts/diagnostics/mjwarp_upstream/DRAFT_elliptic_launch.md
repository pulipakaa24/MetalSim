# Pull request for google-deepmind/mujoco_warp: ready, NOT opened (blocked on the owner's Google CLA)

Status 2026-09-26. Head: github.com/pulipakaa24/mujoco_warp branch `elliptic-jtcj-offcuda`, commit `6d8e29a`
(one commit on upstream `main` `cc97eea`, which is still upstream's head; no rebase needed). Author and committer
`Aditya Pulipaka <adipu@utexas.edu>`, no co-author trailer (upstream AGENTS.md: AI co-authors fail the CLA check).
Commit message unwrapped per upstream AGENTS.md. Issue filed: https://github.com/google-deepmind/mujoco_warp/issues/1704.
Worktree: `upstream/mujoco_warp-pr`.

Checks run 2026-09-26 (CPU only; scratch venv from upstream's `uv.lock`: warp-lang 1.15.0, mujoco 3.13.1):
- `pytest mujoco_warp/_src -q --cpu -n 8`: 1514 passed, 39 skipped, 1 failed
  (`io_test::test_put_data_nefc_zero_dense`, fails identically on `main` `cc97eea`).
- CPU timing (`scripts/diagnostics/competitors/elliptic_cpu_time.py 32 100`, Go2, 3 runs each, mujoco 3.14.0):
  `main` 10.2 / 10.3 / 10.4 ms/step, branch 5.7 / 5.8 / 5.8. (The 12.8 -> 5.8 of 2026-09-25 was on the v3.14.0 fork.)
- Oracle (`elliptic_cpu_check.py`, 2 worlds, 40 steps vs mj_step): Go2 max |dq| 1.8e-7 .. 2.4e-6 (main 1.8e-7 .. 2.3e-6),
  G1 sparse 6.7e-8 .. 3.2e-7 on both.
- Metal numbers are from the v3.14.0-based MetalSim fork (2026-09-25), not re-run (no GPU work in this session).

To open once the CLA shows as signed (the owner runs this; it is the only step left):

```
gh pr create -R google-deepmind/mujoco_warp --head pulipakaa24:elliptic-jtcj-offcuda --base main \
  --title "Size the elliptic-cone Hessian launch by world off CUDA" --body-file <the body below>
```

Then confirm the `cla/google` check is green; if it is red, the commit email (adipu@utexas.edu) or the GitHub
username is not on the signed CLA.

---

## Title

Size the elliptic-cone Hessian launch by world off CUDA

## Body

Fixes #1704.

Off CUDA, `_update_gradient` launched the elliptic-cone Hessian term `_update_gradient_JTCJ_dense` with `dim_block = d.naconmax` ("fall back for CPU"), i.e. one thread per contact slot per lower-triangle Hessian entry, once per Newton iteration. The launch scaled with the contact capacity instead of the contacts that exist (Go2 at 4096 worlds: 33.6 M threads per launch, more than 99 % of them returning immediately), and the active threads accumulated into `ctx_h` with atomics.

This adds `_update_gradient_JTCJ_dense_world`, launched with `dim=(nworld, ndof_tri)` on devices other than CUDA. Each thread owns one (world, Hessian entry), scans its world's constraint rows for elliptic contacts in the CONE state and writes its entry once. It uses the same `_elliptic_hessian_entry_from_projections` contraction as the contact-major kernel, has no atomics, and sums contacts in constraint-row order, so the result is deterministic. The CUDA path is unchanged.

On the Warp CPU device, the Go2 from MuJoCo Menagerie with its own `cone="elliptic" impratio="100"`, 32 worlds, Newton 10 iterations and 20 line-search iterations, goes from 10.2-10.4 to 5.7-5.8 ms/step against `main`. On a Metal port of Warp (M4 Max, the same change on v3.14.0) the cone term went from 72 % of the step to about 20 %, and throughput at 4096 worlds went from 240,281 to 553,510 steps/s.

`pytest mujoco_warp/_src --cpu` passes except `io_test::test_put_data_nefc_zero_dense`, which fails the same way on `main`. Against `mj_step` on the CPU device with elliptic cones, Go2 (dense Jacobian) and G1 (sparse) stay within the same envelope as `main` over 40 steps (max |dq| 1.8e-7 to 2.4e-6 rad).

One tradeoff: I also tried the existing contact-major kernel with `dim_block` sized from a core count on the Metal backend, which exposes `Device.sm_count`. It performed about the same as this kernel there (538,057 vs 553,510 steps/s at the CUDA factor of 6). I went with the world-major kernel because it has no tuning factor and no atomics, and its summation order is deterministic.
