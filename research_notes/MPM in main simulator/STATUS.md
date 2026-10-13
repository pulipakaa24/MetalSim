# MPM in the main simulator: status (resume here)

Goal: MPM material (sand, snow, ...) in metalsim's main simulator (`.venv`: BatchSim on MuJoCo Warp 3.14 fork,
Newton 1.7 line), two-way coupled with the robot, N worlds, rendered by `render/particles.py`; parity with the
`.venv-newton152` (Newton 1.5.2 fork, Isaac Lab 3.0-EA) results. Branch `mpm-main-sim` (worktree
`.claude/worktrees/agent-ab17e9d0efefed741`).

## Decided (2026-10-13; design approved by the coordinator)
- Option (b), `NOTE.md`: vendored copy of the fork's `implicit_mpm` (`scripts/vendor_mpm152.py`, Apache-2.0 notices,
  drift test), `MPMCoupling` (`metalsim/physics/mpm/coupled.py`) reproducing Isaac's `SolverCoupledProxy` with BatchSim
  as the rigid entry. Licensing + drift requirements from the coordinator: done (NOTICE.txt, THIRD_PARTY_NOTICES,
  `tests/test_mpm_main.py`, MPM.md section 11).

## Done (measured)
- Vendored copy in `.venv` bitwise = Newton 1.5.2's own solver (column, every frame; 16-inch).
- Box on sand vs Isaac's `SolverCoupledProxy` (.venv-newton152), GS, 2 worlds, 14-inch: box height within 1.1e-6 m,
  wrench within ~1e-2 N after the proxies take end-of-step kinematics (`runs/mpm_main/box_on_sand/`).
- Found: GS + separate_worlds is not run-to-run repeatable, in Newton's own 1.5.2 too (Jacobi and 1 world are):
  `runs/mpm_main/repeat/mwrep1.log`, `mwrep2.log`. Root cause (code reading): environment-first strain partition vs
  `fill_uniform_color_block_indices` (cell * npe) -> invalid coloring -> data race. Fix written for the fork
  (`runs/mpm_main/fork/b`, kernel `make_partition_color_block_indices_kernel`); CPU dense/fixed multi-env grids VALID
  with it (sparse multi-env grids only build on GPU).

## batch_01 results (15:07): fork fix validated (old INVALID, 930 conflicts; new VALID; 12/12 repeatable); column
bitwise; box pair bitwise until contact then 1.7e-7 m; neutrality; GS repeatable at 1/64/256/1,024; timing measured
(MPM.md section 11). Failed: Jacobi 1,024 repeat (step 34, box reaches the floor -> suspect MuJoCo Warp
non-determinism), free-fall test (lateral 2e-4 m/s from the solver's numerical air drag), repose (my batch marked it
PASS unconditionally: a bug; and a collapse deposit is not a repose angle).

## batch_02 results (15:59) and batch_03 (in flight)
- Rigid side: MuJoCo Warp not repeatable by default (box on floor diverges at step 18 at 1,024 worlds);
  `BatchSimOptions(deterministic=True)` makes box / coupled scenes repeat at 1,024 worlds, but the stacked-boxes test
  at 256 worlds failed in one pytest process. batch_03: options before put_model, one mode per process, record bound
  in scope, replayed-module check, overflow flag; cold / warm cache cross-process tests (`deterministic_batchsim.md`).
- Bearing: every configuration sank at 0.082 m/s = g dt -> staggered-exchange lag (`bearing_and_sinking.md`); default
  switched to "lagged"; batch_03 runs the 12-case matrix.
- Repose: batch_02 lift void (unstable tube actuator, tube at 1e20 m); collapse parity with Newton bitwise; batch_03
  runs the corrected protocol (`angle_of_repose.md`).
- Jobs import the worktree's metalsim: gpu() runs every command with cwd = worktree (watchdog --cwd); batch_03 also
  uses `python -P -m` and logs an import audit.

## Earlier: batch_02 (`runs/mpm_main/batch_02/batch_02.sh`): per-part divergence diagnosis, new opt-in
`BatchSimOptions(deterministic=True)`, rigid regression, collapse parity vs Newton alone, slow-lift repose, bearing
settlement matrix, tests, then fork push + PR if green.

## Earlier: batch_01 (`runs/mpm_main/batch_01/batch_01.sh`, run by the user outside the sandbox)
Fork commit + pin move + re-vendor, coloring check old/new on Metal, multi-world repeat, column bitwise, box-on-sand
Jacobi pair (bitwise before contact), rigid neutrality, angle of repose, 1/64/256/1024-world repeatability, full
tests, 16-inch timing, fork push, commit + push, PR if green. Logs: `runs/mpm_main/batch_01/logs/`, summary
`runs/mpm_main/batch_01/STATUS.txt`.

## Next
- Read batch_01 logs; write the numbers into MPM.md section 11, PARITY/GAPS rows, NOTE.md; update the PR body.
- Extensions (later PRs): ANYmal-on-sand / Franka-Pour scene in the main simulator vs the .venv-newton152 results;
  sensors (contact/force sensor from sand wrench), per-world material randomization, emitters at runtime.

Sandbox note (2026-10-13 afternoon): no GPU, ssh, git commit/push from the agent; every GPU / git step goes into a
batch script the user runs. Local Warp runs need `WARP_CACHE_PATH=$TMPDIR/warpcache` (CPU only).
