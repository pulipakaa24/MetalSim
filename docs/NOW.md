# Now: what is being worked on, what is broken

Updated 2026-10-13 14:30 PDT (main `eb5fee87`). This file is the single place that says what is in flight right
now and what is known to be broken. It is updated with every batch of work; if it is older than the newest merge on
`main`, treat it as stale. Longer-lived status: [STATUS.md](STATUS.md), gaps: [GAPS.md](GAPS.md), MPM:
[MPM.md](../metalsim/physics/mpm/MPM.md).

## In progress

| Work | Branch | State | Next |
|---|---|---|---|
| MPM in the main simulator (MuJoCo Warp `BatchSim` + MPM, two-way proxy coupling as Isaac Lab 3.0 does; Newton 1.5.2 fork's implicit MPM vendored byte-identical) | `mpm-main-sim` | batch_04 running: merge with main, deterministic mode split to its own draft PR, tests, bearing at Isaac's default solve, repose at R/dx 20; MPM PR opens if green |  API (materials, emitters, `MPMCoupling`), tests (angle of repose, momentum, determinism at 1/64/1,024 worlds, rigid neutrality, vendor-drift, license notices), first PR |
| Coupled MPM scenes slower than the L4 when captured (two-way 1.76x, ANYmal on sand 1.29x; MPM.md 8j) | `mpm-coupled-speed` | tiled IJtm product validated: two-way captured 54.8 -> 40.4 ms/frame (1.77x -> 1.30x the L4), eager 0.85x the L4; pure-MPM floors bitwise; Franka-Pour differs only by the tiling choice. Fork commits / pin bump (batch_08) wait for the MPM-main PR | row-tiled product in the Newton 1.5.2 fork, re-run the six floor comparisons + determinism, re-time |
| Franka-Pour vs Isaac: Isaac seeds 1 and 2 (seed study, PARITY §1b.5) | `franka-pour-isaac-seeds` (`5a28b053`) | training on both L4 VMs (seed 2 ETA ~18:21, seed 1 ETA ~19:14 PDT); VMs power off ~30 min after | compare our 5 seeds vs Isaac's 3, verdict, PR |
| Optimization pass on the MPM–BatchSim coupling layer (queued after the first MPM-main PR) | `mpm-main-sim` | queued | research, then optimize (timing on the 16-inch only) |
| Docs audit: every doc checked against the measured record | `docs-audit` (PR #20) | 15 docs corrected (STATUS rewritten to current main, README, GAPS, PARITY, GUIDE, CHANGELOG, RENDER_COVERAGE, MPM.md, …); batch re-measures test counts, then PR | merge |

## Known broken or behind

| Problem | Severity | Where | Owner / plan |
|---|---|---|---|
| Captured coupled MPM scenes slower than the L4 (two-way 1.30x with the pending fix, was 1.76x; ANYmal 1.21x) | performance | MPM.md 8j | in progress above |
| Franka-Pour learns ~0.06 slower than Isaac at its 50-59 (5 of ours vs 1 Isaac seed; not yet settled) | parity, unconfirmed | PARITY §1b.5 | Isaac seeds above |
| With the staggered MPM–rigid exchange (Isaac Lab 3.0's Franka-Pour setting and Isaac's own path) a body resting on sand sinks through any bed (force fed back one step late). Lagged exchange (now `MPMCoupling`'s default) carries a resting box: 10 mm settling with a converged solve (250 it, 1e-6), 25 mm and creeping at 50 iterations; drops penetrate 32–39 mm vs 10–25 mm expected | physics fidelity (coupling + solve convergence) | MPM.md §11 | documented; Franka-Pour keeps Isaac's setting for parity |
| Newton 1.5.2 implicit-MPM sand's angle of repose is 7–9° below atan(μ) (μ 0.4/0.68/0.9: 14.7° / 25–27° / 32.6° vs 21.8° / 34.2° / 42.0°), insensitive to lift speed and iterations, ~1° per resolution step. Same solver as Isaac Lab | physics fidelity (solver) | MPM.md §11 | documented; resolution R/dx 20 being measured |
| Newton 1.5.2 Gauss-Seidel colouring with `separate_worlds` is invalid (strain nodes numbered world-first, coloured in cell order): data race, multi-world MPM runs not repeatable. Also in upstream Newton 1.5.2 and main. Found by code reading | correctness | Newton fork `metalsim-1.5.2-gs-envfirst` | fix being validated in MPM-main batch_01 |
| `scripts/gpu_run.sh` ran jobs in the main checkout, so `python -m pytest` from a worktree imported stale main-checkout code (caused a false test failure) | test harness | scripts/gpu_run.sh | fix PR `fix/gpu-run-caller-cwd` (run with batch_07); affected runs re-run and confirmed clean |
| MPM runs only through Newton (`.venv-newton152`), not in metalsim's own `BatchSim` scenes | missing feature | — | in progress above |
| Tier 1 draws MPM particles but without ray-traced shadows; the ray-traced sensors (depth camera, lidar) do not see them; tier 0 still uses the far-field approximation for area lights | render gaps | RENDER_COVERAGE.md 1c, 3e | not scheduled |
| Newer USD dome-light flavours (DomeLight_1 in Kit, `omni:rtx:usdluxVersion` >= 2505) use the measured legacy convention, unverified against Kit | render, unverified | research_notes/dome_light_azimuth/ | needs a Kit measurement on the L4 |
| MuJoCo Warp `BatchSim` is not run-to-run bitwise repeatable in its default mode (a box on the floor diverges by step 18 at 1,024 worlds). Opt-in deterministic mode in progress: repeats for rigid and coupled scenes at 1,024 worlds, but stacked boxes at 256 worlds still diverge in 2 worlds from step 139 | determinism | GAPS, research_notes/MPM in main simulator/deterministic_ccd_divergence.md | draft PR on branch `batchsim-deterministic`; suspected cause: MuJoCo Warp's box-box collision kernel reserves contact slots via a two-pass counter whose first pass reads uninitialised scratch |
| Development runs in an org-managed sandbox: no Metal GPU, ssh or `gh` from the agent shell; GPU jobs, commits and pushes run as batch scripts by the owner | process | — | — |

Test suites (measured 2026-10-13 14:49 on main `3670f4b8`): rigid suite 424 passed, 10 skipped, 0 failed; `tests/test_mpm.py` 15 passed, 1 expected xfail. The contact-filter test above passed in this run (warm kernel cache).

## How this file is kept current

Every batch of work that changes what is in flight or what is broken updates this file in the same batch
(`scripts/publish_now.sh` lands it on `main` as a doc-only PR merged immediately).
