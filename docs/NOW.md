# Now: what is being worked on, what is broken

Updated 2026-10-13 14:30 PDT (main `eb5fee87`). This file is the single place that says what is in flight right
now and what is known to be broken. It is updated with every batch of work; if it is older than the newest merge on
`main`, treat it as stale. Longer-lived status: [STATUS.md](STATUS.md), gaps: [GAPS.md](GAPS.md), MPM:
[MPM.md](../metalsim/physics/mpm/MPM.md).

## In progress

| Work | Branch | State | Next |
|---|---|---|---|
| MPM in the main simulator (MuJoCo Warp `BatchSim` + MPM, two-way proxy coupling as Isaac Lab 3.0 does; Newton 1.5.2 fork's implicit MPM vendored byte-identical) | `mpm-main-sim` | batch_03 running: bearing (lagged vs staggered exchange), repose (published lifted-cylinder method), hardened opt-in deterministic BatchSim (stacked boxes diverged at 256 worlds in batch_02), test re-runs; PR if code gates pass. Captured coupled step 13.7 ms (1 world) to 224 ms (1,024 worlds) |  API (materials, emitters, `MPMCoupling`), tests (angle of repose, momentum, determinism at 1/64/1,024 worlds, rigid neutrality, vendor-drift, license notices), first PR |
| Coupled MPM scenes slower than the L4 when captured (two-way 1.76x, ANYmal on sand 1.29x; MPM.md 8j) | `mpm-coupled-speed` | late two-way kernel sum 55 -> 36 ms per frame; tiled IJtm validated: pure-MPM floors bitwise, two-way repeatable and inside the L4 floor (0.83); timing still running in batch_03 | row-tiled product in the Newton 1.5.2 fork, re-run the six floor comparisons + determinism, re-time |
| Franka-Pour vs Isaac: Isaac seeds 1 and 2 (seed study, PARITY §1b.5) | `franka-pour-isaac-seeds` (`5a28b053`) | training on both L4 machines (seed 2 ETA ~18:21, seed 1 ETA ~19:14 PDT) | compare our 5 seeds vs Isaac's 3, verdict, PR |
| Optimization pass on the MPM–BatchSim coupling layer (queued after the first MPM-main PR) | `mpm-main-sim` | queued | research, then optimize (timing on the 16-inch only) |
| Docs audit: every doc checked against the measured record | `docs-audit` (PR #20) | 15 docs corrected (STATUS rewritten to current main, README, GAPS, PARITY, GUIDE, CHANGELOG, RENDER_COVERAGE, MPM.md, …); batch re-measures test counts, then PR | merge |

## Known broken or behind

| Problem | Severity | Where | Owner / plan |
|---|---|---|---|
| Captured coupled MPM scenes slower than the L4 (two-way 1.76x, ANYmal 1.29x) | performance | MPM.md 8j | in progress above |
| Franka-Pour learns ~0.06 slower than Isaac at its 50-59 (5 of ours vs 1 Isaac seed; not yet settled) | parity, unconfirmed | PARITY §1b.5 | Isaac seeds above |
| With the staggered MPM–rigid exchange (Isaac Lab 3.0's Franka-Pour setting, and Isaac's own coupling path) a body resting on sand sinks at g·dt per step through any bed: the force is fed back one step late while the velocity is not. A coupling-lag artefact, not weak sand (bearing capacity should hold it within 1–2.5 cm). Affects Isaac too; matching Isaac and matching real sand are separate claims | physics fidelity (coupling), root-caused | research_notes/MPM in main simulator/bearing_and_sinking.md | `MPMCoupling` default switched to Newton's lagged exchange with velocity rewind; validation in batch_03. Franka-Pour keeps Isaac's setting for parity |
| Angle of repose of the MPM sand not yet measured (the earlier run was void: unstable lifting tube). Expected atan(μ) = 21.8 / 34.2 / 42.0° from the solver's yield rule | physics fidelity, unmeasured | research_notes/MPM in main simulator/angle_of_repose.md | rewritten lifted-cylinder probe in batch_03 |
| Newton 1.5.2 Gauss-Seidel colouring with `separate_worlds` is invalid (strain nodes numbered world-first, coloured in cell order): data race, multi-world MPM runs not repeatable. Also in upstream Newton 1.5.2 and main. Found by code reading | correctness | Newton fork `metalsim-1.5.2-gs-envfirst` | fix being validated in MPM-main batch_01 |
| Test runs launched through `scripts/gpu_run.sh` as `python -m pytest` from a worktree imported the MAIN checkout's (stale) code, because gpu_run starts jobs in the main checkout. This produced the false contact-filter test failure and may have made other batch test results test the wrong code | test harness | scripts/gpu_run.sh | fix being made (jobs run in the caller's directory); affected results being re-run |
| MPM runs only through Newton (`.venv-newton152`), not in metalsim's own `BatchSim` scenes | missing feature | — | in progress above |
| Tier 1 draws MPM particles but without ray-traced shadows; the ray-traced sensors (depth camera, lidar) do not see them; tier 0 still uses the far-field approximation for area lights | render gaps | RENDER_COVERAGE.md 1c, 3e | not scheduled |
| Newer USD dome-light flavours (DomeLight_1 in Kit, `omni:rtx:usdluxVersion` >= 2505) use the measured legacy convention, unverified against Kit | render, unverified | research_notes/dome_light_azimuth/ | needs a Kit measurement on the L4 |
| MuJoCo Warp `BatchSim` is not run-to-run bitwise repeatable in its default mode (deterministic mode exists) | determinism | MPM.md 8e, PARITY | by design of the default; documented |
| Development runs in an org-managed sandbox: no Metal GPU, ssh or `gh` from the agent shell; GPU jobs, commits and pushes run as batch scripts by the owner | process | — | — |

Test suites (measured 2026-10-13 14:49 on main `3670f4b8`): rigid suite 424 passed, 10 skipped, 0 failed; `tests/test_mpm.py` 15 passed, 1 expected xfail. The contact-filter test above passed in this run (warm kernel cache).

## How this file is kept current

Every batch of work that changes what is in flight or what is broken updates this file in the same batch
(`scripts/publish_now.sh` lands it on `main` as a doc-only PR merged immediately).
