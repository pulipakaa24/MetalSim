# Now: what is being worked on, what is broken

Updated 2026-10-13 14:30 PDT (main `eb5fee87`). This file is the single place that says what is in flight right
now and what is known to be broken. It is updated with every batch of work; if it is older than the newest merge on
`main`, treat it as stale. Longer-lived status: [STATUS.md](STATUS.md), gaps: [GAPS.md](GAPS.md), MPM:
[MPM.md](../metalsim/physics/mpm/MPM.md).

## In progress

| Work | Branch | State | Next |
|---|---|---|---|
| MPM in the main simulator (MuJoCo Warp `BatchSim` + MPM, two-way proxy coupling as Isaac Lab 3.0 does; Newton 1.5.2 fork's implicit MPM vendored byte-identical) | `mpm-main-sim` (pushed `07d4c1c8`) | `MPMCoupling` + vendored Isaac MPM solver built; box on sand matches Isaac's coupling path. batch_01 running (~2-3 h): validates a Newton fork fix, re-pins, scale and timing, then first PR | API (materials, emitters, `MPMCoupling`), tests (angle of repose, momentum, determinism at 1/64/1,024 worlds, rigid neutrality, vendor-drift, license notices), first PR |
| Coupled MPM scenes slower than the L4 when captured (two-way 1.76x, ANYmal on sand 1.29x; MPM.md 8j) | `mpm-coupled-speed` | tiled IJtm product confirmed (two-way IJtm ~3x faster per call). batch_03 (~1.5 h) validating: tests, floors, Franka-Pour hashes, timing | row-tiled product in the Newton 1.5.2 fork, re-run the six floor comparisons + determinism, re-time |
| Franka-Pour vs Isaac: Isaac seeds 1 and 2 (seed study, PARITY §1b.5) | `franka-pour-isaac-seeds` (`5a28b053`) | training on both L4 VMs (seed 2 ETA ~18:21, seed 1 ETA ~19:14 PDT); VMs power off ~30 min after | compare our 5 seeds vs Isaac's 3, verdict, PR |
| Optimization pass on the MPM–BatchSim coupling layer (queued after the first MPM-main PR) | `mpm-main-sim` | queued | research, then optimize (timing on the 16-inch only) |
| Docs audit: every doc checked against the measured record | `docs-audit` | 15 docs corrected (STATUS rewritten to current main, README, GAPS, PARITY, GUIDE, CHANGELOG, RENDER_COVERAGE, MPM.md, …); batch re-measures test counts, then PR | merge |

## Known broken or behind

| Problem | Severity | Where | Owner / plan |
|---|---|---|---|
| Captured coupled MPM scenes slower than the L4 (two-way 1.76x, ANYmal 1.29x) | performance | MPM.md 8j | in progress above |
| Franka-Pour learns ~0.06 slower than Isaac at its 50-59 (5 of ours vs 1 Isaac seed; not yet settled) | parity, unconfirmed | PARITY §1b.5 | Isaac seeds above |
| Newton 1.5.2 Gauss-Seidel colouring with `separate_worlds` is invalid (strain nodes numbered world-first, coloured in cell order): data race, multi-world MPM runs not repeatable. Also in upstream Newton 1.5.2 and main. Found by code reading | correctness | Newton fork `metalsim-1.5.2-gs-envfirst` | fix being validated in MPM-main batch_01 |
| `tests/test_mpm.py::test_coupled_contact_filter_keeps_source_order` fails (coupled-solver contact-filter output out of order across threadgroups; `install(rigid_deterministic=True)` may not reach that module). Seen on the coupled-speed trees; no committed log shows it ever passing — possibly pre-existing on main | correctness (determinism), unconfirmed | tests/test_mpm.py | coupled-speed agent: base-vs-new run in batch_03, then root cause |
| MPM runs only through Newton (`.venv-newton152`), not in metalsim's own `BatchSim` scenes | missing feature | — | in progress above |
| Tier 1 draws MPM particles but without ray-traced shadows; the ray-traced sensors (depth camera, lidar) do not see them; tier 0 still uses the far-field approximation for area lights | render gaps | RENDER_COVERAGE.md 1c, 3e | not scheduled |
| Newer USD dome-light flavours (DomeLight_1 in Kit, `omni:rtx:usdluxVersion` >= 2505) use the measured legacy convention, unverified against Kit | render, unverified | research_notes/dome_light_azimuth/ | needs a Kit measurement on the L4 |
| MuJoCo Warp `BatchSim` is not run-to-run bitwise repeatable in its default mode (deterministic mode exists) | determinism | MPM.md 8e, PARITY | by design of the default; documented |
| Development runs in an org-managed sandbox: no Metal GPU, ssh or `gh` from the agent shell; GPU jobs, commits and pushes run as batch scripts by the owner | process | — | — |

Test suites on `main`: counts being re-measured by the docs-audit batch (an earlier figure here mixed overlapping suites). One MPM test is suspected failing, see above.

## How this file is kept current

Every batch of work that changes what is in flight or what is broken updates this file in the same batch
(`scripts/publish_now.sh` lands it on `main` as a doc-only PR merged immediately).
