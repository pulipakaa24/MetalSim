# Now: what is being worked on, what is broken

Updated 2026-10-13 14:20 PDT (main `3670f4b8`). This file is the single place that says what is in flight right
now and what is known to be broken. It is updated with every batch of work; if it is older than the newest merge on
`main`, treat it as stale. Longer-lived status: [STATUS.md](STATUS.md), gaps: [GAPS.md](GAPS.md), MPM:
[MPM.md](../metalsim/physics/mpm/MPM.md).

## In progress

| Work | Branch | State | Next |
|---|---|---|---|
| MPM in the main simulator (MuJoCo Warp `BatchSim` + MPM, two-way proxy coupling as Isaac Lab 3.0 does; Newton 1.5.2 fork's implicit MPM vendored byte-identical) | `mpm-main-sim` (pushed `07d4c1c8`) | coupler working; box-on-sand matches Newton's coupled result | API (materials, emitters, `MPMCoupling`), tests (angle of repose, momentum, determinism at 1/64/1,024 worlds, rigid neutrality, vendor-drift, license notices), first PR |
| Coupled MPM scenes slower than the L4 when captured (two-way 1.76x, ANYmal on sand 1.29x; MPM.md 8j) | `mpm-coupled-speed` (new) | cause of most of two-way located: Metal's one-thread-per-row `_bsr_mv_listed_rows` grows 0.6 -> 19 ms/frame | row-tiled product in the Newton 1.5.2 fork, re-run the six floor comparisons + determinism, re-time |
| Franka-Pour vs Isaac: Isaac seeds 1 and 2 (seed study, PARITY §1b.5) | `franka-pour-isaac-seeds` (`5a28b053`) | training on both L4 VMs (seed 2 ETA ~18:21, seed 1 ETA ~19:14 PDT); VMs power off ~30 min after | compare our 5 seeds vs Isaac's 3, verdict, PR |
| Docs audit: every doc checked against the measured record | (new) | starting | PR with corrections |

## Known broken or behind

| Problem | Severity | Where | Owner / plan |
|---|---|---|---|
| Captured coupled MPM scenes slower than the L4 (two-way 1.76x, ANYmal 1.29x) | performance | MPM.md 8j | in progress above |
| Franka-Pour learns ~0.06 slower than Isaac at its 50-59 (5 of ours vs 1 Isaac seed; not yet settled) | parity, unconfirmed | PARITY §1b.5 | Isaac seeds above |
| MPM runs only through Newton (`.venv-newton152`), not in metalsim's own `BatchSim` scenes | missing feature | — | in progress above |
| Tier 1 and the ray-traced sensors (depth camera, lidar) do not see MPM particles; tier 0 still uses the far-field approximation for area lights | render gaps | RENDER_COVERAGE.md 1c, 3e | not scheduled |
| Newer USD dome-light flavours (DomeLight_1 in Kit, `omni:rtx:usdluxVersion` >= 2505) use the measured legacy convention, unverified against Kit | render, unverified | research_notes/dome_light_azimuth/ | needs a Kit measurement on the L4 |
| MuJoCo Warp `BatchSim` is not run-to-run bitwise repeatable in its default mode (deterministic mode exists) | determinism | MPM.md 8e, PARITY | by design of the default; documented |
| Development runs in an org-managed sandbox: no Metal GPU, ssh or `gh` from the agent shell; GPU jobs, commits and pushes run as batch scripts by the owner | process | — | — |

Test suites on `main`: zero known failures (rigid 415 passed, render 173, MPM 15 + 1 expected xfail, as of PR #12).

## How this file is kept current

Every batch of work that changes what is in flight or what is broken updates this file in the same batch
(`scripts/publish_now.sh` lands it on `main` as a doc-only PR merged immediately).
