# Now: what is being worked on, what is broken

Updated 2026-10-14 01:00 PDT; previously 00:25 PDT (main `d962060a`). This file is the single place that says what is in flight right
now and what is known to be broken. It is updated with every batch of work; if it is older than the newest merge on
`main`, treat it as stale. Longer-lived status: [STATUS.md](STATUS.md), gaps: [GAPS.md](GAPS.md), MPM:
[MPM.md](../metalsim/physics/mpm/MPM.md).

## In progress

| Work | Branch | State | Next |
|---|---|---|---|
| MPM in the main simulator (MuJoCo Warp `BatchSim` + MPM, two-way proxy coupling as Isaac Lab 3.0 does; Newton 1.5.2 fork's implicit MPM vendored byte-identical) | `mpm-main-sim` (PR #26) | done: MPMCoupling + vendored Isaac MPM solver (Newton fork 99843d09 with the GS colouring fix); box on sand matches Isaac's path; lagged exchange holds a resting box (10 mm, converged solve); captured coupled step 13.7 ms (1 world) to 224 ms (1,024 worlds) | merge; then deterministic draft PR rebuilt on main; coupling-layer optimisation |  API (materials, emitters, `MPMCoupling`), tests (angle of repose, momentum, determinism at 1/64/1,024 worlds, rigid neutrality, vendor-drift, license notices), first PR |
| GPU latency / occupancy (measured): coupled ANYmal ~30 ms/frame, GPU busy ~22 ms; the ~7.7 ms of in-frame GPU idle follows 17.5 host waits per frame (MuJoCo's captured solver loop ~8, sparse-grid allocation ~5), not command-buffer count (an earlier ~1,100/frame figure was a probe counting bug; real ~34) | `mpm-latency` | lanes closed; fewer command buffers are slower. batch_cb2 attributes each wait, prices skipped indirect ranges, re-checks captured vs eager sand with equal history (the earlier mismatch is probably a probe offset) | next: upstream Newton's capacity-bounded ("rebuildable") sparse grid for ANYmal's sand step, as Isaac Lab 3 uses: removes the ~5 grid host waits per frame and allows capturing the step (est. 29.9 → 24–26 ms; not bitwise vs today's grid, so an L4 run with the same grid is needed for parity); batch after batch_cb2. Robot-loop waits (est. 2.5–3.5 ms) on hold until #29's fix |
| Demo milestone: once the in-flight MPM / coupled-speed / Franka-Pour PRs land, tag main (`demo-2026-10`) with `docs/DEMO.md` (what works, how to run each demo, measured numbers, known limits) | — | planned | then a sand-fidelity workstream aiming at physically right sand, with the Isaac-parity mode kept selectable |
| Scene composition + filming: `python -m metalsim.render.film <task>` and `python -m metalsim.run scene.yaml` | `franka-pour-render` (PR #51), `scene-compose` (PR #52) | Franka-Pour film done: docs/renders/FRANKA_POUR.md on #51 (hero 1920x1080, 256 spp path traced, 3.96 s/frame; 2x2 grid 3.06 s/frame; 13.8 min end to end on the 16-inch). #52: 1 GPU test fails on both Macs (depth valid on exactly 1/3 of pixels); 14-inch install lacks Open Image Denoise and a module the scene tests need | root-cause the depth test, install fixes, then merge both |
| Gaussian-splat renderer published | `splat-renderer` (PR #48) | correctness: 14-inch splat tests 23 passed, render suite 179 passed; Metal vs float64 reference on InteriorGS 0204_840158 (972k splats, 640×360) PSNR ~59 dB, depth error < 1 µm; speed (16-inch): 0.68 ms for one camera, 0.43 ms/camera at 8 | review + merge; fetch script to explain a missing Hugging Face login (InteriorGS fetch failed on the 14-inch) |
| Docs audit: every doc checked against the measured record | `docs-audit` (PR #20) | 15 docs corrected (STATUS rewritten to current main, README, GAPS, PARITY, GUIDE, CHANGELOG, RENDER_COVERAGE, MPM.md, …); batch re-measures test counts, then PR | merge |

## Known broken or behind

| Problem | Severity | Where | Owner / plan |
|---|---|---|---|
| Captured coupled MPM scenes slower than the L4 (two-way 1.21x with the pending fixes, was 1.76x; ANYmal 1.21x) | performance | MPM.md 8j | in progress above |
| Franka-Pour vs Isaac: no difference at iterations 0-49; at 50-59 Isaac +0.035 success (p = 0.09, 5 vs 3 seeds), neither shown nor ruled out | parity, accepted as open | PARITY §1b.5 | owner decided 2026-10-13 not to run Isaac seeds 3-4 |
| With the staggered MPM–rigid exchange (Isaac Lab 3.0's Franka-Pour setting and Isaac's own path) a body resting on sand sinks through any bed (force fed back one step late). Lagged exchange (now `MPMCoupling`'s default) carries a resting box: 10 mm settling with a converged solve (250 it, 1e-6), 25 mm and creeping at 50 iterations; drops penetrate 32–39 mm vs 10–25 mm expected | physics fidelity (coupling + solve convergence) | MPM.md §11 | documented; Franka-Pour keeps Isaac's setting for parity |
| Newton 1.5.2 implicit-MPM sand's angle of repose is 7–9° below atan(μ) (μ 0.4/0.68/0.9: 14.7° / 25–27° / 32.6° vs 21.8° / 34.2° / 42.0°), insensitive to lift speed and iterations, ~1° per resolution step. Same solver as Isaac Lab | physics fidelity (solver) | MPM.md §11 | documented; resolution R/dx 20 being measured |
| Newton 1.5.2 Gauss-Seidel colouring with `separate_worlds` is invalid (strain nodes numbered world-first, coloured in cell order): data race, multi-world MPM runs not repeatable. Also in upstream Newton 1.5.2 and main. Found by code reading | correctness | Newton fork `metalsim-1.5.2-gs-envfirst` | fix being validated in MPM-main batch_01 |
| `scripts/gpu_run.sh` ran jobs in the main checkout, so `python -m pytest` from a worktree imported stale main-checkout code (caused a false test failure) | test harness | scripts/gpu_run.sh | fix PR `fix/gpu-run-caller-cwd` (run with batch_07); affected runs re-run and confirmed clean |
| MPM runs only through Newton (`.venv-newton152`), not in metalsim's own `BatchSim` scenes | missing feature | — | in progress above |
| Tier 1 draws MPM particles but without ray-traced shadows; the ray-traced sensors (depth camera, lidar) do not see them; tier 0 still uses the far-field approximation for area lights | render gaps | RENDER_COVERAGE.md 1c, 3e | not scheduled |
| Newer USD dome-light flavours (DomeLight_1 in Kit, `omni:rtx:usdluxVersion` >= 2505) use the measured legacy convention, unverified against Kit | render, unverified | research_notes/dome_light_azimuth/ | needs a Kit measurement on the L4 |
| MuJoCo Warp `BatchSim` is not run-to-run bitwise repeatable in its default mode. Opt-in deterministic mode passes all suites and costs +3.4 % / +4.0 % on coupled box scenes, but **fails on the G1 when captured** | determinism | GAPS, research_notes/MPM in main simulator/ | PR #29 draft. batch_16: with the solver early exit and the fused update both off the G1 passes at 64 worlds but still trips at 1,024 (a third cause); turning both off also costs +116 % (64 worlds) / +22 % (1,024) on the coupled box, so switching them off is not the fix. Naming the 1,024-world kernel and re-weighing a kernel-level fix |
| `.venv-newton152` imported stock Newton 1.5.2 from 2026-10-13 19:31 until the repair | correctness (local env) | .venv-newton152 | repaired 23:12; guard #47 merged; Franka-Pour 8/8 on the repaired env; cause of the 19:31 rewrite not identified |
| This Mac's shared `.venv-newton152` lacks pyobjc/ninja (it was set up from an old checkout; main's setup script installs them). Setup script gains a final import check | install (local) | scripts/setup_newton152.sh | PR `newton152-verify` + queued venv repair |
| Fresh install from GitHub, both Macs, brand-new clone: every step and all six demos pass; Franka-Pour 8/8; the last 2 skips (Newton bridge) fixed by #46, merged 2026-10-14 | install | PRs #43 #45 #46 merged | one confirming fresh run, then demo tag. Open: the film tool needs Open Image Denoise, which the install does not provide |
| Newton 1.7's SolverMuJoCo vs the MuJoCo Warp fork pin | compatibility | PR #46 | fixed and merged 2026-10-14 |
| Native-texture work (`render/textures.py`, `TEXTURES.md`, texture tests, a basis shim, edits to tier0) exists only uncommitted in this Mac's main checkout, against an old base | unpublished work | main checkout | needs an owner: port onto main or drop |
| Development runs in an org-managed sandbox: no Metal GPU, ssh or `gh` from the agent shell; GPU jobs, commits and pushes run as batch scripts by the owner | process | — | — |

Test suites (measured 2026-10-13 14:49 on main `3670f4b8`): rigid suite 424 passed, 10 skipped, 0 failed; `tests/test_mpm.py` 15 passed, 1 expected xfail. The contact-filter test above passed in this run (warm kernel cache).

## How this file is kept current

Every batch of work that changes what is in flight or what is broken updates this file in the same batch
(`scripts/publish_now.sh` lands it on `main` as a doc-only PR merged immediately).
