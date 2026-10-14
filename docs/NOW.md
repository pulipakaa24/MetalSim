# Now: what is being worked on, what is broken

Updated 2026-10-14 00:05 PDT; previously 2026-10-13 23:15 PDT (main `d962060a`). This file is the single place that says what is in flight right
now and what is known to be broken. It is updated with every batch of work; if it is older than the newest merge on
`main`, treat it as stale. Longer-lived status: [STATUS.md](STATUS.md), gaps: [GAPS.md](GAPS.md), MPM:
[MPM.md](../metalsim/physics/mpm/MPM.md).

## In progress

| Work | Branch | State | Next |
|---|---|---|---|
| MPM in the main simulator (MuJoCo Warp `BatchSim` + MPM, two-way proxy coupling as Isaac Lab 3.0 does; Newton 1.5.2 fork's implicit MPM vendored byte-identical) | `mpm-main-sim` (PR #26) | done: MPMCoupling + vendored Isaac MPM solver (Newton fork 99843d09 with the GS colouring fix); box on sand matches Isaac's path; lagged exchange holds a resting box (10 mm, converged solve); captured coupled step 13.7 ms (1 world) to 224 ms (1,024 worlds) | merge; then deterministic draft PR rebuilt on main; coupling-layer optimisation |  API (materials, emitters, `MPMCoupling`), tests (angle of repose, momentum, determinism at 1/64/1,024 worlds, rigid neutrality, vendor-drift, license notices), first PR |
| GPU latency / occupancy (measured): coupled ANYmal ~31 ms/frame, GPU busy only ~21 ms; the idle ~10 ms tracks ~1,100 command buffers per frame (~9–15 µs GPU idle at each boundary), set by the Warp fork's commit policy and eager launches | `mpm-latency` | ordering lanes closed (Apple GPU ran 0 ms of two lanes concurrently, one queue or two). batch_cb1 tests fewer, larger command buffers (existing fork settings, must stay bitwise) against capturing the sand step as one graph | commit-policy default change (est. 3–6 ms) or sand-step capture (est. toward ~21–23 ms, L4 24.2) depending on the result |
| Demo milestone: once the in-flight MPM / coupled-speed / Franka-Pour PRs land, tag main (`demo-2026-10`) with `docs/DEMO.md` (what works, how to run each demo, measured numbers, known limits) | — | planned | then a sand-fidelity workstream aiming at physically right sand, with the Isaac-parity mode kept selectable |
| Scene composition + filming: `python -m metalsim.render.film <task>` (PR 1) and `python -m metalsim.run scene.yaml` (PR 2, stacked) | `franka-pour-render`, `scene-compose` | built, CPU tests pass; first film batch stopped before doing anything (worktree on the wrong branch) | fixed batch, then films and PRs |
| Publish the Gaussian-splat renderer (until now uncommitted on one machine): code, shaders, tests, docs; reference scenes InteriorGS 0204_840158 (non-commercial, fetched with the user's own token, never committed) and a CC0 dx.gl object | `splat-renderer` | ported onto main, CPU tests pass | GPU batch (correctness on the 14-inch, timing on the 16-inch), then PR |
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
| MuJoCo Warp `BatchSim` is not run-to-run bitwise repeatable in its default mode. Opt-in deterministic mode passes all suites and costs +3.4 % / +4.0 % on coupled box scenes, but **fails on the G1 when captured** | determinism | GAPS, research_notes/MPM in main simulator/ | PR #29 draft. Two independent causes, both MetalSim speed-ups: the captured solver loop's early exit and the fused solver update; with both off it passes at 64 worlds. batch_16 confirms, runs 1,024 worlds and measures the cost of turning both off in deterministic mode only (proposed fix) |
| `.venv-newton152` imported stock Newton 1.5.2 from 2026-10-13 19:31 until the repair | correctness (local env) | .venv-newton152 | repaired 23:12 (fork `addd10cb`, test_mpm 15 passed); only the MPM-main agent's own runs were affected; guard PR #47 (stock env fails tests, setup refuses a checkout off the pin); cause of the 19:31 rewrite not identified |
| This Mac's shared `.venv-newton152` lacks pyobjc/ninja (it was set up from an old checkout; main's setup script installs them). Setup script gains a final import check | install (local) | scripts/setup_newton152.sh | PR `newton152-verify` + queued venv repair |
| Fresh install from GitHub, both Macs, brand-new clone with PR #45's fixes (Isaac Lab source fetch, `.[full]` extra): every step and all six demos pass; Franka-Pour 8/8; 2 skips left on each Mac, both the Newton 1.7 / MuJoCo Warp mismatch | install | PRs #43, #45 | merge; the 2 skips clear when the Newton pin move (#46) is fixed and merged; then demo tag |
| Newton 1.7's SolverMuJoCo vs the MuJoCo Warp fork pin | compatibility | PR #46 (pin move) | pin move done, bridge tests pass (31). (A Franka-Pour failure first blamed on it ran on the stock `.venv-newton152` before its repair; Newton 1.7 is not involved; re-run in batch_16) | merge #46 after batch_16's re-run |
| Native-texture work (`render/textures.py`, `TEXTURES.md`, texture tests, a basis shim, edits to tier0) exists only uncommitted in this Mac's main checkout, against an old base | unpublished work | main checkout | needs an owner: port onto main or drop |
| Development runs in an org-managed sandbox: no Metal GPU, ssh or `gh` from the agent shell; GPU jobs, commits and pushes run as batch scripts by the owner | process | — | — |

Test suites (measured 2026-10-13 14:49 on main `3670f4b8`): rigid suite 424 passed, 10 skipped, 0 failed; `tests/test_mpm.py` 15 passed, 1 expected xfail. The contact-filter test above passed in this run (warm kernel cache).

## How this file is kept current

Every batch of work that changes what is in flight or what is broken updates this file in the same batch
(`scripts/publish_now.sh` lands it on `main` as a doc-only PR merged immediately).
