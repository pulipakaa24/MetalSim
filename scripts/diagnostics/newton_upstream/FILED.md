# Newton upstream: filed issues and pull requests (2026-09-25)

Filed on newton-physics/newton under the user's account, from the drafts in this directory. Bodies, titles and the
exact reproducers are in `filed/` (`issue_N.md`, `pr_N.md`, `repro_*.py`). Reproducers run with
`pip install "newton @ git+https://github.com/newton-physics/newton@45458023f24629c1feffb1f336594808f537db14"`
(warp-lang 1.17.0, CPU).

## Issues

| # | topic | URL |
|---|---|---|
| 1 | revolute angle wrap for limit ranges beyond ±π | https://github.com/newton-physics/newton/issues/4313 |
| 2 | joint relaxation applies linear/angular factors inconsistently (torque 1.36x / gravity 0.78x at the defaults) | https://github.com/newton-physics/newton/issues/4314 |
| 3 | XPBD drive stiffness depends on the iteration count (71-1240 N m/rad for ke 200 over 1-16 it) | https://github.com/newton-physics/newton/issues/4315 |

Related existing issue: #2933 (same missing multiplier accumulation for springs / bending / tets).

## Pull requests (base: newton-physics/newton main @ 45458023; head: pulipakaa24/newton)

| # | branch | commits | fixes | URL |
|---|---|---|---|---|
| 1 | `fix/xpbd-revolute-angle-wrap` | 9fc468c6 | #4313 (+ small-swing rescale) | https://github.com/newton-physics/newton/pull/4316 |
| 2 | `fix/xpbd-joint-relaxation` | 4013f8ad | #4314 | https://github.com/newton-physics/newton/pull/4317 |
| 3 | `feat/xpbd-pd-joint-drive` (stacked on 1 and 2) | 9fc468c6, f594c4c0, c5e9b544 | #4315 | https://github.com/newton-physics/newton/pull/4318 |

- PR 3 keeps the former drive as default (`joint_drive_mode="compliance"`); "pd" / "implicit" are opt-in.
- Not upstreamed (MetalSim-specific for now): joint colouring, joint-only extra iterations, per-body contact forces.
- Newton uses EasyCLA (Linux Foundation), not DCO: the EasyCLA check on all three PRs reads "Missing CLA Authorization"
  until the account owner signs the individual CLA via the link in the check (only the account owner can sign).
  CI for external PRs also waits for a maintainer's manual approval. PR commits use the GitHub noreply address.
- PR 1 found and fixes a second, pre-existing bug: swings under ~0.02 rad were not rescaled from quaternion space
  (half the angle), so a compliance drive near its target at angle 0 settled at a different stiffness (613 vs
  1240 N m/rad at 16 it for target 0 vs 0.5).
- The fork's `metalsim` branch (90e23324, installed in .venv-newtonfork) is unchanged by this; it additionally
  defaults to `joint_drive_mode="pd"` and carries the MetalSim-specific options. It does NOT contain PR 1's
  small-swing rescale fix (only matters for the compliance drive, which MetalSim does not use).
