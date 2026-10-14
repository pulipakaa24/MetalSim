## Deterministic BatchSim (draft: open until the stacked-boxes divergence is fixed)

`BatchSimOptions(deterministic=True)`: MuJoCo Warp's kernels in Warp's RUN_TO_RUN mode (as Newton's
`SolverMuJoCo(deterministic=...)`), set before MuJoCo Warp's first launch, one mode per process (a mix raises), Newton's
record bound held while generated kernels are created, a post-capture check of the replayed modules, and Warp's
deterministic overflow flag in `check_overflow`. Off by default; the default path launches what it did before.

Status (runs on branch `mpm-main-sim`, `runs/mpm_main/batch_02`, `batch_03`):
* MuJoCo Warp alone is not run-to-run repeatable on Metal: a box resting on the floor diverges at step 18 at 1,024
  worlds by default; with this option it repeats (3 runs, 4 / 64 / 1,024 worlds), and so does the coupled MPM scene.
* **Open:** two stacked boxes at 256 worlds still diverge in 2 worlds from step 139, identically in-process and across
  cold / warm kernel caches. Suspect (source reading, `research_notes/MPM in main simulator/deterministic_ccd_divergence.md`):
  MuJoCo Warp's box-box CCD kernel has two consumed counters, and under Warp's two-phase deterministic counters its
  phase 0 runs EPA on suppressed / uninitialised scratch, so phase 1 can make a different number of contact-slot
  reservations than phase 0 counted, which nothing checks. Next: A/B with native CCD disabled, a contact-slot audit,
  a phase-mismatch tripwire in the Warp fork, and a split of the CCD kernel in the MuJoCo Warp fork.
* `check_determinism` must also cover MuJoCo Warp's generated (`module="unique"`) kernels.

Tests: `tests/test_batch_deterministic.py` (each case in its own process; cold / cold / warm kernel cache; refusal on
mixed modes). The `stack, 256` case fails until the fix lands, which is why this PR stays a draft.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
