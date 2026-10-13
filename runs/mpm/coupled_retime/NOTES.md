# Coupled-scene re-timing and MPM close-out: hand-off notes

Branch `mpm-closeout-retime` (from main 52469d93). Machine: the 16-inch M4 Max (Mac16,5). Envs: `.venv-newton152`,
which imports Warp 95f6f121, Newton 3fd1f052 and MuJoCo Warp 3.11.0 from PyPI (`env_check.log`).

Done:
- MPM.md: superseded "Open:" lines marked resolved with pointers (sections 5, 7, 1b). 8d hang paragraph updated with
  the queue-history excerpt (`runs/rigid17/out/gpu_history_2026-10-12_0130-0300.txt`, `s17_base.watch`).
- RENDER_COVERAGE: particle rendering rows updated (PRs #4, #9, #14, #15; T0 and T2 only).
- Verified: both pinned Newton 1.7 commits (a4be3604, 9f317390) contain the mesh-SDF barrier fix; every pinned fork
  commit is on GitHub.
- Timing: `run.sh 3` (interleaved reps; L = the L4 protocol, S = the speed_suite protocol), logs and JSON here.

Next:
- Finish rep 3 of `run.sh` (re-run `bash run.sh 3`: it skips finished runs), then `profile.sh` (parts + per-kernel
  profiles), then `python3 summarize.py --json summary.json`.
- Section 8j in MPM.md (timing table + profile), STATUS.md line ~93, GAPS.md MPM row, CHANGELOG, PR.
