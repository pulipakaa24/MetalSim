# Coupled-scene re-timing and MPM close-out: hand-off notes

Branch `mpm-closeout-retime` (from main 52469d93), PR #16. Machine: the 16-inch M4 Max (Mac16,5). Env:
`.venv-newton152`, which imports Warp 95f6f121, Newton 3fd1f052 and MuJoCo Warp 3.11.0 (`env_check.log`).

Done (all GPU work finished on 2026-10-13):
- `run.sh 3`: three interleaved repetitions, written to `summary.md` / `summary.json`. `profile.sh`: per-part and
  per-kernel profiles in `profile/`, joined with the L4's nsys sums in `profile/joined*.md`.
- MPM.md:
  - 8j: the timing table, the profile and what would close the gap.
  - 8d: the GPU-hang re-attribution.
  - Superseded "Open:" lines marked resolved.
  - 8i: a seed-study pointer.
- STATUS / GAPS MPM rows and CHANGELOG. PARITY §1b.5 now says the Isaac seeds are waiting for L4 time.
- RENDER_COVERAGE particle rows.

Verdict: MPM is not marked done. Two items are open:
1. The Isaac Franka-Pour seeds 1-4 (L4 time).
2. Captured coupled-scene speed: two-way 1.76x, ANYmal 1.29x the L4.

Next if picked up: the two-way IJtm row-tiled product (MPM.md 8j, item 1), then a floor re-check.
