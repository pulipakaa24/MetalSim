#!/bin/bash
# Commit and push the MPM close-out branch, then update PR #16 (title, body) and mark it ready for review.
# No GPU work: the timing and profile jobs already finished. Idempotent: commits only when something changed.
# Log: runs/mpm/coupled_retime/batch_01.log (in the worktree).
set -u
W=.
LOG=$W/runs/mpm/coupled_retime/batch_01.log
exec > >(tee "$LOG") 2>&1
cd "$W" || exit 1
echo "== $(date) batch_01"
git rev-parse --abbrev-ref HEAD
git add CHANGELOG.md docs/GAPS.md docs/STATUS.md docs/PARITY.md docs/RENDER_COVERAGE.md \
    metalsim/physics/mpm/MPM.md runs/mpm/coupled_retime runs/rigid17/out
git status --short
if ! git diff --cached --quiet; then
  git commit -s -q -F - <<'MSG' || { echo "commit FAILED"; exit 1; }
MPM close-out: docs to the measured record, coupled scenes re-timed

STATUS / GAPS MPM rows: all six floor scenes inside the L4 ensemble
(multi_material 0.80, ANYmal 0.94), current pins (Warp 95f6f121, Newton
3fd1f052; mesh-SDF barrier fix verified in both pinned 1.7 commits),
Franka-Pour status. MPM.md: superseded Open lines marked resolved; the
2026-10-12 02:32 hang re-attributed to the 25M-instance particle build
(strong circumstantial evidence); new section 8j re-times two-way and
ANYmal on the 16-inch against the L4 (captured 1.76x / 1.29x, eager
1.09x / 0.67x) with per-kernel profiles. RENDER_COVERAGE: particle
rendering merged (T0 / T2).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
fi
git log --oneline -3
git push origin mpm-closeout-retime || { echo "push FAILED"; exit 1; }
gh pr edit 16 --title "MPM close-out: docs to the measured record; coupled scenes re-timed vs the L4" \
  --body-file "$W/runs/mpm/coupled_retime/pr_body.md" || echo "pr edit FAILED"
gh pr ready 16 || echo "pr ready FAILED (maybe already ready)"
gh pr view 16 --json url,state,isDraft,headRefOid
echo "== done $(date)"
