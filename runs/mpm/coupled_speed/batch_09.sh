#!/bin/bash
# Coupled speed, docs and PR (2026-10-13). No GPU work. Prerequisite (checked): batch_08 done (pins committed).
#   1  commit the docs (MPM.md 8k, STATUS, GAPS, PARITY, README, CHANGELOG, research note) and the batch logs
#   2  merge the newest origin/main (stop on conflict; merge aborted)
#   3  check: vendored implicit_mpm still equals the pin; setup scripts and the vendor script agree
#   4  push mpm-coupled-speed; open the PR
# Log: runs/mpm/coupled_speed/batch_09.log. Resumable (markers in b09/).
set -u
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
W=.
R=$W/runs/mpm/coupled_speed
B=$R/b09
mkdir -p $B
exec > >(tee -a "$R/batch_09.log") 2>&1
echo "== $(date) batch_09 start"
stop() { echo "STOP: $*"; echo "== $(date) batch_09 stopped"; exit 1; }
cd $W || stop cd
grep -q 'batch_08 done' $R/batch_08.log || stop "batch_08 not done"
[ "$(git branch --show-current)" = mpm-coupled-speed ] || stop "not on mpm-coupled-speed"
if [ ! -f $B/docs_committed ]; then
  git add CHANGELOG.md README.md docs/GAPS.md docs/PARITY.md docs/STATUS.md metalsim/physics/mpm/MPM.md \
      research_notes/coupled_speed runs/mpm/coupled_speed
  git commit -s -q -m "MPM.md 8k: coupled scenes closer to the L4; STATUS, GAPS, PARITY, README, CHANGELOG

Two-way captured 54.8 -> 37.6 ms/frame (1.77x -> 1.21x the L4), eager 0.84x;
MPMCoupling coupled step 49.3 -> 34.2 ms (16 bodies); Franka-Pour unchanged;
ANYmal measured (GS launch / gs-soa: bitwise, not faster; the rest is the
MuJoCo Warp robot graph). Restores MPM.md's GS-fix history line to 99843d09
(the pin bump had rewritten it). Research note outcome section.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || stop "docs commit"
  touch $B/docs_committed
fi
git fetch -q origin || stop fetch
if [ ! -f $B/merged ]; then
  git merge -q --no-edit origin/main || { git merge --abort; stop "merging origin/main conflicts (aborted)"; }
  touch $B/merged
fi
echo "merged origin/main $(git rev-parse --short origin/main) -> $(git rev-parse --short HEAD)"
PIN_N=$(sed -n 's/^PIN=\([0-9a-f]\{40\}\).*/\1/p' scripts/setup_newton152.sh)
PIN_V=$(sed -n 's/^PIN = "\([0-9a-f]\{40\}\)".*/\1/p' scripts/vendor_mpm152.py)
PIN_W=$(sed -n 's/^PIN=\([0-9a-f]\{40\}\).*/\1/p' scripts/setup_warp.sh)
echo "pins: warp $PIN_W newton $PIN_N vendor $PIN_V"
[ "$PIN_N" = "$PIN_V" ] || stop "setup_newton152.sh and vendor_mpm152.py pins differ"
.venv/bin/python scripts/vendor_mpm152.py --check $W/upstream/trees/newton_fork || stop "vendored copy differs from the pin"
git push -q -u origin mpm-coupled-speed 2>&1 | grep -v '^remote:'
git log --oneline -3
if [ ! -f $B/pr_opened ]; then
  gh pr create --base main --head mpm-coupled-speed --title "MPM: coupled scenes closer to the L4 (Warp 31cf79d7, Newton addd10cb)" \
      --body-file $R/pr_body_coupled_speed.md && touch $B/pr_opened || echo "FAIL: gh pr create"
fi
gh pr view mpm-coupled-speed --json url,state,headRefOid 2>/dev/null
echo "== $(date) batch_09 done"
