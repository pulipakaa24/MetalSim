#!/bin/bash
# batch 10b (run outside the agent sandbox):  bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b10b/batch_10b.sh
#
# Re-run of batch_10's deterministic-mode cost (DETCOST10 failed: the G1 runs could not open
# assets/isaac/G1/g1_minimal.usd, a fetched, gitignored asset that this worktree did not have). The worktree now links
# assets/isaac/G1 to the main checkout's copy (created again here if missing; never committed). The coupled-box runs of
# batch_10 succeeded and are kept (runs/mpm_main/batch_10_det/box_*.json); only the 12 G1 runs are made: deterministic
# on / off at 64 and 1,024 worlds, 3 interleaved repetitions, one process each, timing class. Then the numbers go into
# the research note, GAPS and #29's body (scratch/b10/det_docs.py), commit, push. Resumable. ~30-45 min of GPU.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b10b
L=$S/logs
OD=$W/runs/mpm_main/batch_10_det
B8=$W/scratch/b08b
PY=$R/.venv/bin/python
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$OD"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_10b start $(date)"
say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE)" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }
health() {
  local u top
  u=$("$R/scripts/gpu_util.sh" 2>/dev/null); top=$(python3 "$R/scripts/gpu_top.py" 2>/dev/null | head -4)
  say "health: utilisation ${u:-?}%"; echo "$top"
  if [ "${u:-0}" -ge 10 ] && echo "$top" | grep -q "none >= 1 %"; then
    sleep 30; u=$("$R/scripts/gpu_util.sh" 2>/dev/null)
    if [ "${u:-0}" -ge 10 ] && python3 "$R/scripts/gpu_top.py" 2>/dev/null | grep -q "none >= 1 %"; then
      mark HEALTH FAIL "GPU ${u}% busy with no owning process: stopping"; exit 90
    fi
  fi
}
GPYP=$W/upstream/mujoco_warp-detccd:$W/upstream/warp-tripwire-git:$W
gpu() {   # gpu NAME KIND MIN LOG -- cmd...   (cwd = the det worktree, the new pins' clones first on PYTHONPATH)
  local name=$1 kind=$2 min=$3 log=$4; shift 5
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$GPYP" "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -3 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
[ -e "$W/assets/isaac/G1" ] || ln -s "$R/assets/isaac/G1" "$W/assets/isaac/G1"
[ -f "$W/assets/isaac/G1/g1_minimal.usd" ] || { mark ASSET FAIL "no assets/isaac/G1/g1_minimal.usd (main checkout: scripts/fetch_isaac_assets.sh)"; exit 1; }
health
if ! done_ DETCOST10B; then
  fails=0
  for r in 1 2 3; do
    for n in 64 1024; do
      for d in 0 1; do
        [ -s "$OD/g1_${n}_${d}_r$r.json" ] || gpu mm10b_g1_${n}_${d}_r$r timing 20 "$L/g1_${n}_${d}_r$r.log" -- "$PY" -P \
            "$W/scripts/diagnostics/det_cost_g1.py" --worlds $n --deterministic $d --frames 100 --warmup 20 \
            --json "$OD/g1_${n}_${d}_r$r.json" || fails=$((fails + 1))
      done
    done
  done
  if [ $fails = 0 ] && python3 "$W/scratch/b10/det_docs.py" "$W" "$OD" "$B8/pr29_body.md" > "$L/det_docs.log" 2>&1; then
    cat "$L/det_docs.log"; mark DETCOST10B PASS "$(grep -E '^\| (G1|coupled)' "$L/det_docs.log" | cut -d'|' -f2,3,6 | tr '\n' ';')"
  else
    cat "$L/det_docs.log" 2>/dev/null; mark DETCOST10B FAIL "$fails failed runs (see $L)"
  fi
fi
mkdir -p "$OD/logs_10b"; cp -R "$L/." "$OD/logs_10b/"; cp "$S/batch_10b.sh" "$STATUS" "$OD/logs_10b/" 2>/dev/null
git -C "$W" add runs/mpm_main/batch_10_det
done_ DETCOST10B && git -C "$W" add docs/GAPS.md "research_notes/MPM in main simulator/deterministic_ccd_divergence.md"
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: cost of the deterministic mode, G1 runs (batch 10b)

The G1 flat-terrain runs of the deterministic-mode cost (batch 10's
failed: the worktree lacked the fetched G1 USD asset); the numbers of
both scenes in the research note and GAPS.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
done_ DETCOST10B && ( cd "$W" && gh pr edit 29 --body-file "$B8/pr29_body.md" ) > "$L/pr29.log" 2>&1
say "=== batch_10b end $(date)"; cat "$STATUS"
