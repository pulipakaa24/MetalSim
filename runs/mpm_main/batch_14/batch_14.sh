#!/bin/bash
# batch 14 (run outside the agent sandbox, after batch_13):
#   bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b14/batch_14.sh
#   FIX_N152=1 bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b14/batch_14.sh   # + (b)'s repair
#
# Owners and causes for batch_12's 15 skips (det worktree, branch batchsim-deterministic):
#  (a) 8 x "needs the Franka-Pour assets" (test_franka_pour_terms 7, test_franka_pour_resume 1): the Franka-Pour files are
#      fetched (scripts/fetch_isaac_assets.sh, SHA-256 checked) and gitignored; neither this worktree nor the main
#      checkout has them. This batch fetches them into the det worktree with that script and re-runs both files.
#  (b) 5 x "needs the (Warp and) Newton fork (metalsim-1.5.2)" (test_mpm.py): this Mac's .venv-newton152 imports Newton
#      from upstream/newton-1.5.2 (newton.pth, rewritten 2026-10-13 19:31), which is STOCK Newton v1.5.2 (cf5378db), not
#      the fork; scripts/setup_newton152.sh installs the fork from upstream/newton-metalsim-1.5.2, which on this Mac sits
#      at the previous pin 3fd1f052 instead of addd10cb. The gates test the fork's API (solve_rheology._color_block_count,
#      active_node_bound). Repair (only with FIX_N152=1; it changes a shared environment): check out the pinned commit in
#      upstream/newton-metalsim-1.5.2 (refused if that tree is dirty) and install it editable into .venv-newton152, as
#      setup_newton152.sh does, through the GPU queue; then re-run tests/test_mpm.py.
#  (c) 2 x "SolverMuJoCo unavailable" (test_render_newton_bridge, test_newton_bridge_poses): Newton 1.7 against MuJoCo
#      Warp 3.14 (contact_force_fn); batch_13 runs batch_11's pin move, which reinstalls .venv's newton at the fixed
#      commit. Both files are re-run here; any "SolverMuJoCo unavailable" left is a FAIL.
# Every pytest runs with -rs (skip reasons printed); logs committed on batchsim-deterministic; result posted on #29.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b14
L=$S/logs
O=$W/runs/mpm_main/batch_14
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_14 start $(date) FIX_N152=${FIX_N152:-0}"
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
gpu() {   # gpu NAME MIN LOG -- cmd...   (cwd and PYTHONPATH = the det worktree)
  local name=$1 min=$2 log=$3; shift 4
  say "gpu job $name (<= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" check "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$W" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -2 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
check_suite() {   # check_suite NAME LOG FORBIDDEN_SKIP_REGEX: PASS = rc 0 and no skip whose reason matches
  local name=$1 log=$2 rc=$3 forbid=$4 s
  s=$(grep -E "passed|failed|error" "$log" | tail -1)
  if [ "$rc" = 0 ] && ! grep -E "^SKIPPED" "$log" | grep -qE "$forbid"; then mark "$name" PASS "$s"
  else mark "$name" FAIL "rc=$rc $s; skips: $(grep -E '^SKIPPED' "$log" | cut -c1-120 | sort | uniq -c | tr '\n' ';')"; fi
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
for p in assets/isaac/G1 upstream/mujoco_menagerie assets/polyhaven assets/tron1/scenes_local; do
  [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"
done
health

# ---------------------------------------------------------------- (a) Franka-Pour assets
if ! done_ FETCH14; then
  ( cd "$W" && zsh scripts/fetch_isaac_assets.sh ) > "$L/fetch_isaac_assets.log" 2>&1
  rc=$?; tail -5 "$L/fetch_isaac_assets.log"
  [ $rc = 0 ] && [ -f "$W/assets/isaac/franka_pour/reset_dataset.pt" ] && mark FETCH14 PASS "SHA-256 checked" \
      || { mark FETCH14 FAIL "rc=$rc (see $L/fetch_isaac_assets.log; network?)"; }
fi
if done_ FETCH14 && ! done_ POUR14; then
  gpu mm14_pour 60 "$L/franka_pour_tests.log" -- "$PY152" -P -m pytest -q -rfEs tests/test_franka_pour_terms.py tests/test_franka_pour_resume.py
  check_suite POUR14 "$L/franka_pour_tests.log" $? "Franka-Pour assets"
fi

# ---------------------------------------------------------------- (b) .venv-newton152 on the Newton fork
N152=$R/upstream/newton-metalsim-1.5.2
PIN152=$(sed -n 's/^PIN=\([0-9a-f]\{40\}\).*/\1/p' "$W/scripts/setup_newton152.sh")
"$PY152" -P -c "import newton, os; from newton._src.solvers.implicit_mpm import solve_rheology as sr; print('newton', os.path.dirname(newton.__file__), 'fork API', hasattr(sr, '_color_block_count'))" \
    > "$L/n152_before.log" 2>&1
cat "$L/n152_before.log"
echo "fork dir $N152 at $(git -C "$N152" rev-parse --short HEAD 2>/dev/null), pin ${PIN152:0:8}" | tee -a "$L/n152_before.log"
if [ "${FIX_N152:-0}" = 1 ] && ! done_ N152FIX14; then
  if [ -n "$(git -C "$N152" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    mark N152FIX14 FAIL "$N152 has local changes: not touched"
  else
    cat > "$S/n152_fix.sh" <<EOF
#!/bin/bash
set -ex
cd $R
git -C upstream/newton-metalsim-1.5.2 fetch -q origin
git -C upstream/newton-metalsim-1.5.2 checkout -q --detach $PIN152
uv pip install -q --python .venv-newton152/bin/python --no-deps -e upstream/newton-metalsim-1.5.2
.venv-newton152/bin/python -P -c "import newton, os; from newton._src.solvers.implicit_mpm import solve_rheology as sr; print('newton', os.path.dirname(newton.__file__), 'fork API', hasattr(sr, '_color_block_count'))"
EOF
    "$R/scripts/gpu_run.sh" n152_fix14 check 10 -- bash "$S/n152_fix.sh" > "$L/n152_fix.log" 2>&1
    cat "$L/n152_fix.log"
    grep -q "newton-metalsim-1.5.2/newton fork API True" "$L/n152_fix.log" && mark N152FIX14 PASS "newton-metalsim-1.5.2 at ${PIN152:0:8}" \
        || mark N152FIX14 FAIL "see $L/n152_fix.log"
  fi
fi
if ! done_ MPM14; then
  gpu mm14_mpm 30 "$L/test_mpm.log" -- "$PY152" -P -m pytest -q -rfEs tests/test_mpm.py
  check_suite MPM14 "$L/test_mpm.log" $? "fork"
fi

# ---------------------------------------------------------------- (c) Newton bridge tests after batch_13's pin move
if ! done_ BRIDGE14; then
  gpu mm14_bridge 30 "$L/newton_bridge_tests.log" -- "$PY" -P -m pytest -q -rfEs tests/test_render_newton_bridge.py \
      tests/test_newton_bridge_poses.py tests/test_render_newton_perworld.py tests/test_g1_newton_engine.py
  check_suite BRIDGE14 "$L/newton_bridge_tests.log" $? "SolverMuJoCo unavailable|g1_minimal"
fi

cp -R "$L" "$O/"; cp "$S/batch_14.sh" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add runs/mpm_main/batch_14
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: owners for batch 12's skips (batch 14)

Franka-Pour assets fetched into the worktree and their tests re-run;
.venv-newton152's Newton checked (it imported stock v1.5.2, not the
fork) and test_mpm.py re-run; the Newton-bridge tests re-run after the
Newton 1.7 / MuJoCo Warp 3.14 pin move.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
{
  echo "Batch 14 (16-inch): causes for batch 12's skips. (a) Franka-Pour assets fetched (SHA-256 checked). (b) .venv-newton152 imported stock Newton v1.5.2, not the metalsim-1.5.2 fork. (c) SolverMuJoCo against MuJoCo Warp 3.14, after the Newton pin move."
  echo; echo '```'; cat "$STATUS"; echo '```'
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr comment 29 --body-file "$S/pr29_comment.md" ) > "$L/pr.log" 2>&1
say "=== batch_14 end $(date)"; cat "$STATUS"
