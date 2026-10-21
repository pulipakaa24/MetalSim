#!/bin/bash
# batch 21 (run outside the agent sandbox):  bash scratch/b21/batch_21.sh
#
# #29 merge (9c090532, local): why test_policy_step_graph_matches_eager and test_one_world_masked_reset fail.
# Their probes' last stdout line was Warp's kernel-cache banner, so each died right after Warp initialised.
# Candidates and what tells them apart:
#   (a) the merge: main's probes (one_world_reset.py, policy_step_ab.py, #33) build an argparse.Namespace without
#       `deterministic`, which the branch's scale_coupled.build() now reads (a.deterministic). Predicts AttributeError
#       in the probe output, and main passes. (launch_reset's det_scope() is a no-op in default mode: _det_records None.)
#   (b) the MuJoCo Warp pin 8eb51fe: predicts plain main (df9485a8, the shared .venv now on 8eb51fe) fails the same two
#       tests, and passes with the old pin 4d05cb7 on PYTHONPATH.
#   (c) anything else: the probe output says what.
#   D  both probes run directly on the merge, output kept           (~2 min)
#   B  the two tests on plain main with the new pin; if they fail, again with the old pin (worktrees under scratch/b21)  (~2-4 min)
#   F  only if D shows (a): fix (scale_coupled.build defaults deterministic to 0), test_mpm_main in full (3.8 min,
#      T_mpm_main), commit fix + the tests' probe-output reporting, push, #29 mergeable state
# ~8-10 min. Resumable (scratch/b21/STATUS.txt).
set -u
W=.
R=.
S=$W/scratch/b21
L=$S/logs
O=$W/runs/mpm_main/batch_21
PY=$R/.venv/bin/python
MJW=$W/upstream/mujoco_warp-detccd
OLDM=4d05cb7d4453d82c94d19ab9fb392df2c62b1acd
MAINWT=$S/main_wt
OLDMJW=$S/mjw_4d05cb7
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_21 start $(date)"
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
gpu() {   # gpu NAME KIND MIN LOG CWD PYTHONPATH -- cmd...
  local name=$1 kind=$2 min=$3 log=$4 cwd=$5 pyp=$6; shift 7
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$cwd" -- env PYTHONPATH="$pyp" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -2 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "not on batchsim-deterministic"; exit 1; }
done_ F21 || [ "$(git -C "$W" rev-parse --short=8 HEAD)" = 9c090532 ] || { say "HEAD is not the batch_20 merge 9c090532: re-plan"; exit 1; }
say "shared MuJoCo Warp checkout: $(git -C "$R/upstream/mujoco_warp" rev-parse --short=7 HEAD)"
health

# ---------------------------------------------------------------- D: the probes on the merge, output kept
if ! done_ D21; then
  git -C "$W" diff --quiet -- metalsim/ || { mark D21 FAIL "metalsim/ modified: the merge is not what is tested"; exit 1; }
  gpu mm21_d_owr check 10 "$L/d_one_world_reset.log" "$W" "$W" -- "$PY" -P "$W/metalsim/physics/mpm/probes/one_world_reset.py"
  r1=$?
  gpu mm21_d_psab check 10 "$L/d_policy_step_ab.log" "$W" "$W" -- "$PY" -P "$W/metalsim/physics/mpm/probes/policy_step_ab.py" \
      --worlds 4 --fraction 0.25 --hash-steps 40 --steps 20 --warmup 2 --rounds 1
  r2=$?
  n=$(grep -l "has no attribute 'deterministic'" "$L/d_one_world_reset.log" "$L/d_policy_step_ab.log" | wc -l | tr -d ' ')
  if [ "$n" = 2 ]; then cause=a_namespace; else cause=other; fi
  echo "$cause" > "$S/cause.txt"
  for f in d_one_world_reset d_policy_step_ab; do echo "--- $f (last 25 lines)"; tail -25 "$L/$f.log"; done
  mark D21 DONE "rc $r1 $r2; cause $cause"
fi
CAUSE=$(cat "$S/cause.txt")

# ---------------------------------------------------------------- B: plain main with the new pin (and the old pin if it fails)
if ! done_ B21; then
  [ -d "$MAINWT" ] || git -C "$W" worktree add -q --detach "$MAINWT" origin/main || { mark B21 FAIL "main worktree"; exit 1; }
  say "main worktree at $(git -C "$MAINWT" rev-parse --short=8 HEAD)"
  K="test_policy_step_graph_matches_eager or test_one_world_masked_reset"
  gpu mm21_main_new check 10 "$L/main_newpin.log" "$MAINWT" "$MAINWT" -- "$PY" -P -m pytest -q -rfEs \
      "$MAINWT/tests/test_mpm_main.py" -k "$K"
  rn=$?
  ro=skipped
  if [ $rn != 0 ]; then
    say "!!! plain main fails these tests with the shared .venv's MuJoCo Warp: its probe's output, then the old pin"
    gpu mm21_main_owr check 10 "$L/main_newpin_one_world_reset.log" "$MAINWT" "$MAINWT" -- "$PY" -P \
        "$MAINWT/metalsim/physics/mpm/probes/one_world_reset.py"
    tail -25 "$L/main_newpin_one_world_reset.log"
    [ -d "$OLDMJW" ] || git -C "$MJW" worktree add -q --detach "$OLDMJW" "$OLDM" || { mark B21 FAIL "old-pin worktree"; exit 1; }
    gpu mm21_main_old check 10 "$L/main_oldpin.log" "$MAINWT" "$OLDMJW:$MAINWT" -- "$PY" -P -m pytest -q -rfEs \
        "$MAINWT/tests/test_mpm_main.py" -k "$K"
    ro=$?
  fi
  mark B21 DONE "main df9485a8: new pin rc $rn, old pin rc $ro$( [ $rn != 0 ] && [ "$ro" = 0 ] && echo ' -> MAIN BROKEN BY THE PIN MOVE (b)')"
fi
if grep -q "MAIN BROKEN BY THE PIN MOVE" "$STATUS"; then
  say "!!! (b): main's .venv is broken by MuJoCo Warp 8eb51fe for these tests; tell the coordinator now. Not fixing here."
fi

# ---------------------------------------------------------------- F: fix (a), full test_mpm_main, commit, push
if [ "$CAUSE" = a_namespace ] && ! done_ F21; then
  python3 "$S/fix_scale_coupled.py" "$W" || { mark F21 FAIL "fix"; exit 1; }
  gpu mm21_mpm_main check 25 "$L/test_mpm_main.log" "$W" "$W" -- "$PY" -P -m pytest -q -rfEs "$W/tests/test_mpm_main.py"
  rc=$?
  if [ $rc != 0 ]; then mark F21 FAIL "test_mpm_main rc=$rc $(tail -1 "$L/test_mpm_main.log")"; exit 1; fi
  mkdir -p "$O"; cp -R "$L" "$O/"; cp "$S/batch_21.sh" "$S/fix_scale_coupled.py" "$STATUS" "$O/" 2>/dev/null
  git -C "$W" add metalsim/physics/mpm/probes/scale_coupled.py tests/test_mpm_main.py runs/mpm_main/batch_21
  git -C "$W" commit -q -s -F - <<'EOF'
Merge fix: main's coupled probes build scale_coupled's Namespace without `deterministic`

scale_coupled.build() reads a.deterministic (this branch's --deterministic), but main's
one_world_reset.py and policy_step_ab.py (#33) build an argparse.Namespace without it, so
both probes raised AttributeError right after Warp initialised and test_mpm_main's
test_policy_step_graph_matches_eager / test_one_world_masked_reset failed on a JSONDecodeError.
build() now defaults it to 0. The tests that parse a probe's last stdout line now fail with
the probe's stdout / stderr tails instead.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  mark F21 PASS "$(git -C "$W" rev-parse --short HEAD) $(tail -1 "$L/test_mpm_main.log")"
fi
if done_ F21 && ! done_ PUSH21; then
  git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
  [ "$(git -C "$W" ls-remote origin refs/heads/batchsim-deterministic | cut -c1-40)" = "$(git -C "$W" rev-parse HEAD)" ] \
      && mark PUSH21 PASS "$(git -C "$W" rev-parse --short HEAD)" || { mark PUSH21 FAIL "push"; exit 1; }
  for i in 1 2 3 4 5 6; do
    st=$(cd "$W" && gh pr view 29 --json mergeable,mergeStateStatus,isDraft --jq '"\(.mergeable) \(.mergeStateStatus) draft=\(.isDraft)"' 2>&1)
    say "#29: $st"
    case "$st" in UNKNOWN*) sleep 20 ;; *) break ;; esac
  done
  mark PR29STATE DONE "$st"
fi
[ "$CAUSE" = a_namespace ] || say "cause is not the Namespace: nothing fixed or pushed; read $L/d_*.log"
say "=== batch_21 end $(date)"; cat "$STATUS"
