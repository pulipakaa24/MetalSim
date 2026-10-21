#!/bin/bash
# MPM in the main simulator, batch 04 (run outside the agent sandbox):  bash <this file>
#
#   0  commit the resolved merge of origin/main (staged by the agent: README, THIRD_PARTY_NOTICES, STATUS, MPM.md);
#      branch `batchsim-deterministic` at 140b6fd9 (the deterministic BatchSim work), push it, open it as a DRAFT PR
#      based on mpm-main-sim; commit the split on mpm-main-sim (BatchSim back to main's, deterministic tests and probe
#      removed, docs and GAPS / STATUS updated with batch_03's bearing and repose results)
#   1  GPU: tests/test_mpm_main.py (new resting-box test, repeat test without the deterministic option),
#      tests/test_mpm.py (.venv-newton152, after the merge), bearing with Isaac's default solve (250 iterations, 1e-4),
#      repose at R/dx 20
#   2  if every gate passes: push the Newton fork branch (+ fast-forward metalsim-1.5.2), push, open the MPM PR
# Resumable (STATUS.txt). Expected ~1-1.5 h.
set -u
W=.
R=.
B=$W/runs/mpm_main/batch_04
L=$B/logs
D=$W/runs/mpm_main
FORK=$W/upstream/newton-metalsim-1.5.2-envfirst
NEW=99843d09840f89ac16c82104ad66ae792e45eb91
DETBASE=140b6fd9
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
P=$W/metalsim/physics/mpm/probes
GH_NEWTON=https://github.com/pulipakaa24/newton.git
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$D/bearing4" "$D/repose4"
STATUS=$B/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_04 start $(date)"

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

gpu() {   # gpu NAME KIND MINUTES LOGFILE -- command...   (cwd = the worktree)
  local name=$1 kind=$2 min=$3 log=$4; shift 5
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --timeout $((min * 60)) --cwd "$W" -- "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -4 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}

jget() { "$PY" -P -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'r': r}))" "$1" "$2"; }

stage() {
  git -C "$W" add -A metalsim tests scripts/vendor_mpm152.py scripts/setup_newton152.sh patches/newton \
      THIRD_PARTY_NOTICES.md README.md pyproject.toml docs/GUIDE.md docs/STATUS.md docs/GAPS.md \
      "research_notes/MPM in main simulator" runs/mpm_main ':!runs/mpm_main/fork' 2>&1 | tail -3
}
commit() {
  stage
  git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
$1

$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}
finish() {
  local rc=$?
  say "finish (rc=$rc): commit + push"
  if done_ SPLIT; then
    commit "MPM in the main simulator: batch 04 runs and logs" "Tests after the merge and the split, bearing with Isaac's default solve, repose at R/dx 20 (batch_04)."
    git -C "$W" push -q origin mpm-main-sim 2>&1 | grep -v "^remote:" | tail -3
  fi
  say "=== batch_04 end $(date) rc=$rc"; cat "$STATUS"
}
trap finish EXIT

# ---------------------------------------------------------------- 0 merge commit, deterministic branch, split
if ! done_ MERGE; then
  if [ -f "$(git -C "$W" rev-parse --git-path MERGE_HEAD)" ]; then
    if [ -n "$(git -C "$W" diff --name-only --diff-filter=U)" ]; then mark MERGE FAIL "unresolved conflicts"; exit 1; fi
    # the index holds only the merge resolution; the agent's later edits are unstaged and go into the next commit
    git -C "$W" commit -q -s -F - <<EOF || { mark MERGE FAIL "commit failed"; exit 1; }
Merge origin/main into mpm-main-sim

Conflicts resolved in README.md, THIRD_PARTY_NOTICES.md, docs/STATUS.md
and metalsim/physics/mpm/MPM.md: main's text kept, this branch's
additions (vendored solver notice, MPMCoupling, Newton pin 99843d09)
added.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  mark MERGE PASS "$(git -C "$W" rev-parse --short HEAD)"
fi
if ! done_ DETBRANCH; then
  git -C "$W" rev-parse -q --verify refs/heads/batchsim-deterministic > /dev/null || git -C "$W" branch batchsim-deterministic "$DETBASE"
  git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
  if git -C "$W" ls-remote origin refs/heads/batchsim-deterministic | grep -q .; then
    ( cd "$W" && { gh pr view batchsim-deterministic > /dev/null 2>&1 || gh pr create --draft --base mpm-main-sim \
        --head batchsim-deterministic --title "Deterministic BatchSim (draft: stacked-boxes divergence open)" \
        --body-file "$B/pr_body_det.md"; } ) > "$L/pr_det.log" 2>&1
    cat "$L/pr_det.log"
    mark DETBRANCH PASS "$(tail -1 "$L/pr_det.log")"
  else mark DETBRANCH FAIL "push failed"; fi
fi
if ! done_ SPLIT; then
  commit "Take the deterministic BatchSim out of the MPM branch; batch 03 results in the docs" \
"BatchSimOptions(deterministic=...) and its tests move to branch
batchsim-deterministic (draft PR) until the two-box divergence at 256
worlds is fixed; BatchSim here is main's again. The repeat test keeps
the scenes whose bodies touch only sand. MPM.md, STATUS and GAPS carry
batch 03's results: staggered exchange sinks resting bodies at g dt per
step, the lagged exchange carries them (10 mm with a converged solve),
the solver's repose angle is 7-9 deg below atan(mu). New test: a
resting box is carried by the sand with the lagged exchange."
  mark SPLIT PASS "$(git -C "$W" rev-parse --short HEAD)"
fi
git -C "$W" push -q origin mpm-main-sim 2>&1 | grep -v "^remote:" | tail -2
[ "$(git -C "$FORK" rev-parse HEAD)" = "$NEW" ] || { mark FORK FAIL "fork clone not at $NEW"; exit 1; }
health

# ---------------------------------------------------------------- 1 GPU
if ! done_ TESTS_MAIN; then
  gpu mm4_tests_main check 90 "$L/tests_mpm_main.log" -- env METALSIM_NEWTON152_DIR="$FORK" "$PY" -P -m pytest -v \
      -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_main.py" && mark TESTS_MAIN PASS || mark TESTS_MAIN FAIL
fi
if ! done_ TESTS_MPM152; then
  gpu mm4_tests_152 check 30 "$L/tests_mpm_newton152.log" -- "$PY152" -P -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_mpm.py" && mark TESTS_MPM152 PASS || mark TESTS_MPM152 FAIL
fi
if ! done_ BEARING4; then
  while read -r name args; do
    [ -z "$name" ] && continue
    gpu "mm4_bear_$name" check 30 "$L/bearing_$name.log" -- "$PY" "$P/bearing.py" $args --json "$D/bearing4/$name.json"
  done <<'EOF'
rest_lagged_i250_t1e4   --mode lagged --rest --iterations 250 --tolerance 1e-4
drop_lagged_i250_t1e4   --mode lagged --iterations 250 --tolerance 1e-4
drop_lagged_i250_t1e6   --mode lagged --iterations 250 --tolerance 1e-6
EOF
  mark BEARING4 DONE "$(for f in "$D"/bearing4/*.json; do echo -n "$(basename "$f" .json): pen $(jget "$f" "round(r['peak_penetration_m']*1000,1)") / final $(jget "$f" "round(r['final_penetration_m']*1000,1)") mm, creep $(jget "$f" "round(r['creep_m_per_s']*1000,1)") mm/s, F/W $(jget "$f" "round(r['force_over_weight_tail'],2)"); "; done)"
fi
if ! done_ REPOSE4; then
  gpu mm4_lift_dx0025 check 60 "$L/lift_mu068_dx0025.log" -- "$PY" "$P/repose_lift.py" --friction 0.68 --voxel 0.0025 \
      --lift-speed 0.025 --json "$D/repose4/mu068_dx0025_v025.json"
  mark REPOSE4 DONE "$(jget "$D/repose4/mu068_dx0025_v025.json" "round(r['final']['angle_fit_mean_deg'],1), r['particles'], r['at_rest']")"
fi

# ---------------------------------------------------------------- 2 fork push, PR
GREEN=1
for s in MERGE SPLIT TESTS_MAIN TESTS_MPM152; do done_ $s || GREEN=0; done
for s in COLOR_NEW MWREPEAT COLUMN_BITWISE BOX_PAIR; do grep -q "^$s PASS" "$D/batch_01/STATUS.txt" || GREEN=0; done
grep -q "^COLLAPSE_PARITY PASS" "$D/batch_02/STATUS.txt" || GREEN=0
if [ $GREEN = 1 ] && ! done_ FORK_PUSH; then
  git -C "$FORK" push -q "$GH_NEWTON" metalsim-1.5.2-gs-envfirst 2>&1 | tail -2
  git -C "$FORK" ls-remote "$GH_NEWTON" refs/heads/metalsim-1.5.2-gs-envfirst | grep -q "$NEW" \
    && mark FORK_PUSH PASS || mark FORK_PUSH FAIL
  git -C "$FORK" push -q "$GH_NEWTON" "$NEW:refs/heads/metalsim-1.5.2" 2>&1 | tail -2
  git -C "$FORK" ls-remote "$GH_NEWTON" refs/heads/metalsim-1.5.2 | grep -q "$NEW" \
    && mark FORK_FF PASS || mark FORK_FF FAIL "metalsim-1.5.2 not fast-forwarded: the pin stays on the topic branch"
fi
trap - EXIT
finish
if [ $GREEN = 1 ] && done_ FORK_PUSH && ! done_ PR; then
  cd "$W" || exit 1
  if gh pr view mpm-main-sim > /dev/null 2>&1; then gh pr edit mpm-main-sim --body-file "$B/pr_body.md"
  else gh pr create --base main --head mpm-main-sim \
      --title "MPM material in the main simulator (BatchSim): Isaac's implicit MPM, proxy coupling, N worlds" \
      --body-file "$B/pr_body.md"; fi > "$L/pr.log" 2>&1
  cat "$L/pr.log"; gh pr view mpm-main-sim --json url -q .url > /dev/null 2>&1 && mark PR PASS "$(tail -1 "$L/pr.log")" || mark PR FAIL
else
  say "PR not opened: gates $( [ $GREEN = 1 ] && echo green || echo 'not green' ) (see STATUS.txt)"
fi
