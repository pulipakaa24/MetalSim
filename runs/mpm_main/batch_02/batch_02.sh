#!/bin/bash
# MPM in the main simulator, batch 02 (run outside the agent sandbox):  bash <this file>
#
# Follows batch_01 (SCALE and TESTS_MAIN failed; REPOSE was recorded, not judged). This batch:
#   0  commit the agent's changes (BatchSim deterministic option, fixed tests, new probes, docs)
#   1  SCALE diagnosis: which part (rigid / wrench / particles) diverges first at 1,024 worlds with Jacobi; the rigid
#      scene alone at 1,024 worlds; then BatchSimOptions(deterministic=True), tiny first (4 worlds), up to 1,024
#   2  rigid regression: test_batch_deterministic.py, test_physics.py, rigid neutrality vs batch_01 (default path
#      unchanged: same module hashes and box trajectory)
#   3  physics: column-collapse parity MetalSim vs Newton alone (reference for the test), slow-lift angle of repose
#      (friction 0.4 / 0.68 / 0.9 at voxel 0.01, 0.68 at 0.0067), bed bearing settlement matrix
#   4  tests/test_mpm_main.py
#   5  only if every gate passed: push the Newton fork branch (+ fast-forward metalsim-1.5.2), commit + push, open the PR
# Every GPU job: scripts/gpu_run.sh + scripts/watchdog.py, health check after each; stops on a GPU anomaly or a
# watchdog kill. Resumable: steps marked PASS in STATUS.txt are skipped. Expected ~1.5-2.5 h.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
B=$W/runs/mpm_main/batch_02
L=$B/logs
D=$W/runs/mpm_main
FORK=$W/upstream/newton-metalsim-1.5.2-envfirst
NEW=99843d09840f89ac16c82104ad66ae792e45eb91
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
P=$W/metalsim/physics/mpm/probes
GH_NEWTON=https://github.com/pulipakaa24/newton.git
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$D/scale" "$D/neutral" "$D/repose" "$D/bearing"
STATUS=$B/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_02 start $(date)"

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

gpu() {   # gpu NAME KIND MINUTES LOGFILE -- command...
  local name=$1 kind=$2 min=$3 log=$4; shift 5
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --timeout $((min * 60)) --cwd "$W" -- "$@" > "$log" 2>&1
  local rc=$?
  tail -4 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}

jget() { "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'r': r}))" "$1" "$2"; }

stage() {
  git -C "$W" add metalsim tests/test_mpm_main.py tests/test_batch_deterministic.py scripts/vendor_mpm152.py \
      scripts/setup_newton152.sh patches/newton THIRD_PARTY_NOTICES.md README.md pyproject.toml docs/GUIDE.md \
      docs/STATUS.md docs/GAPS.md "research_notes/MPM in main simulator" runs/mpm_main ':!runs/mpm_main/fork' 2>&1 | tail -3
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
  commit "MPM in the main simulator: batch 02 runs and logs" "Scale / determinism diagnosis, rigid regression, repose and bearing measurements, test run (batch_02)."
  git -C "$W" push -q origin mpm-main-sim 2>&1 | grep -v "^remote:" | tail -3
  say "=== batch_02 end $(date) rc=$rc"; cat "$STATUS"
}
trap finish EXIT

# ---------------------------------------------------------------- 0 commit the agent's changes
if ! done_ SAVE; then
  commit "BatchSim deterministic option; MPM tests and probes fixed" \
"BatchSimOptions(deterministic=True) runs MuJoCo Warp's kernels in Warp's
RUN_TO_RUN mode (off by default: kernels and results of existing scenes
unchanged). Free-fall test: exact momentum with the solver's numerical
drag off; repose test replaced by a collapse parity test against Newton
alone plus a friction-response test; probes for the slow-lift angle of
repose and the bed bearing settlement; per-part divergence in the scale
probe. Batch 01 numbers in MPM.md section 11."
  mark SAVE PASS "$(git -C "$W" rev-parse --short HEAD)"
fi
[ "$(git -C "$FORK" rev-parse HEAD)" = "$NEW" ] || { mark FORK FAIL "fork clone not at $NEW"; exit 1; }
health

# ---------------------------------------------------------------- 1 scale / determinism diagnosis
SC="$P/scale_coupled.py"
if ! done_ DIAG; then
  gpu mm2_diag_j1024 check 45 "$L/diag_jacobi_w1024_det0.log" -- "$PY" "$SC" --worlds 1024 --steps 40 --runs 2 \
      --solver jacobi --json "$D/scale/diag_jacobi_w1024_det0.json"
  gpu mm2_diag_rigid check 20 "$L/diag_rigid_w1024_det0.log" -- "$PY" "$SC" --worlds 1024 --steps 120 --runs 3 \
      --rigid-only --json "$D/scale/diag_rigid_w1024_det0.json"
  mark DIAG PASS "jacobi parts: $(jget "$D/scale/diag_jacobi_w1024_det0.json" "r.get('first_differing_by_part')") rigid-only: $(jget "$D/scale/diag_rigid_w1024_det0.json" "r['first_differing_step']")"
fi
if ! done_ SCALE_DET; then
  ok=1
  # tiny first: the deterministic MuJoCo Warp kernels are new code on this tree
  for w in 4 64 1024; do
    gpu "mm2_rigid_det_$w" check 25 "$L/rigid_det_w$w.log" -- "$PY" "$SC" --worlds $w --steps 120 --runs 3 \
        --rigid-only --deterministic 1 --json "$D/scale/rigid_det_w$w.json" || { ok=0; break; }
  done
  if [ $ok = 1 ]; then
    for cfg in "jacobi 64" "jacobi 1024" "auto 1024"; do
      set -- $cfg
      gpu "mm2_det_$1_$2" check 45 "$L/scale_det_$1_w$2.log" -- "$PY" "$SC" --worlds $2 --steps 40 --runs 2 \
          --solver $1 --deterministic 1 --json "$D/scale/repeat_det_$1_w$2.json" || ok=0
    done
  fi
  [ $ok = 1 ] && mark SCALE_DET PASS || mark SCALE_DET FAIL
fi

# ---------------------------------------------------------------- 2 rigid regression
if ! done_ TESTS_DET; then
  gpu mm2_tests_det check 30 "$L/tests_batch_deterministic.log" -- "$PY" -m pytest -v -p no:cacheprovider \
      --rootdir "$W" "$W/tests/test_batch_deterministic.py" && mark TESTS_DET PASS || mark TESTS_DET FAIL
fi
if ! done_ TESTS_PHYSICS; then
  gpu mm2_tests_phys check 40 "$L/tests_physics.log" -- "$PY" -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_physics.py" && mark TESTS_PHYSICS PASS || mark TESTS_PHYSICS FAIL
fi
if ! done_ NEUTRAL2; then
  gpu mm2_neutral check 15 "$L/neutral_plain_b02.log" -- "$PY" "$P/rigid_neutral_mpm.py" --mode plain \
      --json "$D/neutral/plain_b02.json"
  "$PY" "$P/rigid_neutral_mpm.py" --compare "$D/neutral/plain1.json" "$D/neutral/plain_b02.json" \
      > "$L/neutral_b02_compare.log" 2>&1 && grep -q "'box': True" "$L/neutral_b02_compare.log" \
      && mark NEUTRAL2 PASS "$(cat "$L/neutral_b02_compare.log")" || mark NEUTRAL2 FAIL "$(cat "$L/neutral_b02_compare.log")"
fi

# ---------------------------------------------------------------- 3 physics
if ! done_ COLLAPSE_PARITY; then
  gpu mm2_col_nt check 20 "$L/collapse_newton.log" -- "$PY152" "$P/pile_repose.py" --engine newton --friction 0.68 \
      --seconds 1.0 --json "$D/repose/collapse_newton_mu0.68.json"
  gpu mm2_col_ms check 20 "$L/collapse_metalsim.log" -- "$PY" "$P/pile_repose.py" --engine metalsim --friction 0.68 \
      --seconds 1.0 --json "$D/repose/collapse_metalsim_mu0.68.json"
  if [ "$(jget "$D/repose/collapse_newton_mu0.68.json" "r['hashes']")" = "$(jget "$D/repose/collapse_metalsim_mu0.68.json" "r['hashes']")" ]; then
    mark COLLAPSE_PARITY PASS "bitwise"
  else mark COLLAPSE_PARITY FAIL "deposits differ: newton $(jget "$D/repose/collapse_newton_mu0.68.json" "r['deposit']") metalsim $(jget "$D/repose/collapse_metalsim_mu0.68.json" "r['deposit']")"; fi
fi
if ! done_ LIFT; then
  for cfg in "0.68 0.01" "0.4 0.01" "0.9 0.01" "0.68 0.0067"; do
    set -- $cfg
    gpu "mm2_lift_$1_$2" check 40 "$L/lift_mu$1_v$2.log" -- "$PY" "$P/repose_lift.py" --friction $1 --voxel $2 \
        --json "$D/repose/lift_mu$1_v$2.json"
  done
  mark LIFT DONE "$(for f in "$D"/repose/lift_*.json; do echo -n "$(basename "$f"): $(jget "$f" "round(r['heap']['angle_deg'],1), round(r['friction_angle_deg'],1), r['at_rest']") ; "; done)"
fi
if ! done_ BEARING; then
  for cfg in "0.02 50 0.08 0.7" "0.02 250 0.08 0.7" "0.01 50 0.08 0.7" "0.01 250 0.08 0.7" "0.02 50 0.16 0.7" \
             "0.02 50 0.08 0.4" "0.02 50 0.08 0.9"; do
    set -- $cfg
    gpu "mm2_bear_$1_$2_$3_$4" check 30 "$L/bearing_v$1_i$2_d$3_mu$4.log" -- "$PY" "$P/bearing.py" --voxel $1 \
        --iterations $2 --depth $3 --friction $4 --json "$D/bearing/bearing_v$1_i$2_d$3_mu$4.json"
  done
  mark BEARING DONE "$(for f in "$D"/bearing/*.json; do echo -n "$(basename "$f" .json): $(jget "$f" "round(r['settlement_m']*1000,1)") mm; "; done)"
fi

# ---------------------------------------------------------------- 4 tests
if ! done_ TESTS_MAIN; then
  gpu mm2_tests_main check 75 "$L/tests_mpm_main.log" -- env METALSIM_NEWTON152_DIR="$FORK" "$PY" -m pytest -v \
      -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_main.py" && mark TESTS_MAIN PASS || mark TESTS_MAIN FAIL
fi

# ---------------------------------------------------------------- 5 push fork, PR (only if everything is green)
GREEN=1
for s in SAVE SCALE_DET TESTS_DET TESTS_PHYSICS NEUTRAL2 COLLAPSE_PARITY TESTS_MAIN; do done_ $s || GREEN=0; done
for s in COLOR_NEW MWREPEAT COLUMN_BITWISE BOX_PAIR TESTS_MPM152; do grep -q "^$s PASS" "$D/batch_01/STATUS.txt" || GREEN=0; done
if [ $GREEN = 1 ] && ! done_ FORK_PUSH; then
  git -C "$FORK" push -q "$GH_NEWTON" metalsim-1.5.2-gs-envfirst 2>&1 | tail -2
  git -C "$FORK" ls-remote "$GH_NEWTON" refs/heads/metalsim-1.5.2-gs-envfirst | grep -q "$NEW" \
    && mark FORK_PUSH PASS || mark FORK_PUSH FAIL
  git -C "$FORK" push -q "$GH_NEWTON" "$NEW:refs/heads/metalsim-1.5.2" 2>&1 | tail -2
  git -C "$FORK" ls-remote "$GH_NEWTON" refs/heads/metalsim-1.5.2 | grep -q "$NEW" \
    && mark FORK_FF PASS || mark FORK_FF FAIL "metalsim-1.5.2 not fast-forwarded (moved since 3fd1f052?): pin stays on the topic branch"
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
