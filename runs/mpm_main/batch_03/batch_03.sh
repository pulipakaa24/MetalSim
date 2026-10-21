#!/bin/bash
# MPM in the main simulator, batch 03 (run outside the agent sandbox):  bash <this file>
#
#   0  commit the agent's changes (robust BatchSim determinism, lagged default, bearing / repose probes, docs), merge main
#   1  import-path audit: which metalsim each job imports (gpu_run starts jobs in the main checkout; this batch's jobs
#      run with --cwd = the worktree through scripts/watchdog.py, and every python -m job uses -P as well)
#   2  determinism: the stacked-boxes case that failed in batch_02 (in-process, cross-process cold / cold / warm cache),
#      a default-mode negative control, then tests/test_batch_deterministic.py
#   3  bearing matrix (research_notes/MPM in main simulator/bearing_and_sinking.md): drop and rest, lagged vs
#      staggered, dt 1/240, iterations / tolerance, collider basis, voxel, friction 0.3 / 1.2
#   4  angle of repose, corrected protocol (angle_of_repose.md): resolution ladder R/dx 5 / 10 / 15 at mu 0.68, lift
#      speed plateau, mu 0.4 / 0.9, iteration check
#   5  regression: tests/test_mpm_main.py, tests/test_physics.py, tests/test_mpm.py (.venv-newton152), neutrality
#   6  if every gate passes: push the Newton fork branch (+ fast-forward metalsim-1.5.2), commit + push, open / update PR
# Bearing and repose are measurements (documented in the PR whatever they show), not gates.
# Every GPU job: scripts/gpu_run.sh + scripts/watchdog.py, health check after each. Resumable. Expected ~2-3 h.
set -u
W=.
R=.
B=$W/runs/mpm_main/batch_03
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
mkdir -p "$L" "$D/scale" "$D/neutral" "$D/repose3" "$D/bearing3" "$D/det3"
STATUS=$B/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_03 start $(date)"

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

gpu() {   # gpu NAME KIND MINUTES LOGFILE -- command...   (the command runs with cwd = the worktree)
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
  commit "MPM in the main simulator: batch 03 runs and logs" "Determinism diagnosis and tests, bearing matrix (lagged vs staggered), repose by the corrected cylinder-lift protocol, regression tests (batch_03)."
  git -C "$W" push -q origin mpm-main-sim 2>&1 | grep -v "^remote:" | tail -3
  say "=== batch_03 end $(date) rc=$rc"; cat "$STATUS"
}
trap finish EXIT

# ---------------------------------------------------------------- 0 commit, merge main
if ! done_ SAVE; then
  commit "BatchSim determinism made robust; lagged exchange by default; physics probes" \
"BatchSimOptions(deterministic=True) now fixes the mode before MuJoCo
Warp's first launch (put_model / put_data inside the deterministic
scope), allows one mode per process (a mix raises), uses the computed
record bound while generated kernels are created, checks that every
replayed MuJoCo Warp module was built deterministic, and reports Warp's
deterministic overflow flag in check_overflow.

MPMCoupling defaults to Newton's lagged exchange with the velocity
rewind: the staggered exchange (Isaac's Franka-Pour setting) lets a
resting body sink at g dt per step (batch 02: 0.082 m/s at 1/120 s in
every bearing configuration). Bearing and cylinder-lift repose probes
rewritten from the research notes; batch 02's lift run was void."
  git -C "$W" fetch -q origin
  if git -C "$W" merge -q --no-edit origin/main; then mark SAVE PASS "$(git -C "$W" rev-parse --short HEAD)"
  else git -C "$W" merge --abort; mark SAVE PASS "merge with origin/main conflicted: left unmerged"; fi
fi
[ "$(git -C "$FORK" rev-parse HEAD)" = "$NEW" ] || { mark FORK FAIL "fork clone not at $NEW"; exit 1; }

# ---------------------------------------------------------------- 1 import-path audit
if ! done_ IMPORTS; then
  for v in "$PY" "$PY152"; do
    ( cd "$R" && "$v" -m pytest -q -p no:cacheprovider --co "$W/tests/test_mpm_main.py" > /dev/null 2>&1; \
      "$v" -c "import metalsim; print('main-checkout cwd, -c :', metalsim.__file__)"; \
      "$v" -P -c "import metalsim; print('main-checkout cwd, -P -c:', metalsim.__file__)" )
    ( cd "$W" && "$v" -c "import metalsim; print('worktree cwd, -c:', metalsim.__file__)" )
  done > "$L/import_audit.log" 2>&1
  cat "$L/import_audit.log"
  grep "worktree cwd" "$L/import_audit.log" | grep -qv "$W" && mark IMPORTS FAIL || mark IMPORTS PASS "batch jobs run with cwd = worktree (watchdog --cwd)"
fi
health

# ---------------------------------------------------------------- 2 determinism
DG="$P/rigid_det_diag.py"
if ! done_ DET_DIAG; then
  gpu mm3_det_stack256 check 20 "$L/det_stack256_inproc.log" -- "$PY" "$DG" --scene stack --worlds 256 --json "$D/det3/stack256_a.json"
  gpu mm3_det_stack256_c1 check 20 "$L/det_stack256_cold1.log" -- env WARP_CACHE_PATH="$W/scratch/mpm_main/cache_cold1" "$PY" "$DG" \
      --scene stack --worlds 256 --runs 1 --json "$D/det3/stack256_cold1.json"
  gpu mm3_det_stack256_c2 check 20 "$L/det_stack256_cold2.log" -- env WARP_CACHE_PATH="$W/scratch/mpm_main/cache_cold2" "$PY" "$DG" \
      --scene stack --worlds 256 --runs 1 --json "$D/det3/stack256_cold2.json"
  gpu mm3_det_stack256_w check 20 "$L/det_stack256_warm.log" -- env WARP_CACHE_PATH="$W/scratch/mpm_main/cache_cold2" "$PY" "$DG" \
      --scene stack --worlds 256 --runs 1 --json "$D/det3/stack256_warm.json"
  gpu mm3_det_neg check 20 "$L/det_box1024_default.log" -- "$PY" "$DG" --scene box --worlds 1024 --deterministic 0 \
      --json "$D/det3/box1024_default.json"
  {
    "$PY" -P "$DG" --compare "$D/det3/stack256_cold1.json" "$D/det3/stack256_cold2.json"
    "$PY" -P "$DG" --compare "$D/det3/stack256_cold2.json" "$D/det3/stack256_warm.json"
    "$PY" -P "$DG" --compare "$D/det3/stack256_a.json" "$D/det3/stack256_warm.json"
  } > "$L/det_compare.log" 2>&1
  cat "$L/det_compare.log"
  mark DET_DIAG DONE "in-process: $(jget "$D/det3/stack256_a.json" "r['in_process_first_diff']") cross: $(tr '\n' ' ' < "$L/det_compare.log") default-mode control: $(jget "$D/det3/box1024_default.json" "r['in_process_first_diff']")"
fi
if ! done_ TESTS_DET; then
  gpu mm3_tests_det check 60 "$L/tests_batch_deterministic.log" -- "$PY" -P -m pytest -v -p no:cacheprovider \
      --rootdir "$W" "$W/tests/test_batch_deterministic.py" && mark TESTS_DET PASS || mark TESTS_DET FAIL
fi

# ---------------------------------------------------------------- 3 bearing matrix
BR="$P/bearing.py"
if ! done_ BEARING3; then
  i=0
  while read -r name args; do
    [ -z "$name" ] && continue
    gpu "mm3_bear_$name" check 30 "$L/bearing_$name.log" -- "$PY" "$BR" $args --json "$D/bearing3/$name.json"
  done <<'EOF'
drop_staggered      --mode staggered
drop_lagged         --mode lagged
drop_staggered_240  --mode staggered --fps 240
drop_lagged_240     --mode lagged --fps 240
rest_lagged         --mode lagged --rest
rest_staggered      --mode staggered --rest
rest_lagged_i250    --mode lagged --rest --iterations 250 --tolerance 1e-6
rest_lagged_q1      --mode lagged --rest --basis Q1
rest_lagged_pic27   --mode lagged --rest --basis pic27
rest_lagged_v01     --mode lagged --rest --voxel 0.01
rest_lagged_mu03    --mode lagged --rest --friction 0.3
rest_lagged_mu12    --mode lagged --rest --friction 1.2
EOF
  mark BEARING3 DONE "$(for f in "$D"/bearing3/*.json; do echo -n "$(basename "$f" .json): pen $(jget "$f" "round(r['peak_penetration_m']*1000,1)") / final $(jget "$f" "round(r['final_penetration_m']*1000,1)") mm, creep $(jget "$f" "round(r['creep_m_per_s']*1000,1)") mm/s, F/W $(jget "$f" "round(r['force_over_weight_tail'],2)"); "; done)"
fi

# ---------------------------------------------------------------- 4 angle of repose
RL="$P/repose_lift.py"
if ! done_ REPOSE3; then
  while read -r name args; do
    [ -z "$name" ] && continue
    gpu "mm3_lift_$name" check 45 "$L/lift_$name.log" -- "$PY" "$RL" $args --json "$D/repose3/$name.json"
  done <<'EOF'
mu068_dx010_v025   --friction 0.68 --voxel 0.01 --lift-speed 0.025
mu068_dx005_v025   --friction 0.68 --voxel 0.005 --lift-speed 0.025
mu068_dx0033_v025  --friction 0.68 --voxel 0.0033 --lift-speed 0.025
mu068_dx005_v0125  --friction 0.68 --voxel 0.005 --lift-speed 0.0125
mu068_dx005_v050   --friction 0.68 --voxel 0.005 --lift-speed 0.05
mu068_dx005_i1000  --friction 0.68 --voxel 0.005 --lift-speed 0.025 --iterations 1000
mu040_dx005_v025   --friction 0.4 --voxel 0.005 --lift-speed 0.025
mu090_dx005_v025   --friction 0.9 --voxel 0.005 --lift-speed 0.025
EOF
  mark REPOSE3 DONE "$(for f in "$D"/repose3/*.json; do echo -n "$(basename "$f" .json): $(jget "$f" "round(r['final']['angle_fit_mean_deg'],1), round(r['final']['angle_fit_sd_deg'],1), round(r['atan_mu_deg'],1), r['at_rest'], round(r['angle_drift_last_0.5s_deg'],2)"); "; done)"
fi

# ---------------------------------------------------------------- 5 regression
if ! done_ TESTS_MAIN; then
  gpu mm3_tests_main check 90 "$L/tests_mpm_main.log" -- env METALSIM_NEWTON152_DIR="$FORK" "$PY" -P -m pytest -v \
      -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_main.py" && mark TESTS_MAIN PASS || mark TESTS_MAIN FAIL
fi
if ! done_ TESTS_PHYSICS; then
  gpu mm3_tests_phys check 40 "$L/tests_physics.log" -- "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_physics.py" && mark TESTS_PHYSICS PASS || mark TESTS_PHYSICS FAIL
fi
if ! done_ TESTS_MPM152; then
  gpu mm3_tests_152 check 30 "$L/tests_mpm_newton152.log" -- "$PY152" -P -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_mpm.py" && mark TESTS_MPM152 PASS || mark TESTS_MPM152 FAIL
fi
if ! done_ NEUTRAL3; then
  gpu mm3_neutral check 15 "$L/neutral_plain_b03.log" -- "$PY" "$P/rigid_neutral_mpm.py" --mode plain \
      --json "$D/neutral/plain_b03.json"
  "$PY" -P "$P/rigid_neutral_mpm.py" --compare "$D/neutral/plain1.json" "$D/neutral/plain_b03.json" \
      > "$L/neutral_b03_compare.log" 2>&1 && grep -q "'box': True" "$L/neutral_b03_compare.log" \
      && mark NEUTRAL3 PASS "$(cat "$L/neutral_b03_compare.log")" || mark NEUTRAL3 FAIL "$(cat "$L/neutral_b03_compare.log")"
fi

# ---------------------------------------------------------------- 6 push fork, PR (only if every gate passes)
GREEN=1
for s in SAVE IMPORTS TESTS_DET TESTS_MAIN TESTS_PHYSICS TESTS_MPM152 NEUTRAL3; do done_ $s || GREEN=0; done
for s in COLOR_NEW MWREPEAT COLUMN_BITWISE BOX_PAIR; do grep -q "^$s PASS" "$D/batch_01/STATUS.txt" || GREEN=0; done
for s in SCALE_DET COLLAPSE_PARITY; do grep -q "^$s PASS" "$D/batch_02/STATUS.txt" || GREEN=0; done
if [ $GREEN = 1 ] && ! done_ FORK_PUSH; then
  git -C "$FORK" push -q "$GH_NEWTON" metalsim-1.5.2-gs-envfirst 2>&1 | tail -2
  git -C "$FORK" ls-remote "$GH_NEWTON" refs/heads/metalsim-1.5.2-gs-envfirst | grep -q "$NEW" \
    && mark FORK_PUSH PASS || mark FORK_PUSH FAIL
  git -C "$FORK" push -q "$GH_NEWTON" "$NEW:refs/heads/metalsim-1.5.2" 2>&1 | tail -2
  git -C "$FORK" ls-remote "$GH_NEWTON" refs/heads/metalsim-1.5.2 | grep -q "$NEW" \
    && mark FORK_FF PASS || mark FORK_FF FAIL "metalsim-1.5.2 not fast-forwarded (moved since 3fd1f052?): the pin stays on the topic branch"
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
