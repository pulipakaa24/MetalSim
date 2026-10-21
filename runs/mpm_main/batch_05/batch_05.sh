#!/bin/bash
# batch 05 (run outside the agent sandbox):  bash scratch/b05/batch_05.sh
#
#   0  wait until PR #26 (mpm-main-sim) is merged into main (up to 3 h), fetch
#   1  branch mpm-followups from origin/main: MPM.md section 11 gets batch 04's numbers (bearing with Isaac's default
#      solve, repose at R/dx 20) + batch 04's logs; push; PR to main
#   2  branch batchsim-deterministic from origin/main (fast-forward of the old one): re-add the deterministic BatchSim
#      (batch.py, tests/test_batch_deterministic.py, probes, coupled.py scope) from 140b6fd9; push; DRAFT PR to main
#   3  on that branch, GPU diagnostics of the two-box divergence (research_notes/MPM in main simulator/
#      deterministic_ccd_divergence.md): native CCD off, Warp's single-threadgroup finalize off, small-record resolve
#      off, contact-slot audit (CCD on / off); commit the logs; push
# Resumable (scratch/b05/STATUS.txt). Expected ~30-60 min after #26 is merged.
set -u
W=.
R=.
S=$W/scratch/b05
L=$S/logs
PY=$R/.venv/bin/python
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_05 start $(date)"

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
gpu() {
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
commit_msg() {   # commit_msg SUBJECT BODY  (commits what is staged)
  git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
$1

$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}

# ---------------------------------------------------------------- 0 wait for #26
if ! done_ MERGED26; then
  for i in $(seq 1 180); do
    st=$(cd "$W" && gh pr view 26 --json state -q .state 2>/dev/null)
    [ "$st" = "MERGED" ] && break
    [ $i = 1 ] && say "waiting for PR #26 to be merged (state: $st)"
    sleep 60
  done
  [ "$st" = "MERGED" ] || { mark MERGED26 FAIL "PR #26 not merged after 3 h (state $st)"; exit 1; }
  mark MERGED26 PASS
fi
git -C "$W" fetch -q origin || exit 1

# ---------------------------------------------------------------- 1 docs follow-up branch
if ! done_ DOCS; then
  git -C "$W" checkout -q -B mpm-followups origin/main || { mark DOCS FAIL "checkout"; exit 1; }
  "$PY" -P "$S/edit_docs_b04.py" "$W" || { mark DOCS FAIL "edit"; exit 1; }
  git -C "$W" add metalsim/physics/mpm/MPM.md runs/mpm_main/batch_04
  commit_msg "MPM.md: bearing with Isaac's default solve, repose at R/dx 20" \
"Batch 04's numbers (16-inch): with the lagged exchange and 250
iterations at 1e-4 a resting box settles 10.2 mm and a 1.7 m/s drop
penetrates 29.8 mm; the repose angle at R/dx 20 is 27.8 deg for mu 0.68
(atan 34.2). Batch 04's logs."
  git -C "$W" push -q -u origin mpm-followups 2>&1 | grep -v "^remote:" | tail -2
  ( cd "$W" && { gh pr view mpm-followups > /dev/null 2>&1 || gh pr create --base main --head mpm-followups \
      --title "MPM.md: bearing with Isaac's default solve, repose at R/dx 20 (batch 04)" --body-file "$S/pr_body_docs.md"; } ) \
      > "$L/pr_docs.log" 2>&1
  cat "$L/pr_docs.log"; mark DOCS PASS "$(tail -1 "$L/pr_docs.log")"
fi

# ---------------------------------------------------------------- 2 deterministic branch on main
if ! done_ DETBRANCH; then
  git -C "$W" checkout -q -B batchsim-deterministic origin/main || { mark DETBRANCH FAIL "checkout"; exit 1; }
  git -C "$W" checkout 140b6fd9 -- metalsim/physics/batch.py tests/test_batch_deterministic.py \
      metalsim/physics/mpm/probes/scale_coupled.py || { mark DETBRANCH FAIL "restore"; exit 1; }
  cp "$S/rigid_det_diag.py" "$S/contact_audit.py" "$W/metalsim/physics/mpm/probes/"
  "$PY" -P "$S/edit_det_branch.py" "$W" || { mark DETBRANCH FAIL "edit"; exit 1; }
  git -C "$W" add metalsim/physics/batch.py tests/test_batch_deterministic.py metalsim/physics/mpm/probes/scale_coupled.py \
      metalsim/physics/mpm/probes/rigid_det_diag.py metalsim/physics/mpm/probes/contact_audit.py \
      metalsim/physics/mpm/coupled.py docs/GUIDE.md
  commit_msg "Deterministic BatchSim (experimental): re-add on main" \
"BatchSimOptions(deterministic=True) runs MuJoCo Warp's kernels in
Warp's RUN_TO_RUN mode (set before MuJoCo Warp's first launch, one mode
per process, record bound in scope, post-capture module check,
deterministic overflow in check_overflow), with its process-isolated
tests and the divergence probes (rigid_det_diag.py with a native-CCD
switch, contact_audit.py). Open: two stacked boxes diverge in 2 of 256
worlds from step 139; tests/test_batch_deterministic.py's stack-256
case fails until that is fixed, so this stays a draft."
  git -C "$W" push -q -u origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
  ( cd "$W" && { gh pr view batchsim-deterministic --json state -q .state 2>/dev/null | grep -q OPEN || \
      gh pr create --draft --base main --head batchsim-deterministic \
      --title "Deterministic BatchSim (draft: box-box divergence open)" --body-file "$S/pr_body_det.md"; } ) > "$L/pr_det.log" 2>&1
  cat "$L/pr_det.log"; mark DETBRANCH PASS "$(tail -1 "$L/pr_det.log")"
fi
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || git -C "$W" checkout -q batchsim-deterministic

# ---------------------------------------------------------------- 3 diagnostics
P=$W/metalsim/physics/mpm/probes
O=$W/runs/mpm_main/batch_05
mkdir -p "$O"
health
if ! done_ DIAG5; then
  gpu mm5_ccd_off check 20 "$L/stack256_nativeccd_off.log" -- "$PY" "$P/rigid_det_diag.py" --scene stack --worlds 256 \
      --steps 200 --nativeccd-off --json "$O/stack256_nativeccd_off.json"
  gpu mm5_tgfin_off check 20 "$L/stack256_tg_finalize_off.log" -- env WP_METAL_DET_TG_FINALIZE=0 "$PY" "$P/rigid_det_diag.py" \
      --scene stack --worlds 256 --steps 200 --json "$O/stack256_tg_finalize_off.json"
  gpu mm5_small_off check 20 "$L/stack256_small_records_off.log" -- env WP_METAL_DET_SMALL_RECORDS=0 "$PY" "$P/rigid_det_diag.py" \
      --scene stack --worlds 256 --steps 200 --json "$O/stack256_small_records_off.json"
  gpu mm5_audit check 25 "$L/contact_audit_ccd.log" -- "$PY" "$P/contact_audit.py" --worlds 256 --first 130 --last 150 \
      --json "$O/contact_audit_ccd.json"
  gpu mm5_audit_off check 25 "$L/contact_audit_nativeccd_off.log" -- "$PY" "$P/contact_audit.py" --worlds 256 --first 130 \
      --last 150 --nativeccd-off --json "$O/contact_audit_nativeccd_off.json"
  mark DIAG5 DONE "$(for f in stack256_nativeccd_off stack256_tg_finalize_off stack256_small_records_off; do echo -n "$f: $(grep '^{' "$L/$f.log" | tail -1 | cut -c1-220); "; done)"
fi
cp -R "$L" "$O/"
cp "$S/batch_05.sh" "$S/STATUS.txt" "$O/" 2>/dev/null
git -C "$W" add runs/mpm_main/batch_05
commit_msg "Deterministic BatchSim: batch 05 divergence diagnostics" "Native CCD off, Warp's single-threadgroup finalize off, small-record resolve off, contact-slot audit (batch_05)."
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
say "=== batch_05 end $(date)"; cat "$STATUS"
