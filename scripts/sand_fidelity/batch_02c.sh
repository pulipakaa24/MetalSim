#!/bin/bash
# sand_fidelity batch 02c (research_notes/sand_fidelity/PLAN_2026-10-14.md section 13): the candidate physically-right
# preset (P1d strain + Newton #4660 variant + 1000 iterations) against physical references, plus the pinning reruns.
# Run outside the agent sandbox, alone, AFTER scripts/sand_fidelity/pr_p1_p2.sh (the rest runs use the amended P2):
#     bash scripts/sand_fidelity/batch_02c.sh           (FORCE_DX20=1 runs the R/dx 20 preset heap unconditionally)
#
#   0  CPU checks (variants regenerate, variant + sand tests on CPU, both evaluators' self-tests via
#      tests/test_sand_metrics.py, no machine paths), commit the 02c code.
#   1  GPU (kind train = exclusive when the device-memory estimate is >= 2 GB):
#        preset  rep10_pre (R/dx 10), tilt_pre (mu 0.68, box floor, runaway from atan(mu) - 4), sph_glass_pre,
#                sph_steel_pre, drop_pre (box drop, dt 1/960, voxel 0.01), rest_v005_pre (P2), and rep20_pre (R/dx 20,
#                ~22 min) only if rep10_pre rose >= 0.5 deg over its 250-iteration value (pre-stated: if iterations
#                do not move R/dx 10, the R/dx 20 run is not informative) or FORCE_DX20=1
#        checks  sph_steel_p1d (P1d, 250: the mass-scaling check of the 16.3 mm hit), sph20_p1d (P1d at 20 cells, P2),
#                rest_v005_ppc2 (parity, 8 ppc, P2: sized to fit the 55.7 GB working set), rest_v01_ppc2 (bridge),
#                c1_c (C1 with the collider-only variant pr4660c), t1b_fix (parity t1b with the fixed deposit fit)
#   2  evaluator -> runs/sand_fidelity/batch_02c/summary.md + plan section 14; commit; push sand-fidelity-02b (02b's
#      results were committed but not pushed: 02b stopped on its last job's OOM) and sand-fidelity-02c; draft PRs.
# Safety as in 02b (watchdog, anomalies stop GPU work, OOM is an anomaly, no push after an anomaly). Resumable.
# Expected ~54 min without rep20_pre, ~76 min with it (arithmetic: plan section 13).
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R=$(dirname "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)")
PY=$R/.venv/bin/python
P=$W/metalsim/physics/mpm/probes
WT_P2=$R/upstream/warp-innate-p2
OUT=$W/runs/sand_fidelity/batch_02c
L=$OUT/logs
STATUS=$OUT/STATUS.txt
JOBS=$OUT/jobs.tsv
B02B_COMMIT=d33423ec
PLAN=research_notes/sand_fidelity/PLAN_2026-10-14.md
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$OUT/npz"; touch "$STATUS"
[ -f "$JOBS" ] || printf 'name\trc\tseconds\test_seconds\tkind\tmem_gb\n' > "$JOBS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== sand_fidelity batch_02c start $(date), HEAD $(git -C "$W" rev-parse --short HEAD)"

say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE|FINDING)( |$)" "$STATUS"; }
err_() { grep -qE "^$1 ERR( |$)" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }
jget() { "$PY" -P -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'r': r}))" "$1" "$2"; }
ANOMALY=0
health() {
  local u
  u=$("$R/scripts/gpu_util.sh" 2>/dev/null); say "health: utilisation ${u:-?}%"
  if [ "${u:-0}" -ge 10 ] && python3 "$R/scripts/gpu_top.py" 2>/dev/null | grep -q "none >= 1 %"; then
    sleep 30; u=$("$R/scripts/gpu_util.sh" 2>/dev/null)
    if [ "${u:-0}" -ge 10 ] && python3 "$R/scripts/gpu_top.py" 2>/dev/null | grep -q "none >= 1 %"; then
      mark HEALTH FAIL "GPU ${u}% busy with no owning process"; return 1
    fi
  fi
  return 0
}
job() {   # job NAME EST_SECONDS MEM_GB -- command...
  local name=$1 est=$2 mem=$3; shift 4
  if done_ "$name" || { err_ "$name" && [ "${RETRY_ERR:-0}" != 1 ]; }; then say "skip $name"; return 0; fi
  local kind=check; [ "$mem" -ge 2 ] && kind=train
  local tmo=$(( est * 3 > 600 ? est * 3 : 600 )) log=$L/$name.log t0 rc
  say "gpu job $name ($kind, ~${mem} GB, estimate ${est} s) -> logs/$name.log"
  t0=$(date +%s)
  "$R/scripts/gpu_run.sh" "sf02c_$name" $kind $(( tmo / 60 + 5 )) -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --rss-gb 40 --timeout "$tmo" --cwd "$W" -- "$@" > "$log" 2>&1 < /dev/null
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$est" "$kind" "$mem" >> "$JOBS"
  tail -3 "$log" | sed 's/^/    /'
  if [ $rc -eq 0 ]; then mark "$name" DONE "$(( $(date +%s) - t0 )) s"
  elif [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" ANOMALY "watchdog rc=$rc"; return 1
  elif [ $rc -eq 75 ]; then mark "$name" ANOMALY "gpu_run refused: runaway GPU work"; return 1
  elif [ $rc -eq 3 ] && [ -s "$OUT/$name.json" ] && [ "$(jget "$OUT/$name.json" "r.get('error')")" = nonfinite ]; then
    mark "$name" FINDING "non-finite, localised in $name.json"
  elif [ $rc -eq 3 ]; then mark "$name" ANOMALY "non-finite state not localised"; return 1
  elif grep -q "OutOfMemory\|out of memory" "$log"; then mark "$name" ANOMALY "GPU out of memory"; return 1
  else mark "$name" ERR "rc=$rc (see logs/$name.log)"; fi
  health || return 1
  return 0
}
commit() {
  git -C "$W" diff --cached --quiet && { say "nothing staged for: $1"; return 0; }
  git -C "$W" commit -q -s -F - <<EOF
$1

$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}
CODE_FILES=("$PLAN" scripts/vendor_mpm_variant.py patches/newton/pr4660-1de47831-collider-only.diff
  metalsim/physics/mpm/implicit_mpm_pr4660c metalsim/physics/mpm/solver.py metalsim/physics/mpm/newton_compat.py
  metalsim/physics/mpm/probes/sand_metrics.py metalsim/physics/mpm/probes/tilt_box.py
  metalsim/physics/mpm/probes/bearing_fidelity.py metalsim/physics/mpm/probes/sphere_drop.py
  scripts/sand_fidelity/evaluate_batch_02c.py scripts/sand_fidelity/batch_02c.sh scripts/sand_fidelity/pr_p1_p2.sh
  scripts/sand_fidelity/p2/amend_growth.py tests/test_mpm_variant.py tests/test_sand_metrics.py THIRD_PARTY_NOTICES.md)

# ---------------------------------------------------------------------------------------------------- 0 CPU checks
CODE_HASH=$(cat "${CODE_FILES[@]:1:2}" "${CODE_FILES[@]:4}" | shasum | cut -c1-10)
if ! done_ "CPU_CHECKS_$CODE_HASH"; then
  ok=1
  bash -n "$W/scripts/sand_fidelity/batch_02c.sh" && bash -n "$W/scripts/sand_fidelity/pr_p1_p2.sh" || ok=0
  "$PY" -P "$W/scripts/vendor_mpm_variant.py" --check > "$L/variant_check.log" 2>&1 || ok=0
  git -C "$W" add -- "${CODE_FILES[@]}"
  env METALSIM_VARIANT_TESTS_METAL=0 "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_variant.py" \
      "$W/tests/test_mpm_variant_pr4660.py" "$W/tests/test_sand_metrics.py" "$W/tests/test_no_machine_paths.py" \
      > "$L/cpu_tests.log" 2>&1 || ok=0
  tail -3 "$L/cpu_tests.log"
  if [ $ok = 1 ]; then mark "CPU_CHECKS_$CODE_HASH" PASS; else mark "CPU_CHECKS_$CODE_HASH" FAIL "see logs/"; exit 1; fi
fi
if ! done_ "CODE_COMMIT_$CODE_HASH"; then
  git -C "$W" add -- "${CODE_FILES[@]}"
  commit "Sand fidelity batch 02c: preset arms, collider-only variant, deposit-fit fix" \
"Plan section 13: 02b analysed against its raw JSONs (the #4660 variant puts
the sphere at Uehara's depth and the heap at 30.7 deg; its collider sign
fix, not node selection, stops the floor / side-wall leak; t1b's -52 deg
deposit was the fit; the voxel-0.005 S2 rest needs ~58 GB). Research
variant pr4660c (the PR's collider hunks only), deposit fit on the pile's
free face, --iterations / --ppc-axis / --start-below probe options, the
02c batch and evaluator, the P1 / P2 PR script and the P2 growth amendment." \
  || { mark "CODE_COMMIT_$CODE_HASH" FAIL; exit 1; }
  mark "CODE_COMMIT_$CODE_HASH" PASS "$(git -C "$W" rev-parse --short HEAD)"
fi
health || exit 90

# ---------------------------------------------------------------------------------------------------- 1 GPU
SPH=$P/sphere_drop.py; TLT=$P/tilt_box.py; REP=$P/sand_repose.py; BRG=$P/bearing_fidelity.py
PRE=(--strain-basis P1d --solver-variant pr4660 --iterations 1000)
P2OK=0
[ -f "$WT_P2/warp/_src/metal_primitives.py" ] && grep -q "size = max(nbytes, 2 \* capacity, 1 << 16)" "$WT_P2/warp/_src/metal_primitives.py" && P2OK=1
P2ENV=(env "PYTHONPATH=$WT_P2:$W")
run_all() {
  job rep10_pre 390 1 -- "$PY" "$REP" --voxel 0.005 "${PRE[@]}" --json "$OUT/rep10_pre.json" || return 1
  job sph_glass_pre 290 1 -- "$PY" "$SPH" --walls "${PRE[@]}" --json "$OUT/sph_glass_pre.json" || return 1
  job sph_steel_pre 290 1 -- "$PY" "$SPH" --walls --ball-density 7800 "${PRE[@]}" --json "$OUT/sph_steel_pre.json" || return 1
  job sph_steel_p1d 103 1 -- "$PY" "$SPH" --walls --ball-density 7800 --strain-basis P1d --solver-variant pr4660 \
      --json "$OUT/sph_steel_p1d.json" || return 1
  job c1_c 128 1 -- "$PY" "$TLT" --friction 0.4 --voxel 0.005 --fixed-tilt 20.3 --hold 8 --solver-variant pr4660c \
      --json "$OUT/c1_c.json" || return 1
  job t1b_fix 214 1 -- "$PY" "$TLT" --friction 0.68 --mode runaway --hold 6 --floor box --npz "$OUT/npz/t1b_fix.npz" \
      --json "$OUT/t1b_fix.json" || return 1
  job tilt_pre 690 1 -- "$PY" "$TLT" --friction 0.68 --mode runaway --hold 4 --floor box --start-below 4 "${PRE[@]}" \
      --npz "$OUT/npz/tilt_pre.npz" --json "$OUT/tilt_pre.json" || return 1
  job drop_pre 415 1 -- "$PY" "$BRG" --drop 0.15 --fps 960 --voxel 0.01 --seconds 0.6 "${PRE[@]}" --json "$OUT/drop_pre.json" || return 1
  job rest_v01_ppc2 33 1 -- "$PY" "$BRG" --rest --voxel 0.01 --ppc-axis 2 --json "$OUT/rest_v01_ppc2.json" || return 1
  if [ $P2OK = 1 ]; then
    job rest_v005_ppc2 72 8 -- "${P2ENV[@]}" "$PY" "$BRG" --rest --voxel 0.005 --ppc-axis 2 --json "$OUT/rest_v005_ppc2.json" || return 1
    job rest_v005_pre 284 8 -- "${P2ENV[@]}" "$PY" "$BRG" --rest --voxel 0.005 --ppc-axis 2 "${PRE[@]}" \
        --json "$OUT/rest_v005_pre.json" || return 1
    job sph20_p1d 334 5 -- "${P2ENV[@]}" "$PY" "$SPH" --walls --voxel 0.0025 --bed 0.15 0.15 0.10 --strain-basis P1d \
        --solver-variant pr4660 --json "$OUT/sph20_p1d.json" || return 1
  else
    mark P2_ARMS ERR "the P2 worktree does not have the doubling amendment: run pr_p1_p2.sh first"
  fi
  local go=${FORCE_DX20:-0}
  if [ "$go" != 1 ] && [ -s "$OUT/rep10_pre.json" ] && \
     [ "$(jget "$OUT/rep10_pre.json" "r['final']['angle_fit_mean_deg'] - 29.89 >= 0.5")" = True ]; then go=1; fi
  if [ "$go" = 1 ]; then
    job rep20_pre 1335 3 -- "$PY" "$REP" --voxel 0.0025 "${PRE[@]}" --json "$OUT/rep20_pre.json" || return 1
  else
    done_ rep20_pre || mark rep20_pre SKIPPED "rep10_pre did not rise >= 0.5 deg (pre-stated condition)"
  fi
  return 0
}
if run_all; then mark GPU DONE; else ANOMALY=1; mark GPU STOPPED "see STATUS above"; fi

# ---------------------------------------------------------------------------------------------------- 2 evaluate, commit, PRs
"$PY" -P "$W/scripts/sand_fidelity/evaluate_batch_02c.py" --results "$OUT" --plan "$W/$PLAN" > "$L/evaluate.log" 2>&1 \
  && mark EVALUATE DONE || { mark EVALUATE FAIL "see logs/evaluate.log"; exit 3; }
grep -E "^\* " "$OUT/summary.md" | tail -20
"$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests_final.log" 2>&1 \
  || { mark FINAL_CHECKS FAIL "not committing"; exit 1; }
git -C "$W" add -- "$PLAN" "$OUT"/*.json "$OUT/summary.md" "$OUT/jobs.tsv" "$OUT/STATUS.txt" "$OUT/logs"
commit "Sand fidelity batch 02c: results" \
"The P1d + #4660 + 1000-iteration preset against atan(mu), Uehara and B5;
mass and resolution checks of the sphere; the collider-only variant on
C1; the resized voxel-0.005 rest runs. Verdicts in
runs/sand_fidelity/batch_02c/summary.md and the plan note's section 14.$( [ $ANOMALY = 1 ] && printf '\n\nIncomplete: the batch stopped on an anomaly (STATUS.txt).')" \
  || { mark RESULTS_COMMIT FAIL; exit 1; }
mark RESULTS_COMMIT PASS "$(git -C "$W" rev-parse --short HEAD)"
if [ $ANOMALY = 1 ]; then say "anomaly: committed locally, NOT pushed; rerun to resume."; exit 4; fi
pushref() {   # pushref COMMIT BRANCH
  for i in 1 2 3; do
    git -C "$W" push -q origin "$1:refs/heads/$2" 2>&1 | grep -v "^remote:" | tail -3
    git -C "$W" ls-remote origin "refs/heads/$2" 2>/dev/null | grep -q "$(git -C "$W" rev-parse "$1")" && return 0
    sleep 30
  done
  return 1
}
pr() {   # pr HEAD BASE TITLE BODYFILE MARK
  done_ "$5" && return 0
  ( cd "$W" && gh pr view "$1" > /dev/null 2>&1 ) && { mark "$5" PASS "exists"; return 0; }
  ( cd "$W" && gh pr create --draft --base "$2" --head "$1" --title "$3" --body-file "$4" ) > "$L/$5.log" 2>&1
  cat "$L/$5.log"; grep -q "github.com" "$L/$5.log" && mark "$5" PASS "$(tail -1 "$L/$5.log")" || mark "$5" FAIL
}
pushref "$B02B_COMMIT" sand-fidelity-02b && mark PUSH_02B PASS || { mark PUSH_02B FAIL; exit 1; }
pushref HEAD sand-fidelity-02c && mark PUSH_02C PASS || { mark PUSH_02C FAIL; exit 1; }
for b in 02b 02c; do
  {
    if [ $b = 02b ]; then
      echo "Sand-fidelity batch 02b (stacked on #67): the Newton #4660 variant (flag-gated, default off; P1 also opened on"
      echo "its own from main) on heap / sphere / tilt / C1, the 02a reruns and the P2 arms. Results:"
      echo "\`runs/sand_fidelity/batch_02b/summary.md\`; analysis in the plan note's section 13 (02c PR)."
    else
      echo "Sand-fidelity batch 02c (stacked on the 02b PR): the candidate physically-right preset against physical references."
      echo; grep -E "^\* " "$OUT/summary.md" | tail -16
    fi
    echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$OUT/pr_body_$b.md"
done
pr sand-fidelity-02b sand-fidelity-02a "Sand fidelity batch 02b: Newton #4660 variant, sphere resolution, tilt and contact reruns" \
   "$OUT/pr_body_02b.md" PR_02B
pr sand-fidelity-02c sand-fidelity-02b "Sand fidelity batch 02c: candidate physically-right preset (P1d + #4660 + 1000 iterations)" \
   "$OUT/pr_body_02c.md" PR_02C
say "=== sand_fidelity batch_02c end $(date)"; cat "$STATUS"
