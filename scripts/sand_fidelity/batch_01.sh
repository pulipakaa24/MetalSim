#!/bin/bash
# sand_fidelity batch 01: the first GPU batch of the sand-fidelity workstream
# (research_notes/sand_fidelity/PLAN_2026-10-14.md, section 6). Run it outside the agent sandbox, from anywhere:
#
#     bash scripts/sand_fidelity/batch_01.sh          # in the worktree that holds this file
#
# No solver changes: existing code paths (MPMCoupling, the vendored Newton 1.5.2 solver) and new probes only.
# Isaac-parity behaviour is untouched (no library file changes).
#
#   0  CPU checks: bash -n, py_compile, tests/test_sand_metrics.py (incl. the evaluator self-test on synthetic data),
#      tests/test_no_machine_paths.py; then commit the code (plan note, probes, evaluator, tests, this script).
#      A failing check stops here: nothing is committed or pushed.
#   1  GPU, one job at a time through the main checkout's scripts/gpu_run.sh (queue kind "check": correctness runs,
#      14-inch-capable, default this Mac), each under scripts/watchdog.py (CPU-stall 180 s, RSS 40 GB, timeout):
#        gates first   a1_dx10 (repose R/dx 10 must reproduce repose3: 26.33 +- 0.3 deg)
#                      a5_v02_S2 (resting box must reproduce bearing3: 10.2 +- 1 mm)
#        arm 1 repose profile + stress (R1, R3)     a1_dx20, a1_tall
#        arm 4 drop dt sweep, converged (C1)        a4_dt120 .. a4_dt960, a4_dt480_v01
#        arm 3 basis variants at R/dx 10 (R2)       a3_b2, a3_q1d, a3_p1d, a3_gimp, a3_ppc64
#        arm 2 tilting box (B1, R3)                 a2_mu040, a2_mu068, a2_mu090, a2_mu068_v01
#        arm 6 sphere-drop smoke (B4)               a6_sphere
#        arm 5 resting box (C2)                     a5_{v02,v01,v005}_{S2,pic27}[_proj]  (voxel 0.005 last: ~4 min each)
#   2  evaluator -> runs/sand_fidelity/batch_01/summary.md (+ results section in the plan note); commit results;
#      push to origin sand-fidelity-01; open a DRAFT PR.
#
# Stops (no further GPU jobs) on an anomaly: a watchdog kill (rc 124 timeout / 125 CPU stall = possible GPU hang /
# 126 memory), a runaway GPU refusal (rc 75), non-finite state (rc 3), a GPU busy with no owning process, or a failed
# gate. After an anomaly the results so far are evaluated and committed locally but NOT pushed; after a failed gate
# nothing more is committed. A probe that raises (rc 1, e.g. a basis this solver does not support) is recorded as ERR
# and the batch goes on. Resumable: STATUS.txt records each step; a rerun skips DONE / ERR jobs
# (RETRY_ERR=1 reruns ERR jobs). Logs: runs/sand_fidelity/batch_01/logs/ (readable from the sandbox).
#
# Expected wall time ~62 min, range 58-78 (arithmetic: research note section 6b; numbers from runs/gpu_history.jsonl).
#
# DRY_RUN=1: control-flow check without GPU or git writes (prints each job's command, skips gates, commits, push
# and PR; results and logs go to $TMPDIR/sf01_dry). Used to verify this script from the agent sandbox.
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R=$(dirname "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)")
PY=$R/.venv/bin/python
P=$W/metalsim/physics/mpm/probes
DRY=${DRY_RUN:-0}
OUT=$W/runs/sand_fidelity/batch_01
[ "$DRY" = 1 ] && OUT=${TMPDIR:-/tmp}/sf01_dry
L=$OUT/logs
NPZ=$OUT/npz                      # positions + stresses at rest: large, kept locally, not committed
STATUS=$OUT/STATUS.txt
JOBS=$OUT/jobs.tsv
BRANCH=sand-fidelity-01
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$NPZ"
touch "$STATUS"
[ -f "$JOBS" ] || printf 'name\trc\tseconds\test_seconds\n' > "$JOBS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== sand_fidelity batch_01 start $(date) in $W (main checkout $R), HEAD $(git -C "$W" rev-parse --short HEAD)"

say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE)( |$)" "$STATUS"; }
err_() { grep -qE "^$1 ERR( |$)" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }
ANOMALY=0

health() {
  local u top
  u=$("$R/scripts/gpu_util.sh" 2>/dev/null); top=$(python3 "$R/scripts/gpu_top.py" 2>/dev/null | head -4)
  say "health: utilisation ${u:-?}%"; echo "$top"
  if [ "${u:-0}" -ge 10 ] && echo "$top" | grep -q "none >= 1 %"; then
    sleep 30; u=$("$R/scripts/gpu_util.sh" 2>/dev/null)
    if [ "${u:-0}" -ge 10 ] && python3 "$R/scripts/gpu_top.py" 2>/dev/null | grep -q "none >= 1 %"; then
      mark HEALTH FAIL "GPU ${u}% busy with no owning process"; return 1
    fi
  fi
  return 0
}

# job NAME EST_SECONDS -- command...   one GPU job: queue, watchdog, log, rc handling. Returns 0 if the job ran (DONE
# or ERR), 1 on an anomaly (the caller stops launching GPU work).
job() {
  local name=$1 est=$2; shift 3
  if done_ "$name" || { err_ "$name" && [ "${RETRY_ERR:-0}" != 1 ]; }; then say "skip $name (already in STATUS)"; return 0; fi
  local tmo=$(( est * 3 > 600 ? est * 3 : 600 ))
  local log=$L/$name.log t0 rc
  say "gpu job $name (estimate ${est} s, timeout ${tmo} s) -> logs/$name.log"
  if [ "$DRY" = 1 ]; then say "DRY $*"; mark "$name" DONE dry; return 0; fi
  t0=$(date +%s)
  "$R/scripts/gpu_run.sh" "sf01_$name" check $(( tmo / 60 + 5 )) -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --rss-gb 40 --timeout "$tmo" --cwd "$W" -- "$@" > "$log" 2>&1 < /dev/null
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$est" >> "$JOBS"
  tail -3 "$log" | sed 's/^/    /'
  if [ $rc -eq 0 ] && [ -s "$OUT/$name.json" ]; then mark "$name" DONE "$(( $(date +%s) - t0 )) s"
  elif [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" ANOMALY "watchdog rc=$rc (124 timeout, 125 CPU stall, 126 memory)"; return 1
  elif [ $rc -eq 75 ]; then mark "$name" ANOMALY "gpu_run refused: runaway GPU work (runs/GPU_RUNAWAY_ALERT)"; return 1
  elif [ $rc -eq 3 ]; then mark "$name" ANOMALY "non-finite particles or body state"; return 1
  else mark "$name" ERR "rc=$rc (see logs/$name.log)"; fi
  health || return 1
  return 0
}

jget() { "$PY" -P -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'r': r}))" "$1" "$2"; }

CODE_FILES=(research_notes/sand_fidelity/PLAN_2026-10-14.md metalsim/physics/mpm/probes/sand_metrics.py
  metalsim/physics/mpm/probes/sand_repose.py metalsim/physics/mpm/probes/tilt_box.py
  metalsim/physics/mpm/probes/bearing_fidelity.py metalsim/physics/mpm/probes/sphere_drop.py
  scripts/sand_fidelity/evaluate_batch_01.py scripts/sand_fidelity/batch_01.sh tests/test_sand_metrics.py)

gitw() { if [ "$DRY" = 1 ]; then say "DRY git $*"; else git -C "$W" "$@"; fi; }

commit() {   # commit SUBJECT BODY  (only what is staged)
  [ "$DRY" = 1 ] && { say "DRY commit: $1"; return 0; }
  git -C "$W" diff --cached --quiet && { say "nothing staged for: $1"; return 0; }
  git -C "$W" commit -q -s -F - <<EOF
$1

$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}

# ---------------------------------------------------------------------------------------------------- 0 CPU checks
if ! done_ CPU_CHECKS; then
  ok=1
  bash -n "$W/scripts/sand_fidelity/batch_01.sh" || ok=0
  "$PY" -m py_compile "$P/sand_metrics.py" "$P/sand_repose.py" "$P/tilt_box.py" "$P/bearing_fidelity.py" \
      "$P/sphere_drop.py" "$W/scripts/sand_fidelity/evaluate_batch_01.py" "$W/tests/test_sand_metrics.py" || ok=0
  gitw add -- "${CODE_FILES[@]}"                   # tracked before the path test: it checks tracked files only
  "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_sand_metrics.py" \
      "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests.log" 2>&1 || ok=0
  tail -3 "$L/cpu_tests.log"
  if [ $ok = 1 ]; then mark CPU_CHECKS PASS; else mark CPU_CHECKS FAIL "see logs/cpu_tests.log"; exit 1; fi
fi
if ! done_ CODE_COMMIT; then
  gitw add -- "${CODE_FILES[@]}"
  commit "Sand fidelity batch 01: research plan, probes, evaluator" \
"Research plan for physically right sand (research_notes/sand_fidelity/
PLAN_2026-10-14.md) and the probes of its first GPU batch, with no
solver change: sand_repose.py (lifted-cylinder repose with per-height
flank angles and flank stress invariants, basis/ppc variants),
tilt_box.py (tilting-box theta_fail / theta_stop by rotating gravity),
bearing_fidelity.py (box drop/rest with optional project_outside and
overlap/heave measures), sphere_drop.py (Uehara smoke test); shared CPU
measurements in sand_metrics.py with tests; the batch script and its
evaluator." || { mark CODE_COMMIT FAIL; exit 1; }
  mark CODE_COMMIT PASS "$(git -C "$W" rev-parse --short HEAD)"
fi
health || exit 90

# ---------------------------------------------------------------------------------------------------- 1 GPU
run_all() {
  # gates
  job a1_dx10 80 -- "$PY" "$P/sand_repose.py" --voxel 0.005 --json "$OUT/a1_dx10.json" --npz "$NPZ/a1_dx10.npz" || return 1
  if ! done_ GATE_REPOSE; then
    if [ "$DRY" = 1 ]; then say "DRY gate skipped"
    elif [ -s "$OUT/a1_dx10.json" ] && [ "$(jget "$OUT/a1_dx10.json" "abs(r['final']['angle_fit_mean_deg'] - 26.33) <= 0.3")" = True ]; then
      mark GATE_REPOSE PASS "$(jget "$OUT/a1_dx10.json" "round(r['final']['angle_fit_mean_deg'], 2)")"
    else mark GATE_REPOSE FAIL "R/dx 10 does not reproduce repose3 (26.33 deg)"; GATE_FAILED=1; return 1; fi
  fi
  job a5_v02_S2 28 -- "$PY" "$P/bearing_fidelity.py" --rest --voxel 0.02 --basis S2 --json "$OUT/a5_v02_S2.json" || return 1
  if ! done_ GATE_REST; then
    if [ "$DRY" = 1 ]; then say "DRY gate skipped"
    elif [ -s "$OUT/a5_v02_S2.json" ] && [ "$(jget "$OUT/a5_v02_S2.json" "abs(r['final_penetration_m'] - 0.0102) <= 0.001")" = True ]; then
      mark GATE_REST PASS "$(jget "$OUT/a5_v02_S2.json" "round(r['final_penetration_m'] * 1000, 1)") mm"
    else mark GATE_REST FAIL "resting box does not reproduce bearing3 (10.2 mm)"; GATE_FAILED=1; return 1; fi
  fi
  # arm 1
  job a1_dx20 410 -- "$PY" "$P/sand_repose.py" --voxel 0.0025 --json "$OUT/a1_dx20.json" --npz "$NPZ/a1_dx20.npz" || return 1
  job a1_tall 159 -- "$PY" "$P/sand_repose.py" --voxel 0.005 --fill 0.10 --json "$OUT/a1_tall.json" --npz "$NPZ/a1_tall.npz" || return 1
  # arm 4 (converged solve: 250 iterations, tolerance 1e-6 are bearing_fidelity's defaults)
  job a4_dt120 28 -- "$PY" "$P/bearing_fidelity.py" --drop 0.15 --fps 120 --voxel 0.02 --json "$OUT/a4_dt120.json" || return 1
  job a4_dt240 43 -- "$PY" "$P/bearing_fidelity.py" --drop 0.15 --fps 240 --voxel 0.02 --json "$OUT/a4_dt240.json" || return 1
  job a4_dt480 73 -- "$PY" "$P/bearing_fidelity.py" --drop 0.15 --fps 480 --voxel 0.02 --json "$OUT/a4_dt480.json" || return 1
  job a4_dt960 133 -- "$PY" "$P/bearing_fidelity.py" --drop 0.15 --fps 960 --voxel 0.02 --json "$OUT/a4_dt960.json" || return 1
  job a4_dt480_v01 173 -- "$PY" "$P/bearing_fidelity.py" --drop 0.15 --fps 480 --voxel 0.01 --json "$OUT/a4_dt480_v01.json" || return 1
  # arm 3 (R/dx 10, mu 0.68; baseline = a1_dx10)
  job a3_b2 78 -- "$PY" "$P/sand_repose.py" --voxel 0.005 --velocity-basis B2 --json "$OUT/a3_b2.json" || return 1
  job a3_q1d 78 -- "$PY" "$P/sand_repose.py" --voxel 0.005 --strain-basis Q1d --json "$OUT/a3_q1d.json" || return 1
  job a3_p1d 78 -- "$PY" "$P/sand_repose.py" --voxel 0.005 --strain-basis P1d --json "$OUT/a3_p1d.json" || return 1
  job a3_gimp 78 -- "$PY" "$P/sand_repose.py" --voxel 0.005 --integration-scheme gimp --json "$OUT/a3_gimp.json" || return 1
  job a3_ppc64 141 -- "$PY" "$P/sand_repose.py" --voxel 0.005 --ppc-axis 4 --json "$OUT/a3_ppc64.json" || return 1
  # arm 2
  job a2_mu040 195 -- "$PY" "$P/tilt_box.py" --friction 0.4 --voxel 0.005 --json "$OUT/a2_mu040.json" || return 1
  job a2_mu068 208 -- "$PY" "$P/tilt_box.py" --friction 0.68 --voxel 0.005 --json "$OUT/a2_mu068.json" || return 1
  job a2_mu090 215 -- "$PY" "$P/tilt_box.py" --friction 0.9 --voxel 0.005 --json "$OUT/a2_mu090.json" || return 1
  job a2_mu068_v01 102 -- "$PY" "$P/tilt_box.py" --friction 0.68 --voxel 0.01 --json "$OUT/a2_mu068_v01.json" || return 1
  # arm 6
  job a6_sphere 87 -- "$PY" "$P/sphere_drop.py" --h 0.2 --fps 480 --voxel 0.005 --json "$OUT/a6_sphere.json" || return 1
  # arm 5 (a5_v02_S2 ran as a gate)
  local v vox est b p
  for v in v02 v01 v005; do
    case $v in v02) vox=0.02; est=28;; v01) vox=0.01; est=53;; v005) vox=0.005; est=248;; esac
    for b in S2 pic27; do
      for p in "" _proj; do
        [ "$v$b$p" = v02S2 ] && continue
        job "a5_${v}_${b}${p}" $est -- "$PY" "$P/bearing_fidelity.py" --rest --voxel $vox --basis $b \
            ${p:+--project} --json "$OUT/a5_${v}_${b}${p}.json" || return 1
      done
    done
  done
  return 0
}
GATE_FAILED=0
if run_all; then mark GPU DONE; else ANOMALY=1; [ $GATE_FAILED = 1 ] || mark GPU STOPPED "anomaly: see STATUS above"; fi
if [ $GATE_FAILED = 1 ]; then
  say "a gate failed: nothing more is committed or pushed. Inspect logs/ and STATUS.txt."; exit 2
fi

# ---------------------------------------------------------------------------------------------------- 2 evaluate, commit, PR
"$PY" -P "$W/scripts/sand_fidelity/evaluate_batch_01.py" --results "$OUT" \
    $( [ "$DRY" = 1 ] && echo --no-plan ) --plan "$W/research_notes/sand_fidelity/PLAN_2026-10-14.md" > "$L/evaluate.log" 2>&1 \
  && mark EVALUATE DONE || { mark EVALUATE FAIL "see logs/evaluate.log"; exit 3; }
sed -n '/^## Verdicts/,/^## 1\./p' "$OUT/summary.md" | sed '$d'
if ! "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests_final.log" 2>&1; then
  mark FINAL_CHECKS FAIL "see logs/cpu_tests_final.log: not committing"; exit 1
fi
gitw add -- research_notes/sand_fidelity/PLAN_2026-10-14.md "$OUT"/*.json "$OUT/summary.md" "$OUT/jobs.tsv" \
    "$OUT/STATUS.txt" "$OUT/logs"
commit "Sand fidelity batch 01: results" \
"Runs of the six arms of research_notes/sand_fidelity/PLAN_2026-10-14.md
section 6 (repose profile and stress, tilting box, basis variants, drop
dt sweep, resting-box contact, sphere-drop smoke); verdicts against the
pre-stated predictions in runs/sand_fidelity/batch_01/summary.md and in
the plan note's results section.$( [ $ANOMALY = 1 ] && printf '\n\nIncomplete: the batch stopped on an anomaly (STATUS.txt).')" \
  || { mark RESULTS_COMMIT FAIL; exit 1; }
mark RESULTS_COMMIT PASS "$(git -C "$W" rev-parse --short HEAD)"
if [ "$DRY" = 1 ]; then say "DRY: stop before push and PR"; cat "$STATUS"; exit 0; fi
if [ $ANOMALY = 1 ]; then
  say "anomaly: results committed locally, NOT pushed. Inspect STATUS.txt / logs, then rerun this script to resume."; exit 4
fi

git -C "$W" push -q origin "HEAD:refs/heads/$BRANCH" 2>&1 | grep -v "^remote:" | tail -3
if git -C "$W" ls-remote origin "refs/heads/$BRANCH" | grep -q "$(git -C "$W" rev-parse HEAD)"; then
  mark PUSH PASS "$BRANCH"
else mark PUSH FAIL; exit 1; fi
if ! done_ PR; then
  {
    echo "Sand-fidelity workstream, first GPU batch (no solver changes; Isaac-parity behaviour untouched)."
    echo
    echo "Plan: \`research_notes/sand_fidelity/PLAN_2026-10-14.md\` (section 6; results in its section 7)."
    echo "Probes: \`metalsim/physics/mpm/probes/{sand_repose,tilt_box,bearing_fidelity,sphere_drop,sand_metrics}.py\`;"
    echo "batch: \`scripts/sand_fidelity/batch_01.sh\`; evaluator: \`scripts/sand_fidelity/evaluate_batch_01.py\`."
    echo
    sed -n '/^## Verdicts/,/^## 1\./p' "$OUT/summary.md" | sed '$d'
    echo "Full tables: \`runs/sand_fidelity/batch_01/summary.md\`."
    echo
    echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$OUT/pr_body.md"
  ( cd "$W" && gh pr create --draft --base main --head "$BRANCH" \
      --title "Sand fidelity batch 01: repose, tilting box, basis, dt and contact diagnostics" \
      --body-file "$OUT/pr_body.md" ) > "$L/pr.log" 2>&1
  cat "$L/pr.log"
  ( cd "$W" && gh pr view "$BRANCH" --json url -q .url ) > /dev/null 2>&1 && mark PR PASS "$(tail -1 "$L/pr.log")" || mark PR FAIL
fi
say "=== sand_fidelity batch_01 end $(date)"; cat "$STATUS"
