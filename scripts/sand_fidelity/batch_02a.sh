#!/bin/bash
# sand_fidelity batch 02a (research_notes/sand_fidelity/PLAN_2026-10-14.md sections 8-9). Run it outside the agent
# sandbox:   bash scripts/sand_fidelity/batch_02a.sh        (DRY_RUN=1: control flow only, no GPU, no git writes)
#
# No solver changes: new probe options only (sphere walls / control / kinematic / density / 20 cells; tilt runaway,
# fixed tilt, floor and wall variants, iteration logging; box tilt recording). Isaac-parity behaviour untouched.
#
#   0  CPU checks (bash -n, py_compile, tests/test_sand_metrics.py incl. both evaluators' self-tests,
#      tests/test_no_machine_paths.py), then commit the code with the plan note's sections 8-9. A failed check
#      stops here: nothing is committed or pushed.
#   1  GPU, one job at a time through the main checkout's scripts/gpu_run.sh (kind "check": correctness runs,
#      14-inch-capable, default this Mac), each under scripts/watchdog.py (CPU stall 180 s, RSS 40 GB, timeout 3x the
#      estimate, at least 10 min), in the plan's order:
#        S sphere (8.6)        s_walled, s_ctrl_walled, s_ctrl_unwalled, s_kin, s_steel, s20a, s20b
#        L lag (8.7)           l_slip (repeat of batch 01 a4_dt480), l_h15_dt*, l_h05_dt*, l_v01_dt240/960
#        C mu 0.4 ladder (8.3) c1 base, c2 box floor, c3 rough side walls, c4 no side walls, c5 c1 logged
#        T tilt (8.1, 8.2)     t1 mu 0.68 runaway (iterations logged), t2 mu 0.9 runaway, t3/t3b creep at
#                              atan(mu) - 4 deg with 1000 / 250 iterations, r250/r1000 resting box voxel 0.01
#   2  evaluator -> runs/sand_fidelity/batch_02a/summary.md + plan note section 10; commit; push to the stacked
#      branch sand-fidelity-02a; open a DRAFT PR based on sand-fidelity-01 (PR #64).
#
# Stops launching GPU work on an anomaly: watchdog rc 124 (timeout) / 125 (CPU stall: possible GPU hang) / 126
# (memory), gpu_run rc 75 (runaway GPU work), a GPU busy with no owning process, or non-finite state that the probe did
# not localise (rc 3 without a "nonfinite" JSON). A localised tilt-box blow-up (rc 3 with its JSON) is a FINDING and
# the batch goes on; a probe that raises (rc 1, e.g. the 20-cell spheres hitting the Warp sort-scratch limit of plan
# 8.5) is ERR and the batch goes on. After an anomaly the results so far are committed locally, NOT pushed.
# Resumable: STATUS.txt; a rerun skips DONE / FINDING / ERR jobs (RETRY_ERR=1 reruns ERR jobs).
#
# Expected ~57 min, range 48-75 (arithmetic in the plan note, section 9; numbers from runs/gpu_history.jsonl).
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R=$(dirname "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)")
PY=$R/.venv/bin/python
P=$W/metalsim/physics/mpm/probes
DRY=${DRY_RUN:-0}
OUT=$W/runs/sand_fidelity/batch_02a
[ "$DRY" = 1 ] && OUT=${TMPDIR:-/tmp}/sf02a_dry
B01=$W/runs/sand_fidelity/batch_01
L=$OUT/logs
STATUS=$OUT/STATUS.txt
JOBS=$OUT/jobs.tsv
BRANCH=sand-fidelity-02a
BASE=sand-fidelity-01
PLAN=research_notes/sand_fidelity/PLAN_2026-10-14.md
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"
touch "$STATUS"
[ -f "$JOBS" ] || printf 'name\trc\tseconds\test_seconds\n' > "$JOBS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== sand_fidelity batch_02a start $(date) in $W (main checkout $R), HEAD $(git -C "$W" rev-parse --short HEAD)"

say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE|FINDING)( |$)" "$STATUS"; }
err_() { grep -qE "^$1 ERR( |$)" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }
jget() { "$PY" -P -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'r': r}))" "$1" "$2"; }
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

# job NAME EST_SECONDS -- command...   Returns 0 when the job ran (DONE, FINDING or ERR), 1 on an anomaly.
job() {
  local name=$1 est=$2; shift 3
  if done_ "$name" || { err_ "$name" && [ "${RETRY_ERR:-0}" != 1 ]; }; then say "skip $name (already in STATUS)"; return 0; fi
  local tmo=$(( est * 3 > 600 ? est * 3 : 600 ))
  local log=$L/$name.log t0 rc
  say "gpu job $name (estimate ${est} s, timeout ${tmo} s) -> logs/$name.log"
  if [ "$DRY" = 1 ]; then say "DRY $*"; mark "$name" DONE dry; return 0; fi
  t0=$(date +%s)
  "$R/scripts/gpu_run.sh" "sf02a_$name" check $(( tmo / 60 + 5 )) -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --rss-gb 40 --timeout "$tmo" --cwd "$W" -- "$@" > "$log" 2>&1 < /dev/null
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$est" >> "$JOBS"
  tail -3 "$log" | sed 's/^/    /'
  if [ $rc -eq 0 ] && [ -s "$OUT/$name.json" ]; then mark "$name" DONE "$(( $(date +%s) - t0 )) s"
  elif [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" ANOMALY "watchdog rc=$rc (124 timeout, 125 CPU stall, 126 memory)"; return 1
  elif [ $rc -eq 75 ]; then mark "$name" ANOMALY "gpu_run refused: runaway GPU work (runs/GPU_RUNAWAY_ALERT)"; return 1
  elif [ $rc -eq 3 ] && [ -s "$OUT/$name.json" ] && [ "$(jget "$OUT/$name.json" "r.get('error')")" = nonfinite ]; then
    mark "$name" FINDING "non-finite, localised in $name.json"
  elif [ $rc -eq 3 ]; then mark "$name" ANOMALY "non-finite state not localised by the probe"; return 1
  else mark "$name" ERR "rc=$rc (see logs/$name.log)"; fi
  health || return 1
  return 0
}

CODE_FILES=("$PLAN" metalsim/physics/mpm/probes/sand_metrics.py metalsim/physics/mpm/probes/solve_log.py
  metalsim/physics/mpm/probes/tilt_box.py metalsim/physics/mpm/probes/bearing_fidelity.py
  metalsim/physics/mpm/probes/sphere_drop.py scripts/sand_fidelity/evaluate_batch_02a.py
  scripts/sand_fidelity/batch_02a.sh tests/test_sand_metrics.py)

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
CODE_HASH=$(cat "${CODE_FILES[@]:1}" | shasum | cut -c1-10)   # code only: the plan note changes with every evaluation
if ! done_ "CPU_CHECKS_$CODE_HASH"; then
  ok=1
  bash -n "$W/scripts/sand_fidelity/batch_02a.sh" || ok=0
  "$PY" -m py_compile "$P/sand_metrics.py" "$P/solve_log.py" "$P/tilt_box.py" "$P/bearing_fidelity.py" \
      "$P/sphere_drop.py" "$W/scripts/sand_fidelity/evaluate_batch_02a.py" "$W/tests/test_sand_metrics.py" || ok=0
  gitw add -- "${CODE_FILES[@]}"                   # tracked before the path test: it checks tracked files only
  "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_sand_metrics.py" \
      "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests.log" 2>&1 || ok=0
  tail -3 "$L/cpu_tests.log"
  if [ $ok = 1 ]; then mark "CPU_CHECKS_$CODE_HASH" PASS; else mark "CPU_CHECKS_$CODE_HASH" FAIL "see logs/cpu_tests.log"; exit 1; fi
fi
if ! done_ "CODE_COMMIT_$CODE_HASH"; then
  gitw add -- "${CODE_FILES[@]}"
  commit "Sand fidelity batch 02a: analysis of batch 01, probes for the sphere, lag, mu 0.4 and tilt arms" \
"Plan note sections 8-9: batch 01 analysed against its raw JSONs (the
rheology solve never reaches its tolerance, also on CUDA in the L4
parity records; tilt creep below atan(mu); mu 0.4 edge leak; pic27
with project_outside invalid by construction; the voxel-0.005 S2 crash
is the Warp fork's sort scratch; the sphere bed was unwalled) and the
batch 02a draft. Probe options: sphere walls, ball-free control,
kinematic indentation, density, bed size; tilt runaway / fixed-tilt
modes, floor and wall variants, per-solve iteration logging
(solve_log.py); box tilt recording. Evaluator with pre-stated
predictions and its self-test." || { mark "CODE_COMMIT_$CODE_HASH" FAIL; exit 1; }
  mark "CODE_COMMIT_$CODE_HASH" PASS "$(git -C "$W" rev-parse --short HEAD 2>/dev/null)"
fi
health || exit 90

# ---------------------------------------------------------------------------------------------------- 1 GPU
SPH="$P/sphere_drop.py"; BRG="$P/bearing_fidelity.py"; TLT="$P/tilt_box.py"
run_all() {
  # S: sphere (plan 8.6). Uehara at mu 0.7, D 0.05, h 0.2: 16.3 mm
  job s_walled 79 -- "$PY" "$SPH" --walls --json "$OUT/s_walled.json" || return 1
  job s_ctrl_walled 79 -- "$PY" "$SPH" --walls --no-ball --json "$OUT/s_ctrl_walled.json" || return 1
  job s_ctrl_unwalled 79 -- "$PY" "$SPH" --no-ball --json "$OUT/s_ctrl_unwalled.json" || return 1
  job s_kin 199 -- "$PY" "$SPH" --walls --kinematic 0.02 --kin-depth 0.06 --fps 240 --json "$OUT/s_kin.json" || return 1
  job s_steel 79 -- "$PY" "$SPH" --walls --ball-density 7800 --json "$OUT/s_steel.json" || return 1
  job s20a 138 -- "$PY" "$SPH" --walls --voxel 0.0025 --bed 0.15 0.15 0.10 --json "$OUT/s20a.json" || return 1
  job s20b 138 -- "$PY" "$SPH" --walls --diameter 0.10 --bed 0.3 0.3 0.2 --json "$OUT/s20b.json" || return 1
  # L: lag (plan 8.7), impact-phase metric fixed in advance (sand_metrics.impact_phase)
  job l_slip 55 -- "$PY" "$BRG" --drop 0.15 --fps 480 --voxel 0.02 --json "$OUT/l_slip.json" || return 1
  local h tag fps est
  for h in 0.15 0.05; do
    tag=h${h#0.}; tag=${tag/./}
    for fps in 120 240 480 960; do
      case $fps in 120) est=25;; 240) est=29;; 480) est=38;; 960) est=57;; esac
      job "l_${tag}_dt$fps" $est -- "$PY" "$BRG" --drop $h --fps $fps --voxel 0.02 --seconds 0.8 \
          --json "$OUT/l_${tag}_dt$fps.json" || return 1
    done
  done
  job l_v01_dt240 51 -- "$PY" "$BRG" --drop 0.15 --fps 240 --voxel 0.01 --seconds 0.8 --json "$OUT/l_v01_dt240.json" || return 1
  job l_v01_dt960 142 -- "$PY" "$BRG" --drop 0.15 --fps 960 --voxel 0.01 --seconds 0.8 --json "$OUT/l_v01_dt960.json" || return 1
  # C: mu 0.4 reproduction ladder (plan 8.3): fixed tilt 20.3 deg (batch 01's hold), 8 s hold
  local C=(--friction 0.4 --voxel 0.005 --fixed-tilt 20.3 --hold 8)
  job c1 148 -- "$PY" "$TLT" "${C[@]}" --json "$OUT/c1.json" || return 1
  job c2 148 -- "$PY" "$TLT" "${C[@]}" --floor box --json "$OUT/c2.json" || return 1
  job c3 148 -- "$PY" "$TLT" "${C[@]}" --side-friction 1.0 --json "$OUT/c3.json" || return 1
  job c4 299 -- "$PY" "$TLT" "${C[@]}" --no-side-walls --width 0.1 --json "$OUT/c4.json" || return 1
  job c5 212 -- "$PY" "$TLT" "${C[@]}" --log-iterations --json "$OUT/c5.json" || return 1
  # T: tilt runaway, iterations (plan 8.1, 8.2); atan(0.68) - 4 = 30.2 deg
  job t1 308 -- "$PY" "$TLT" --friction 0.68 --mode runaway --hold 6 --log-iterations --json "$OUT/t1.json" || return 1
  job t2 217 -- "$PY" "$TLT" --friction 0.9 --mode runaway --hold 6 --json "$OUT/t2.json" || return 1
  job t3 255 -- "$PY" "$TLT" --friction 0.68 --fixed-tilt 30.2 --hold 3 --iterations 1000 --json "$OUT/t3.json" || return 1
  job t3b 86 -- "$PY" "$TLT" --friction 0.68 --fixed-tilt 30.2 --hold 3 --json "$OUT/t3b.json" || return 1
  job r250 63 -- "$PY" "$BRG" --rest --voxel 0.01 --log-iterations --json "$OUT/r250.json" || return 1
  job r1000 122 -- "$PY" "$BRG" --rest --voxel 0.01 --iterations 1000 --json "$OUT/r1000.json" || return 1
  return 0
}
if run_all; then mark GPU DONE; else ANOMALY=1; mark GPU STOPPED "anomaly: see STATUS above"; fi

# ---------------------------------------------------------------------------------------------------- 2 evaluate, commit, PR
"$PY" -P "$W/scripts/sand_fidelity/evaluate_batch_02a.py" --results "$OUT" --batch01 "$B01" \
    $( [ "$DRY" = 1 ] && echo --no-plan ) --plan "$W/$PLAN" > "$L/evaluate.log" 2>&1 \
  && mark EVALUATE DONE || { mark EVALUATE FAIL "see logs/evaluate.log"; exit 3; }
grep -E "^\* " "$OUT/summary.md" | tail -40
if ! "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests_final.log" 2>&1; then
  mark FINAL_CHECKS FAIL "see logs/cpu_tests_final.log: not committing"; exit 1
fi
gitw add -- "$PLAN" "$OUT"/*.json "$OUT/summary.md" "$OUT/jobs.tsv" "$OUT/STATUS.txt" "$OUT/logs"
commit "Sand fidelity batch 02a: results" \
"Sphere (walls, controls, kinematic indentation, steel, 20 cells), lag
reconfirmation with the impact-phase metric fixed in advance, the mu 0.4
reproduction ladder, tilt runaway and iteration logging. Verdicts against
the pre-stated predictions in runs/sand_fidelity/batch_02a/summary.md and
the plan note's section 10.$( [ $ANOMALY = 1 ] && printf '\n\nIncomplete: the batch stopped on an anomaly (STATUS.txt).')" \
  || { mark RESULTS_COMMIT FAIL; exit 1; }
mark RESULTS_COMMIT PASS "$(git -C "$W" rev-parse --short HEAD 2>/dev/null)"
if [ "$DRY" = 1 ]; then say "DRY: stop before push and PR"; cat "$STATUS"; exit 0; fi
if [ $ANOMALY = 1 ]; then
  say "anomaly: results committed locally, NOT pushed. Inspect STATUS.txt / logs, then rerun this script to resume."; exit 4
fi
for i in 1 2 3; do                                  # bounded retries
  git -C "$W" push -q origin "HEAD:refs/heads/$BRANCH" 2>&1 | grep -v "^remote:" | tail -3
  git -C "$W" ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | grep -q "$(git -C "$W" rev-parse HEAD)" && break
  say "push attempt $i did not land; retrying in 30 s"; sleep 30
done
if git -C "$W" ls-remote origin "refs/heads/$BRANCH" | grep -q "$(git -C "$W" rev-parse HEAD)"; then
  mark PUSH PASS "$BRANCH"
else mark PUSH FAIL; exit 1; fi
if ! done_ PR; then
  {
    echo "Sand-fidelity batch 02a (stacked on #64; no solver changes; Isaac-parity behaviour untouched)."
    echo
    echo "Plan: \`$PLAN\` sections 8-9 (batch 01 analysis, batch 02 draft); results in its section 10."
    echo "Batch: \`scripts/sand_fidelity/batch_02a.sh\`; evaluator: \`scripts/sand_fidelity/evaluate_batch_02a.py\`."
    echo
    grep -E "^\* " "$OUT/summary.md" | tail -30
    echo
    echo "Full tables: \`runs/sand_fidelity/batch_02a/summary.md\`."
    echo
    echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$OUT/pr_body.md"
  ( cd "$W" && gh pr create --draft --base "$BASE" --head "$BRANCH" \
      --title "Sand fidelity batch 02a: sphere impact, lag, mu 0.4 repro, tilt runaway and solver iterations" \
      --body-file "$OUT/pr_body.md" ) > "$L/pr.log" 2>&1
  cat "$L/pr.log"
  ( cd "$W" && gh pr view "$BRANCH" --json url -q .url ) > /dev/null 2>&1 && mark PR PASS "$(tail -1 "$L/pr.log")" || mark PR FAIL
fi
say "=== sand_fidelity batch_02a end $(date)"; cat "$STATUS"
