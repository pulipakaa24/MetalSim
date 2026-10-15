#!/bin/bash
# sand_fidelity batch 02b (research_notes/sand_fidelity/PLAN_2026-10-14.md section 11). Run outside the agent sandbox,
# alone (no other batch relayed in parallel):   bash scripts/sand_fidelity/batch_02b.sh
#
#   0  CPU checks: bash -n, py_compile, scripts/vendor_mpm_variant.py --check (the pr4660 variant regenerates exactly),
#      tests/test_mpm_variant.py, tests/test_mpm_variant_pr4660.py on Warp's CPU device (Newton PR #4660's own tests),
#      tests/test_sand_metrics.py, tests/test_no_machine_paths.py. Then two commits: (P1) the flag-gated solver variant
#      alone (scripts/vendor_mpm_variant.py, patches/newton/pr4660-*.diff, metalsim/physics/mpm/implicit_mpm_pr4660/,
#      the MPMConfig.solver_variant wiring, its tests, THIRD_PARTY_NOTICES.md), so it can be cherry-picked onto main
#      as its own PR; (02b) the probes, evaluator, this script and the plan note.
#   1  GPU. Each job names a queue kind and a device-memory estimate: >= 2 GB runs as kind "train" (exclusive: no
#      other job shares the GPU; gpu_lock has no memory hint, and two "check" jobs overlapping ran out of GPU memory in
#      02a), smaller ones as "check". Estimate: 0.5 GB + 3 KB per particle, + 2-3 GB of sort scratch for S2 above 1M
#      particles (plan 8.5); heuristic, the GPU-memory high-water is not logged anywhere yet.
#        A tests (gate: a failure stops the batch; nothing is pushed)
#          p1_tests     variant tests on Metal and CPU
#          parity_main  tests/test_mpm_main.py (.venv): the default path bitwise as before
#          parity_152   tests/test_mpm.py (.venv-newton152)
#        B P1 physics (MPMConfig(solver_variant="pr4660")): repose P1d / Q1d at R/dx 10, P1d at R/dx 20; walled sphere
#          P1d; tilt mu 0.68 P1d (box floor, runaway); C1 with the variant (edge-leak coverage)
#        C reruns: s_kin (OOM'd in 02a); t1b / t2b = 02a's t1 / t2 on a box floor (t1 / t2 used the leaking plane +
#          frictionless-wall junction, so they were invalid)
#        D with P2 (the Warp fork worktree upstream/warp-innate-p2 at f8a1d679, via PYTHONPATH; not pushed or merged):
#          s20a / s20b (1.15M particles), the voxel-0.005 S2 resting box with and without project_outside
#   2  evaluator -> runs/sand_fidelity/batch_02b/summary.md + plan section 12; results commit; push the stacked branch
#      sand-fidelity-02b; draft PR based on sand-fidelity-02a (#67).
# Stops on watchdog rc 124-126, gpu_run rc 75, a GPU busy with no owner, non-finite state the probe did not localise,
# or a failed test gate (after fixing, rerun with RETRY_ERR=1). Resumable (STATUS.txt). Expected ~55 min of GPU (arithmetic: plan section 11), plus any wait
# for the GPU to drain before the exclusive jobs.
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R=$(dirname "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)")
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
P=$W/metalsim/physics/mpm/probes
WT_P2=$R/upstream/warp-innate-p2
P2_COMMIT=f8a1d679
OUT=$W/runs/sand_fidelity/batch_02b
L=$OUT/logs
STATUS=$OUT/STATUS.txt
JOBS=$OUT/jobs.tsv
BRANCH=sand-fidelity-02b
BASE=sand-fidelity-02a
PLAN=research_notes/sand_fidelity/PLAN_2026-10-14.md
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"; touch "$STATUS"
[ -f "$JOBS" ] || printf 'name\trc\tseconds\test_seconds\tkind\tmem_gb\n' > "$JOBS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== sand_fidelity batch_02b start $(date), HEAD $(git -C "$W" rev-parse --short HEAD)"

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

# job NAME EST_SECONDS MEM_GB -- command...   (kind: train = exclusive when MEM_GB >= 2, else check)
# Returns 0 if the job ran (DONE, FINDING or ERR), 1 on an anomaly.
job() {
  local name=$1 est=$2 mem=$3; shift 4
  if done_ "$name" || { err_ "$name" && [ "${RETRY_ERR:-0}" != 1 ]; }; then say "skip $name (already in STATUS)"; return 0; fi
  local kind=check; [ "$mem" -ge 2 ] && kind=train
  local tmo=$(( est * 3 > 600 ? est * 3 : 600 )) log=$L/$name.log t0 rc
  say "gpu job $name ($kind, ~${mem} GB, estimate ${est} s, timeout ${tmo} s) -> logs/$name.log"
  t0=$(date +%s)
  "$R/scripts/gpu_run.sh" "sf02b_$name" $kind $(( tmo / 60 + 5 )) -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --rss-gb 40 --timeout "$tmo" --cwd "$W" -- "$@" > "$log" 2>&1 < /dev/null
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$est" "$kind" "$mem" >> "$JOBS"
  tail -3 "$log" | sed 's/^/    /'
  if [ $rc -eq 0 ]; then mark "$name" DONE "$(( $(date +%s) - t0 )) s"
  elif [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" ANOMALY "watchdog rc=$rc (124 timeout, 125 CPU stall, 126 memory)"; return 1
  elif [ $rc -eq 75 ]; then mark "$name" ANOMALY "gpu_run refused: runaway GPU work"; return 1
  elif [ $rc -eq 3 ] && [ -s "$OUT/$name.json" ] && [ "$(jget "$OUT/$name.json" "r.get('error')")" = nonfinite ]; then
    mark "$name" FINDING "non-finite, localised in $name.json"
  elif [ $rc -eq 3 ]; then mark "$name" ANOMALY "non-finite state not localised by the probe"; return 1
  elif grep -q "OutOfMemory\|out of memory" "$log"; then mark "$name" ANOMALY "GPU out of memory (logs/$name.log)"; return 1
  else mark "$name" ERR "rc=$rc (see logs/$name.log)"; fi
  health || return 1
  return 0
}

P1_FILES=(scripts/vendor_mpm_variant.py patches/newton/pr4660-1de47831-implicit-mpm.diff
  metalsim/physics/mpm/implicit_mpm_pr4660 metalsim/physics/mpm/solver.py metalsim/physics/mpm/coupled.py
  metalsim/physics/mpm/newton_compat.py tests/test_mpm_variant.py tests/test_mpm_variant_pr4660.py THIRD_PARTY_NOTICES.md)
CODE_FILES=("$PLAN" metalsim/physics/mpm/probes/sand_repose.py metalsim/physics/mpm/probes/sphere_drop.py
  metalsim/physics/mpm/probes/tilt_box.py scripts/sand_fidelity/evaluate_batch_02b.py scripts/sand_fidelity/batch_02b.sh
  tests/test_sand_metrics.py)

commit() {   # commit SUBJECT BODY  (only what is staged)
  git -C "$W" diff --cached --quiet && { say "nothing staged for: $1"; return 0; }
  git -C "$W" commit -q -s -F - <<EOF
$1

$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}

# ---------------------------------------------------------------------------------------------------- 0 CPU checks
CODE_HASH=$(cat "${CODE_FILES[@]:1}" "${P1_FILES[@]:0:2}" metalsim/physics/mpm/{solver,coupled,newton_compat}.py \
            tests/test_mpm_variant*.py | shasum | cut -c1-10)
if ! done_ "CPU_CHECKS_$CODE_HASH"; then
  ok=1
  bash -n "$W/scripts/sand_fidelity/batch_02b.sh" || ok=0
  "$PY" -m py_compile "$P"/sand_repose.py "$P"/sphere_drop.py "$P"/tilt_box.py "$W"/scripts/sand_fidelity/evaluate_batch_02b.py \
      "$W"/scripts/vendor_mpm_variant.py "$W"/metalsim/physics/mpm/implicit_mpm_pr4660/*.py || ok=0
  "$PY" -P "$W/scripts/vendor_mpm_variant.py" --check > "$L/variant_check.log" 2>&1 || ok=0
  git -C "$W" add -- "${P1_FILES[@]}" "${CODE_FILES[@]}"   # tracked before the path test: it checks tracked files only
  env METALSIM_VARIANT_TESTS_METAL=0 "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_mpm_variant.py" "$W/tests/test_mpm_variant_pr4660.py" "$W/tests/test_sand_metrics.py" \
      "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests.log" 2>&1 || ok=0
  tail -3 "$L/cpu_tests.log"
  if [ $ok = 1 ]; then mark "CPU_CHECKS_$CODE_HASH" PASS; else mark "CPU_CHECKS_$CODE_HASH" FAIL "see logs/"; exit 1; fi
fi
if ! done_ P1_COMMIT; then
  git -C "$W" reset -q -- "${CODE_FILES[@]}"
  git -C "$W" add -- "${P1_FILES[@]}"
  commit "MPM: flag-gated solver variant with Newton PR #4660 (default off)" \
"MPMConfig(solver_variant=\"pr4660\") selects a second copy of the vendored
Newton 1.5.2 implicit-MPM package with Newton draft PR #4660 (head
1de47831: P1d/Q1d nodal mass diagonalisation, Gauss-Legendre Q1d nodes,
sign-consistent material moments, angle-weighted collider pseudonormals,
viscosity only on strain nodes with volume) applied. The default
(\"parity\") is the vendored copy, untouched and never imported alongside:
newton_compat patches a variant's solve loop only once it is selected.
scripts/vendor_mpm_variant.py generates the variant from the vendored
files and patches/newton/pr4660-1de47831-implicit-mpm.diff (every hunk
applies without fuzz; the PR's base is Newton main), and the tests
regenerate and compare it. tests/test_mpm_variant_pr4660.py runs the
PR's own tests against the variant (CPU and Metal)." || { mark P1_COMMIT FAIL; exit 1; }
  mark P1_COMMIT PASS "$(git -C "$W" rev-parse --short HEAD)"
fi
if ! done_ "CODE_COMMIT_$CODE_HASH"; then
  git -C "$W" add -- "${CODE_FILES[@]}"
  commit "Sand fidelity batch 02b: variant / strain-basis probe options, evaluator, batch" \
"Probes take --solver-variant and (sphere, tilt) --strain-basis; the 02b
batch runs the P1 tests and parity suites first, then the P1 physics
arms, the 02a reruns and the P2 arms, with large jobs exclusive." || { mark "CODE_COMMIT_$CODE_HASH" FAIL; exit 1; }
  mark "CODE_COMMIT_$CODE_HASH" PASS "$(git -C "$W" rev-parse --short HEAD)"
fi
health || exit 90

# ---------------------------------------------------------------------------------------------------- 1 GPU
SPH=$P/sphere_drop.py; TLT=$P/tilt_box.py; REP=$P/sand_repose.py; BRG=$P/bearing_fidelity.py
V=(--solver-variant pr4660)
P2ENV=(env "PYTHONPATH=$WT_P2:$W")
gate() {   # gate NAME: a test job that must pass
  if done_ "$1"; then return 0; fi
  mark "$1" FAIL "see logs/$2.log"; GATE_FAILED=1; return 1
}
run_all() {
  # A tests
  job p1_tests 90 1 -- env METALSIM_VARIANT_TESTS_METAL=1 METALSIM_VARIANT_TESTS_CPU=0 "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_mpm_variant.py" "$W/tests/test_mpm_variant_pr4660.py" || return 1
  grep -q "^p1_tests DONE" "$STATUS" && mark P1_TESTS_METAL PASS || { gate P1_TESTS_METAL p1_tests; return 1; }
  job parity_main 280 1 -- "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_main.py" || return 1
  grep -q "^parity_main DONE" "$STATUS" && mark PARITY_MAIN PASS || { gate PARITY_MAIN parity_main; return 1; }
  job parity_152 110 1 -- "$PY152" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm.py" || return 1
  grep -q "^parity_152 DONE" "$STATUS" && mark PARITY_152 PASS || { gate PARITY_152 parity_152; return 1; }
  # B P1 physics
  job rep_p1d_v 160 1 -- "$PY" "$REP" --voxel 0.005 --strain-basis P1d "${V[@]}" --json "$OUT/rep_p1d_v.json" || return 1
  job rep_q1d_v 220 1 -- "$PY" "$REP" --voxel 0.005 --strain-basis Q1d "${V[@]}" --json "$OUT/rep_q1d_v.json" || return 1
  job rep_p1d_dx20_v 495 3 -- "$PY" "$REP" --voxel 0.0025 --strain-basis P1d "${V[@]}" --json "$OUT/rep_p1d_dx20_v.json" || return 1
  job sph_p1d_v 115 1 -- "$PY" "$SPH" --walls --strain-basis P1d "${V[@]}" --json "$OUT/sph_p1d_v.json" || return 1
  job tilt_p1d_v 320 1 -- "$PY" "$TLT" --friction 0.68 --mode runaway --hold 6 --floor box --strain-basis P1d "${V[@]}" \
      --json "$OUT/tilt_p1d_v.json" || return 1
  job c1_v 150 1 -- "$PY" "$TLT" --friction 0.4 --voxel 0.005 --fixed-tilt 20.3 --hold 8 "${V[@]}" --json "$OUT/c1_v.json" || return 1
  # C reruns
  job s_kin 199 2 -- "$PY" "$SPH" --walls --kinematic 0.02 --kin-depth 0.06 --fps 240 --json "$OUT/s_kin.json" || return 1
  job t1b 308 1 -- "$PY" "$TLT" --friction 0.68 --mode runaway --hold 6 --floor box --log-iterations --json "$OUT/t1b.json" || return 1
  job t2b 217 1 -- "$PY" "$TLT" --friction 0.9 --mode runaway --hold 6 --floor box --json "$OUT/t2b.json" || return 1
  # D with P2
  if [ -d "$WT_P2" ] && git -C "$WT_P2" merge-base --is-ancestor "$P2_COMMIT" HEAD 2>/dev/null; then
    job s20a 138 5 -- "${P2ENV[@]}" "$PY" "$SPH" --walls --voxel 0.0025 --bed 0.15 0.15 0.10 --json "$OUT/s20a.json" || return 1
    job s20b 138 5 -- "${P2ENV[@]}" "$PY" "$SPH" --walls --diameter 0.10 --bed 0.3 0.3 0.2 --json "$OUT/s20b.json" || return 1
    job rest_v005_S2 130 8 -- "${P2ENV[@]}" "$PY" "$BRG" --rest --voxel 0.005 --basis S2 --json "$OUT/rest_v005_S2.json" || return 1
    job rest_v005_S2_proj 130 8 -- "${P2ENV[@]}" "$PY" "$BRG" --rest --voxel 0.005 --basis S2 --project \
        --json "$OUT/rest_v005_S2_proj.json" || return 1
  else
    mark P2_ARMS ERR "P2 worktree $WT_P2 missing or not at $P2_COMMIT: D arms skipped"
  fi
  return 0
}
GATE_FAILED=0
if run_all; then mark GPU DONE; else ANOMALY=1; mark GPU STOPPED "see STATUS above"; fi
if [ $GATE_FAILED = 1 ]; then say "a test gate failed: results are not committed or pushed. See logs/."; exit 2; fi

# ---------------------------------------------------------------------------------------------------- 2 evaluate, commit, PR
"$PY" -P "$W/scripts/sand_fidelity/evaluate_batch_02b.py" --results "$OUT" --plan "$W/$PLAN" > "$L/evaluate.log" 2>&1 \
  && mark EVALUATE DONE || { mark EVALUATE FAIL "see logs/evaluate.log"; exit 3; }
grep -E "^\* " "$OUT/summary.md" | tail -25
"$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests_final.log" 2>&1 \
  || { mark FINAL_CHECKS FAIL "not committing"; exit 1; }
git -C "$W" add -- "$PLAN" "$OUT"/*.json "$OUT/summary.md" "$OUT/jobs.tsv" "$OUT/STATUS.txt" "$OUT/logs"
commit "Sand fidelity batch 02b: results" \
"P1 tests and parity suites, the #4660 variant on repose / sphere / tilt /
C1, the 02a reruns (s_kin, t1/t2 on a box floor) and the P2 arms.
Verdicts in runs/sand_fidelity/batch_02b/summary.md and the plan note's
section 12.$( [ $ANOMALY = 1 ] && printf '\n\nIncomplete: the batch stopped on an anomaly (STATUS.txt).')" \
  || { mark RESULTS_COMMIT FAIL; exit 1; }
mark RESULTS_COMMIT PASS "$(git -C "$W" rev-parse --short HEAD)"
if [ $ANOMALY = 1 ]; then say "anomaly: committed locally, NOT pushed; rerun to resume."; exit 4; fi
for i in 1 2 3; do
  git -C "$W" push -q origin "HEAD:refs/heads/$BRANCH" 2>&1 | grep -v "^remote:" | tail -3
  git -C "$W" ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | grep -q "$(git -C "$W" rev-parse HEAD)" && break
  say "push attempt $i did not land; retrying in 30 s"; sleep 30
done
git -C "$W" ls-remote origin "refs/heads/$BRANCH" | grep -q "$(git -C "$W" rev-parse HEAD)" && mark PUSH PASS "$BRANCH" \
  || { mark PUSH FAIL; exit 1; }
if ! done_ PR; then
  {
    echo "Sand-fidelity batch 02b (stacked on #67). Contains the P1 commit (flag-gated Newton PR #4660 variant, default"
    echo "off; parity suites bitwise) so it can be cherry-picked onto main as its own PR, and the 02b runs."
    echo
    grep -E "^\* " "$OUT/summary.md" | tail -25
    echo
    echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$OUT/pr_body.md"
  ( cd "$W" && gh pr create --draft --base "$BASE" --head "$BRANCH" \
      --title "Sand fidelity batch 02b: Newton #4660 variant (flag-gated), sphere resolution, tilt and contact reruns" \
      --body-file "$OUT/pr_body.md" ) > "$L/pr.log" 2>&1
  cat "$L/pr.log"
  ( cd "$W" && gh pr view "$BRANCH" --json url -q .url ) > /dev/null 2>&1 && mark PR PASS "$(tail -1 "$L/pr.log")" || mark PR FAIL
fi
say "=== sand_fidelity batch_02b end $(date)"; cat "$STATUS"
