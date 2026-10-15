#!/bin/bash
# sand_fidelity batch 02d (research_notes/sand_fidelity/PLAN_2026-10-14.md section 15.3): particles-per-cell
# convergence and the heap / layer angles against mu, the facts for the preset decision (made by the owner, not here).
# Run outside the agent sandbox, alone:   bash scripts/sand_fidelity/batch_02d.sh
#
#   C  CPU checks before the commit: every intended file is changed (other dirty paths are listed: run logs from earlier
#      batches), py_compile, the solver variants regenerate.
#   K  commit (batch_23's template): explicit file list; `git add` exit code checked; staged count == list length;
#      after the commit `git status --porcelain` is empty for every listed file and the commit names exactly them.
#   T  tests on the committed tree (metalsim/, scripts/, tests/ must be clean): variant + sand-metric CPU tests (all
#      evaluators' self-tests), no machine paths. A failure stops the batch before any GPU work.
#   G  GPU (kind train = exclusive for >= 2 GB, the 1.04M-particle sphere included):
#        heap_mu080, heap_mu090   P1d + #4660, R/dx 10, 250 iterations (iterations do not move the heap: 02c)
#        tilt_pre_mu090           preset (P1d + #4660 + 1000 it) tilt, box floor, deposit angle
#        sph27_p1d                the P1d + #4660 sphere at 27 ppc (1.04M particles, P2 worktree, exclusive)
#        rest_v01_ppc64           parity resting box, voxel 0.01, 64 ppc (P2 worktree, exclusive)
#        rest_v01_pre27           preset resting box, voxel 0.01, 27 ppc: lowest corner and tilt
#      A tilt probe that ejects material or goes non-finite writes its JSON and is a FINDING; the batch goes on.
#   E  evaluator -> runs/sand_fidelity/batch_02d/summary.md + plan section 16 (facts, no choice)
#   R  results commit (same template; the live batch.log and STATUS.txt are left out: they change while committing)
#   P  push sand-fidelity-02d; draft PR based on sand-fidelity-02c (#73).
# Stops on watchdog rc 124-126, gpu_run rc 75, an idle-but-busy GPU, an out-of-memory error, non-finite state the probe
# did not localise; no push after an anomaly. Resumable (STATUS.txt). Expected ~26 min (plan section 15.3, 16.0).
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R=$(dirname "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)")
PY=$R/.venv/bin/python
P=$W/metalsim/physics/mpm/probes
WT_P2=$R/upstream/warp-innate-p2
OUT=$W/runs/sand_fidelity/batch_02d
L=$OUT/logs
STATUS=$OUT/STATUS.txt
JOBS=$OUT/jobs.tsv
BRANCH=sand-fidelity-02d
BASE=sand-fidelity-02c
PLAN=research_notes/sand_fidelity/PLAN_2026-10-14.md
FILES=("$PLAN" metalsim/physics/mpm/probes/tilt_box.py metalsim/physics/mpm/probes/bearing_fidelity.py
       metalsim/physics/mpm/probes/sphere_drop.py scripts/sand_fidelity/evaluate_batch_02d.py
       scripts/sand_fidelity/batch_02d.sh tests/test_sand_metrics.py)
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"; touch "$STATUS"
[ -f "$JOBS" ] || printf 'name\trc\tseconds\test_seconds\tkind\tmem_gb\n' > "$JOBS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== sand_fidelity batch_02d start $(date), HEAD $(git -C "$W" rev-parse --short HEAD)"
say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE|FINDING)( |$)" "$STATUS"; }
err_() { grep -qE "^$1 ERR( |$)" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }
fail() { mark "$1" FAIL "$2"; exit 1; }
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
job() {   # job NAME EST_SECONDS MEM_GB -- command...   (kind train = exclusive when MEM_GB >= 2)
  local name=$1 est=$2 mem=$3; shift 4
  if done_ "$name" || { err_ "$name" && [ "${RETRY_ERR:-0}" != 1 ]; }; then say "skip $name"; return 0; fi
  local kind=check; [ "$mem" -ge 2 ] && kind=train
  local tmo=$(( est * 3 > 600 ? est * 3 : 600 )) log=$L/$name.log t0 rc err
  say "gpu job $name ($kind, ~${mem} GB, estimate ${est} s) -> logs/$name.log"
  t0=$(date +%s)
  "$R/scripts/gpu_run.sh" "sf02d_$name" $kind $(( tmo / 60 + 5 )) -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --rss-gb 40 --timeout "$tmo" --cwd "$W" -- "$@" > "$log" 2>&1 < /dev/null
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$est" "$kind" "$mem" >> "$JOBS"
  tail -3 "$log" | sed 's/^/    /'
  err=""; [ -s "$OUT/$name.json" ] && err=$(jget "$OUT/$name.json" "r.get('error')")
  if [ $rc -eq 0 ]; then mark "$name" DONE "$(( $(date +%s) - t0 )) s"
  elif [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" ANOMALY "watchdog rc=$rc"; return 1
  elif [ $rc -eq 75 ]; then mark "$name" ANOMALY "gpu_run refused: runaway GPU work"; return 1
  elif [ $rc -eq 3 ] && { [ "$err" = nonfinite ] || [ "$err" = ejected ]; }; then mark "$name" FINDING "$err, localised in $name.json"
  elif [ $rc -eq 3 ]; then mark "$name" ANOMALY "non-finite / ejected state not localised"; return 1
  elif grep -q "OutOfMemory\|out of memory" "$log"; then mark "$name" ANOMALY "GPU out of memory"; return 1
  else mark "$name" ERR "rc=$rc (see logs/$name.log)"; fi
  health || return 1
  return 0
}
# commit_files SUBJECT BODY FILE...   batch_23's template: add (exit code checked), staged count == list, commit, then
# porcelain clean for the list and the commit names exactly the list
commit_files() {
  local subject=$1 body=$2; shift 2
  local files=("$@") n m left
  git -C "$W" add -- "${files[@]}" || return 1
  n=$(git -C "$W" diff --cached --name-only | wc -l | tr -d ' ')
  [ "$n" = "${#files[@]}" ] || { say "staged $n files, expected ${#files[@]}"; git -C "$W" diff --cached --name-only; return 1; }
  git -C "$W" commit -q -s -F - <<EOF || return 1
$subject

$body

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  left=$(git -C "$W" status --porcelain -- "${files[@]}")
  [ -z "$left" ] || { echo "$left"; say "listed files still modified after the commit"; return 1; }
  m=$(git -C "$W" show --name-only --format= HEAD | sed '/^$/d' | wc -l | tr -d ' ')
  [ "$m" = "${#files[@]}" ] || { say "the commit names $m files, expected ${#files[@]}"; return 1; }
  return 0
}

# ---------------------------------------------------------------- C: CPU checks before the commit
if ! done_ CPU_02D && ! done_ COMMIT_02D; then
  changed=$(git -C "$W" status --porcelain -- "${FILES[@]}" | awk '{print $2}' | sort)
  want=$(printf '%s\n' "${FILES[@]}" | sort)
  [ "$changed" = "$want" ] || { echo "want:"; echo "$want"; echo "changed:"; echo "$changed"; fail CPU_02D "not every intended file is changed"; }
  others=$(git -C "$W" status --porcelain --untracked-files=no | awk '{print $2}' | grep -v -x -F -f <(printf '%s\n' "${FILES[@]}") || true)
  [ -n "$others" ] && { say "other tracked changes (not committed by this batch):"; echo "$others" | sed 's/^/    /'; }
  echo "$others" | grep -qvE "^runs/|^$" && fail CPU_02D "tracked changes outside runs/ that this batch would leave behind"
  "$PY" -m py_compile "$P/tilt_box.py" "$P/bearing_fidelity.py" "$P/sphere_drop.py" \
      "$W/scripts/sand_fidelity/evaluate_batch_02d.py" "$W/tests/test_sand_metrics.py" || fail CPU_02D py_compile
  bash -n "$W/scripts/sand_fidelity/batch_02d.sh" || fail CPU_02D "bash -n"
  "$PY" -P "$W/scripts/vendor_mpm_variant.py" --check > "$L/variant_check.log" 2>&1 || fail CPU_02D "variants differ"
  mark CPU_02D PASS
fi

# ---------------------------------------------------------------- K: commit
if ! done_ COMMIT_02D; then
  commit_files "Sand fidelity batch 02d: ppc and mu arms, ejection finding, lowest-corner rest metric" \
"Plan section 15: 02c analysed (iterations do not move the heap; the
plane-shear deposit stands 2.5 deg above atan(mu) while the heap stands
4 deg below: Drucker-Prager's geometry dependence; ppc matters for
resting contact; the preset resting box rocks 3 deg; the parity tilt at
250 iterations ejected material). The tilt probe reports ejection as a
finding like NaN; the box probe reports the lowest corner and the tilt;
the sphere probe takes --ppc-axis; the 02d batch and its evaluator
(facts for the preset decision, no choice)." "${FILES[@]}" || fail COMMIT_02D "see the lines above"
  mark COMMIT_02D PASS "$(git -C "$W" log --oneline -1)"
fi

# ---------------------------------------------------------------- T: tests on the committed tree
if ! done_ TESTS_02D; then
  [ -z "$(git -C "$W" status --porcelain --untracked-files=no -- metalsim scripts tests)" ] \
    || fail TESTS_02D "metalsim/, scripts/ or tests/ differ from the commit"
  env METALSIM_VARIANT_TESTS_METAL=0 "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_variant.py" \
      "$W/tests/test_mpm_variant_pr4660.py" "$W/tests/test_sand_metrics.py" "$W/tests/test_no_machine_paths.py" \
      > "$L/cpu_tests.log" 2>&1 || { tail -5 "$L/cpu_tests.log"; fail TESTS_02D "see logs/cpu_tests.log"; }
  mark TESTS_02D PASS "$(tail -1 "$L/cpu_tests.log")"
fi
health || exit 90

# ---------------------------------------------------------------- G: GPU
REP=$P/sand_repose.py; TLT=$P/tilt_box.py; SPH=$P/sphere_drop.py; BRG=$P/bearing_fidelity.py
V=(--strain-basis P1d --solver-variant pr4660)
PRE=(--strain-basis P1d --solver-variant pr4660 --iterations 1000)
P2ENV=(env "PYTHONPATH=$WT_P2:$W")
P2OK=0
grep -q "size = max(nbytes, 2 \* capacity, 1 << 16)" "$WT_P2/warp/_src/metal_primitives.py" 2>/dev/null && P2OK=1
run_all() {
  job heap_mu080 140 1 -- "$PY" "$REP" --voxel 0.005 --friction 0.80 "${V[@]}" --json "$OUT/heap_mu080.json" || return 1
  job heap_mu090 140 1 -- "$PY" "$REP" --voxel 0.005 --friction 0.90 "${V[@]}" --json "$OUT/heap_mu090.json" || return 1
  job tilt_pre_mu090 710 1 -- "$PY" "$TLT" --friction 0.9 --mode runaway --hold 4 --floor box --start-below 4 "${PRE[@]}" \
      --npz "$OUT/tilt_pre_mu090.npz" --json "$OUT/tilt_pre_mu090.json" || return 1
  job rest_v01_pre27 160 1 -- "$PY" "$BRG" --rest --voxel 0.01 "${PRE[@]}" --json "$OUT/rest_v01_pre27.json" || return 1
  if [ $P2OK = 1 ]; then
    job sph27_p1d 270 5 -- "${P2ENV[@]}" "$PY" "$SPH" --walls --ppc-axis 3 "${V[@]}" --json "$OUT/sph27_p1d.json" || return 1
    job rest_v01_ppc64 85 3 -- "${P2ENV[@]}" "$PY" "$BRG" --rest --voxel 0.01 --ppc-axis 4 --json "$OUT/rest_v01_ppc64.json" || return 1
  else
    mark P2_ARMS ERR "P2 worktree $WT_P2 lacks the doubling amendment (pulipakaa24/warp#1): sph27 / ppc64 skipped"
  fi
  return 0
}
if run_all; then mark GPU DONE; else ANOMALY=1; mark GPU STOPPED "see STATUS above"; fi

# ---------------------------------------------------------------- E: evaluate
"$PY" -P "$W/scripts/sand_fidelity/evaluate_batch_02d.py" --results "$OUT" --plan "$W/$PLAN" > "$L/evaluate.log" 2>&1 \
  || fail EVALUATE "see logs/evaluate.log"
mark EVALUATE DONE
grep -E "^\* |^\|" "$OUT/summary.md" | tail -22

# ---------------------------------------------------------------- R: results commit
if ! done_ RESULTS_02D; then
  "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_no_machine_paths.py" > "$L/paths_final.log" 2>&1 \
    || fail RESULTS_02D "machine paths"
  RFILES=("$PLAN" "$OUT/summary.md" "$OUT/summary.json" "$OUT/jobs.tsv")
  for f in "$OUT"/*.json "$L"/*.log; do
    case "$f" in *summary.json|*/batch.log) ;; *) RFILES+=("$f");; esac
  done
  RFILES=("${RFILES[@]#$W/}")
  commit_files "Sand fidelity batch 02d: results" \
"ppc convergence (sphere 27 ppc, parity rest 64 ppc, preset rest 27 ppc
with lowest corner and tilt) and the heap / layer angles at mu 0.80 /
0.90: facts for the preset decision in runs/sand_fidelity/batch_02d/
summary.md and the plan note's section 16.$( [ $ANOMALY = 1 ] && printf '\n\nIncomplete: the batch stopped on an anomaly (STATUS.txt).')" \
    "${RFILES[@]}" || fail RESULTS_02D "see the lines above"
  mark RESULTS_02D PASS "$(git -C "$W" log --oneline -1)"
fi
if [ $ANOMALY = 1 ]; then say "anomaly: committed locally, NOT pushed; rerun to resume."; exit 4; fi

# ---------------------------------------------------------------- P: push, draft PR
if ! done_ PR_02D; then
  for i in 1 2 3; do
    git -C "$W" push -q origin "HEAD:refs/heads/$BRANCH" 2>&1 | grep -v "^remote:" | tail -3
    [ "$(git -C "$W" ls-remote origin "refs/heads/$BRANCH" | cut -c1-40)" = "$(git -C "$W" rev-parse HEAD)" ] && break
    sleep 30
  done
  [ "$(git -C "$W" ls-remote origin "refs/heads/$BRANCH" | cut -c1-40)" = "$(git -C "$W" rev-parse HEAD)" ] || fail PR_02D push
  {
    echo "Sand-fidelity batch 02d (stacked on #73): particles-per-cell convergence and heap / layer angles against mu,"
    echo "reported as facts for the physical-preset decision (calibrate mu on the heap and document the layer offset,"
    echo "or a Lode-dependent yield surface); the batch does not choose."
    echo; grep -E "^\* |^\|" "$OUT/summary.md" | tail -22
    echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$OUT/pr_body.md"
  ( cd "$W" && gh pr create --draft --base "$BASE" --head "$BRANCH" \
      --title "Sand fidelity batch 02d: ppc convergence and heap / layer vs mu (preset decision facts)" \
      --body-file "$OUT/pr_body.md" ) > "$L/pr.log.txt" 2>&1
  cat "$L/pr.log.txt"
  grep -q "github.com/.*/pull/" "$L/pr.log.txt" && mark PR_02D PASS "$(grep -o 'https://github.com/[^ ]*/pull/[0-9]*' "$L/pr.log.txt" | tail -1)" \
    || fail PR_02D "gh pr create"
fi
say "=== sand_fidelity batch_02d end $(date)"; cat "$STATUS"
