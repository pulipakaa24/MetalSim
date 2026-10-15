#!/bin/bash
# sand_fidelity batch 03a (research_notes/sand_fidelity/PLAN_2026-10-14.md sections 17-18): the Lode-dependent yield
# surface as solver variants pr4660mc (Mohr-Coulomb, inner-rounded) / pr4660mn (Matsuoka-Nakai), the
# MPMConfig(fidelity=..., yield_surface=...) keys, the CPU tests, and the Metal / parity checks. No acceptance arms
# (those are 03b). Run outside the agent sandbox, alone:   bash scripts/sand_fidelity/batch_03a.sh
#
#   C  CPU checks before the commit: every intended file is changed (other dirty tracked paths must be run logs),
#      py_compile, bash -n, the variants regenerate from the vendored copy + patches.
#   K  commit (batch_23's template): explicit file list; `git add` exit code checked; staged count == list length;
#      after the commit `git status --porcelain` is empty for every listed file and the commit names exactly them.
#   T  tests on the committed tree (metalsim/, scripts/, tests/ clean): test_mpm_lode (kernel, config, material
#      conventions), test_mpm_variant, test_mpm_variant_pr4660 (CPU), test_sand_metrics, test_no_machine_paths.
#      Any failure or ANY SKIP stops the batch before GPU work.
#   G  GPU, small check jobs, one at a time (none is large):
#        lode_metal     the Lode kernel tests on Metal (METALSIM_LODE_DEVICE=metal:0)
#        pr4660_metal   test_mpm_variant_pr4660 on Metal (pr4660 files unchanged: regression guard)
#        parity_main    tests/test_mpm_main.py (.venv): the parity path, bitwise
#        parity_152     tests/test_mpm.py (.venv-newton152): parity against Newton 1.5.2 itself
#        mc_smoke       sand_repose voxel 0.01 (R/dx 5), P1d, 250 it, pr4660mc: finite, heap angle reported
#        mn_smoke       the same with pr4660mn
#      Every pytest job must pass with zero skips (parity_152's one xfail is the documented opt-in simd_gs case).
#   R  results commit (STATUS.txt and the live batch.log are left out).
#   P  push sand-fidelity-03a; draft PR based on sand-fidelity-02d (#75).
# Stops on any test failure or skip, watchdog rc 124-126, gpu_run rc 75, an idle-but-busy GPU, an out-of-memory
# error, a smoke with non-finite state; nothing is pushed after a stop. Resumable (STATUS.txt). Expected ~16 min
# (plan section 18.5).
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R=$(dirname "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)")
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
P=$W/metalsim/physics/mpm/probes
OUT=$W/runs/sand_fidelity/batch_03a
L=$OUT/logs
STATUS=$OUT/STATUS.txt
JOBS=$OUT/jobs.tsv
BRANCH=sand-fidelity-03a
BASE=sand-fidelity-02d
PLAN=research_notes/sand_fidelity/PLAN_2026-10-14.md
GEN=(NOTICE.txt __init__.py contact_solver_kernels.py implicit_mpm_model.py implicit_mpm_solver_kernels.py
     rasterized_collisions.py render_grains.py rheology_solver_kernels.py solve_rheology.py solver_implicit_mpm.py)
FILES=("$PLAN" THIRD_PARTY_NOTICES.md metalsim/physics/mpm/coupled.py metalsim/physics/mpm/newton_compat.py
       metalsim/physics/mpm/solver.py scripts/vendor_mpm_variant.py patches/metalsim/lode-yield-surface.diff
       scripts/sand_fidelity/lode/gc_check.py scripts/sand_fidelity/batch_03a.sh tests/test_mpm_lode.py)
for f in "${GEN[@]}"; do
  FILES+=("metalsim/physics/mpm/implicit_mpm_pr4660mc/$f" "metalsim/physics/mpm/implicit_mpm_pr4660mn/$f")
done
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"; touch "$STATUS"
[ -f "$JOBS" ] || printf 'name\trc\tseconds\test_seconds\tkind\tmem_gb\n' > "$JOBS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== sand_fidelity batch_03a start $(date), HEAD $(git -C "$W" rev-parse --short HEAD)"
say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE|FINDING)( |$)" "$STATUS"; }
err_() { grep -qE "^$1 (ERR|ANOMALY)( |$)" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }
fail() { mark "$1" FAIL "$2"; exit 1; }
jget() { "$PY" -P -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'r': r}))" "$1" "$2"; }
# pytest_clean LOG: the last summary line has passes and no failure, error or skip
pytest_clean() {
  local s; s=$(grep -E "^[0-9]+ (passed|failed)|^=+ .*(passed|failed)" "$1" | tail -1)
  echo "$s" | grep -q "passed" && ! echo "$s" | grep -qE "[0-9]+ failed|[0-9]+ errors?|[0-9]+ skipped"
}
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
job() {   # job NAME EST_SECONDS MEM_GB -- command...   (kind train = exclusive when MEM_GB >= 2); returns 1 to stop
  local name=$1 est=$2 mem=$3; shift 4
  if done_ "$name"; then say "skip $name (done)"; return 0; fi
  local kind=check; [ "$mem" -ge 2 ] && kind=train
  local tmo=$(( est * 3 > 600 ? est * 3 : 600 )) log=$L/$name.log t0 rc
  say "gpu job $name ($kind, ~${mem} GB, estimate ${est} s) -> logs/$name.log"
  t0=$(date +%s)
  "$R/scripts/gpu_run.sh" "sf03a_$name" $kind $(( tmo / 60 + 5 )) -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --rss-gb 40 --timeout "$tmo" --cwd "$W" -- "$@" > "$log" 2>&1 < /dev/null
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$est" "$kind" "$mem" >> "$JOBS"
  tail -3 "$log" | sed 's/^/    /'
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" ANOMALY "watchdog rc=$rc"; return 1
  elif [ $rc -eq 75 ]; then mark "$name" ANOMALY "gpu_run refused: runaway GPU work"; return 1
  elif grep -q "OutOfMemory\|out of memory" "$log"; then mark "$name" ANOMALY "GPU out of memory"; return 1
  elif [ $rc -ne 0 ]; then mark "$name" ERR "rc=$rc (see logs/$name.log)"; return 1
  fi
  mark "$name" DONE "$(( $(date +%s) - t0 )) s"
  health || return 1
  return 0
}
tjob() {  # a pytest GPU job: must pass with zero skips
  local name=$1
  job "$@" || return 1
  if ! pytest_clean "$L/$name.log"; then
    sed -i '' "/^$name DONE/d" "$STATUS"
    mark "$name" ANOMALY "pytest not clean (failure, error or skip): $(grep -E 'passed|failed' "$L/$name.log" | tail -1)"
    return 1
  fi
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
if ! done_ CPU_03A && ! done_ COMMIT_03A; then
  changed=$(git -C "$W" status --porcelain --untracked-files=all -- "${FILES[@]}" | awk '{print $2}' | sort)
  want=$(printf '%s\n' "${FILES[@]}" | sort)
  [ "$changed" = "$want" ] || { echo "want:"; echo "$want"; echo "changed:"; echo "$changed"; fail CPU_03A "not every intended file is changed"; }
  others=$(git -C "$W" status --porcelain --untracked-files=no | awk '{print $2}' | grep -v -x -F -f <(printf '%s\n' "${FILES[@]}") || true)
  [ -n "$others" ] && { say "other tracked changes (not committed by this batch):"; echo "$others" | sed 's/^/    /'; }
  echo "$others" | grep -qvE "^runs/|^$" && fail CPU_03A "tracked changes outside runs/ that this batch would leave behind"
  "$PY" -m py_compile "$W/metalsim/physics/mpm/coupled.py" "$W/metalsim/physics/mpm/newton_compat.py" \
      "$W/metalsim/physics/mpm/solver.py" "$W/scripts/vendor_mpm_variant.py" "$W/scripts/sand_fidelity/lode/gc_check.py" \
      "$W/tests/test_mpm_lode.py" "$W"/metalsim/physics/mpm/implicit_mpm_pr4660m[cn]/*.py || fail CPU_03A py_compile
  bash -n "$W/scripts/sand_fidelity/batch_03a.sh" || fail CPU_03A "bash -n"
  "$PY" -P "$W/scripts/vendor_mpm_variant.py" --check > "$L/variant_check.log" 2>&1 || fail CPU_03A "variants differ"
  git -C "$W" diff --quiet HEAD -- metalsim/physics/mpm/implicit_mpm metalsim/physics/mpm/implicit_mpm_pr4660 \
      metalsim/physics/mpm/implicit_mpm_pr4660c || fail CPU_03A "the vendored copy or an existing variant changed"
  mark CPU_03A PASS
fi

# ---------------------------------------------------------------- K: commit
if ! done_ COMMIT_03A; then
  commit_files "Sand fidelity batch 03a: Lode-dependent yield surface (pr4660mc / pr4660mn), fidelity config" \
"Plan sections 17-18 (option B). Solver variants pr4660mc / pr4660mn:
Newton PR #4660's variant plus patches/metalsim/lode-yield-surface.diff,
which scales the Drucker-Prager deviatoric radius by the generalized-
classical Lode function (Lagioia & Panteghini 2016): Mohr-Coulomb with
inner-rounded corners (beta 0.9999), or Matsuoka-Nakai. The Lode angle
comes from the world-frame deviator; the Lode factor is solved by a
bracketed secant (at most 8 anisotropic solves). The vendored parity
copy and the pr4660 / pr4660c variants are unchanged.

MPMConfig(fidelity='parity'|'physical', yield_surface=...): parity is
the default and resolves to the previous defaults; physical is P1d,
1000 iterations, Mohr-Coulomb. CPU tests pin the radii against the
exact surfaces, convexity, rotation invariance, return to the surface
(< 1e-3 up to Delassus anisotropy 100), bitwise pr4660 at k = 1, the
config validation, and the material conventions (yield_stress = tau_y
in simple shear; viscosity acts as tau = eta gamma_dot / 2)." "${FILES[@]}" || fail COMMIT_03A "see the lines above"
  mark COMMIT_03A PASS "$(git -C "$W" log --oneline -1)"
fi

# ---------------------------------------------------------------- T: tests on the committed tree
if ! done_ TESTS_03A; then
  [ -z "$(git -C "$W" status --porcelain --untracked-files=no -- metalsim scripts tests patches)" ] \
    || fail TESTS_03A "metalsim/, scripts/, tests/ or patches/ differ from the commit"
  env METALSIM_VARIANT_TESTS_METAL=0 "$PY" -P -m pytest -q -rs -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_mpm_lode.py" "$W/tests/test_mpm_variant.py" "$W/tests/test_mpm_variant_pr4660.py" \
      "$W/tests/test_sand_metrics.py" "$W/tests/test_no_machine_paths.py" > "$L/cpu_tests.log" 2>&1 \
    || { tail -15 "$L/cpu_tests.log"; fail TESTS_03A "see logs/cpu_tests.log"; }
  pytest_clean "$L/cpu_tests.log" || { tail -15 "$L/cpu_tests.log"; fail TESTS_03A "skipped tests (see logs/cpu_tests.log)"; }
  mark TESTS_03A PASS "$(tail -1 "$L/cpu_tests.log")"
fi
health || exit 90

# ---------------------------------------------------------------- G: GPU
REP=$P/sand_repose.py
run_all() {
  tjob lode_metal 120 1 -- env METALSIM_LODE_DEVICE=metal:0 "$PY" -P -m pytest -q -rs -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_mpm_lode.py" -k "not viscosity and not yield_stress_is and not config and not only_the" || return 1
  tjob pr4660_metal 104 1 -- env METALSIM_VARIANT_TESTS_METAL=1 METALSIM_VARIANT_TESTS_CPU=0 "$PY" -P -m pytest -q -rs \
      -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_variant_pr4660.py" || return 1
  tjob parity_main 316 1 -- "$PY" -P -m pytest -q -rs -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_main.py" || return 1
  tjob parity_152 114 1 -- "$PY152" -P -m pytest -q -rs -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm.py" || return 1
  local v
  for v in mc mn; do
    job ${v}_smoke 90 1 -- "$PY" "$REP" --voxel 0.01 --rest 2.0 --strain-basis P1d --solver-variant pr4660$v \
        --json "$OUT/${v}_smoke.json" || return 1
    [ "$(jget "$OUT/${v}_smoke.json" "r.get('error') is None and r['final']['angle_fit_mean_deg'] == r['final']['angle_fit_mean_deg']")" = True ] \
      || { mark ${v}_smoke ANOMALY "non-finite or error in ${v}_smoke.json"; return 1; }
    say "${v}_smoke: heap $(jget "$OUT/${v}_smoke.json" "round(r['final']['angle_fit_mean_deg'], 2)") deg at R/dx 5" \
        "(atan mu $(jget "$OUT/${v}_smoke.json" "round(r['atan_mu_deg'], 2)")), not gating"
  done
  return 0
}
if run_all; then mark GPU DONE; else ANOMALY=1; mark GPU STOPPED "see STATUS above"; fi

# ---------------------------------------------------------------- R: results commit
{
  echo "# sand_fidelity batch 03a: results"; echo
  echo "Gates (plan section 18.5): CPU tests, Lode kernels on Metal, pr4660 Metal tests, parity (main and 1.5.2), all"
  echo "with zero skips; Lode smokes finite. Heap angles at R/dx 5 are reported, not gated (03b is the acceptance)."; echo
  for n in TESTS_03A lode_metal pr4660_metal parity_main parity_152 mc_smoke mn_smoke; do
    echo "* $n: $(grep -E "^$n " "$STATUS" | tail -1 | cut -d' ' -f2-)"
  done
  for v in mc mn; do
    [ -s "$OUT/${v}_smoke.json" ] && echo "* ${v}_smoke heap: $(jget "$OUT/${v}_smoke.json" "round(r['final']['angle_fit_mean_deg'], 2)") deg" \
      "(atan mu $(jget "$OUT/${v}_smoke.json" "round(r['atan_mu_deg'], 2)") deg, R/dx 5, P1d, 250 it)"
  done
  for n in cpu_tests lode_metal pr4660_metal parity_main parity_152; do
    [ -s "$L/$n.log" ] && echo "* $n pytest: $(grep -E 'passed|failed' "$L/$n.log" | tail -1)"
  done
} > "$OUT/summary.md"
cat "$OUT/summary.md"
if ! done_ RESULTS_03A; then
  "$PY" -P -m pytest -q -p no:cacheprovider --rootdir "$W" "$W/tests/test_no_machine_paths.py" > "$L/paths_final.log" 2>&1 \
    || fail RESULTS_03A "machine paths"
  RFILES=("$OUT/summary.md" "$OUT/jobs.tsv")
  for f in "$OUT"/*.json "$L"/*.log; do
    case "$f" in */batch.log) ;; *) [ -e "$f" ] && RFILES+=("$f");; esac
  done
  RFILES=("${RFILES[@]#$W/}")
  commit_files "Sand fidelity batch 03a: results" \
"CPU, Metal and parity checks of the Lode-dependent yield surface
variants; runs/sand_fidelity/batch_03a/summary.md.$( [ $ANOMALY = 1 ] && printf '\n\nIncomplete: the batch stopped (STATUS.txt).')" \
    "${RFILES[@]}" || fail RESULTS_03A "see the lines above"
  mark RESULTS_03A PASS "$(git -C "$W" log --oneline -1)"
fi
if [ $ANOMALY = 1 ]; then say "stopped: committed locally, NOT pushed; fix, then rerun to resume."; exit 4; fi

# ---------------------------------------------------------------- P: push, draft PR
if ! done_ PR_03A; then
  for i in 1 2 3; do
    git -C "$W" push -q origin "HEAD:refs/heads/$BRANCH" 2>&1 | grep -v "^remote:" | tail -3
    [ "$(git -C "$W" ls-remote origin "refs/heads/$BRANCH" | cut -c1-40)" = "$(git -C "$W" rev-parse HEAD)" ] && break
    sleep 30
  done
  [ "$(git -C "$W" ls-remote origin "refs/heads/$BRANCH" | cut -c1-40)" = "$(git -C "$W" rev-parse HEAD)" ] || fail PR_03A push
  {
    echo "Sand-fidelity batch 03a (stacked on #75): option B's Lode-dependent yield surface as opt-in solver variants"
    echo "(\`pr4660mc\` Mohr-Coulomb, inner-rounded; \`pr4660mn\` Matsuoka-Nakai) and \`MPMConfig(fidelity=\"parity\"|\"physical\","
    echo "yield_surface=...)\`. Parity stays the default and unchanged (vendored copy untouched; parity tests below)."
    echo "Plan: research_notes/sand_fidelity/PLAN_2026-10-14.md sections 17-18. Acceptance arms (heap/layer/sphere/drop/rest,"
    echo "timing) are batch 03b."
    echo; grep -E "^\* " "$OUT/summary.md"
    echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$OUT/pr_body.md"
  ( cd "$W" && gh pr create --draft --base "$BASE" --head "$BRANCH" \
      --title "Sand fidelity 03a: Lode-dependent yield surface (Mohr-Coulomb / Matsuoka-Nakai), fidelity config" \
      --body-file "$OUT/pr_body.md" ) > "$L/pr.log.txt" 2>&1
  cat "$L/pr.log.txt"
  grep -q "github.com/.*/pull/" "$L/pr.log.txt" && mark PR_03A PASS "$(grep -o 'https://github.com/[^ ]*/pull/[0-9]*' "$L/pr.log.txt" | tail -1)" \
    || fail PR_03A "gh pr create"
fi
say "=== sand_fidelity batch_03a end $(date)"; cat "$STATUS"
