#!/bin/bash
# batch 10 (supersedes batch_09, which pointed at batch_08's state; run outside the agent sandbox):  bash scratch/b10/batch_10.sh
#
# Coupling-opt PR #33, follow-up of batch_07 (worktree scratch/wt-copt, branch coupling-opt):
#   - the one-world masked reset (it raised: Newton 1.5.2's masked reset refuses grid warm starts without separate FEM
#     environments) now runs through MPMCoupling._solver_reset; probes/one_world_reset.py checks it against the
#     solver's full reset; tests/test_mpm_main.py (+ test_one_world_masked_reset)
#   - policy_step_ab.py runs every A / B run in its own process (batch_07's 1,024-world run ran out of memory: the
#     probe kept every run's scene alive in one process); re-timed at 1 / 64 / 1,024 worlds (timing class, 16-inch)
#   - the numbers into MPM.md section 11, GUIDE, the research note, CHANGELOG (scratch/b10/copt_docs.py); commit, push,
#     PR #33 body replaced and out of draft (only if every check passed)
# Part 2, deterministic-mode cost for PR #29 (det worktree, branch batchsim-deterministic, after batch_08b run 1):
#   BatchSim deterministic=True vs default on the G1 flat velocity task (scripts/diagnostics/det_cost_g1.py) and the
#   coupled box scene (probes/scale_coupled.py --time --deterministic) at 64 and 1,024 worlds, every run its own
#   process, 3 interleaved repetitions, timing class; numbers into the research note, GAPS and #29's body
#   (scratch/b10/det_docs.py, scratch/b08b/pr29_body.md, which batch_08b's phase D also uses).
# Run after the overnight line (GPU exclusive). Resumable (scratch/b10/STATUS.txt). ~1.5 h of GPU.
set -u
W=.
R=.
S=$W/scratch/b10
L=$S/logs
WT=$W/scratch/wt-copt
O=$WT/runs/mpm_main/batch_10
PY=$R/.venv/bin/python
P=$WT/metalsim/physics/mpm/probes
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_10 start $(date)"

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
gpu() {   # gpu NAME KIND MIN LOG -- cmd...   (cwd = $GCWD, PYTHONPATH = $GPYP; default the coupling-opt worktree)
  local name=$1 kind=$2 min=$3 log=$4; shift 5
  local cwd=${GCWD:-$WT} pyp=${GPYP:-$WT}
  say "gpu job $name ($kind, <= $min min, cwd $cwd) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$cwd" -- env PYTHONPATH="$pyp" "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -3 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
[ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = coupling-opt ] || { say "scratch/wt-copt is not on coupling-opt"; exit 1; }
health

if ! done_ ONEWORLD10; then
  gpu mm10_oneworld check 15 "$L/one_world_reset.log" -- "$PY" -P "$P/one_world_reset.py" --json "$O/one_world_reset.json"
  rc=$?
  [ $rc = 0 ] && mark ONEWORLD10 PASS "$(grep '^{' "$L/one_world_reset.log" | tail -1)" || mark ONEWORLD10 FAIL "rc=$rc $(grep '^{' "$L/one_world_reset.log" | tail -1)"
fi
if ! done_ TESTS10; then
  gpu mm10_tests check 40 "$L/test_mpm_main.log" -- "$PY" -P -m pytest -q -rfE tests/test_mpm_main.py
  rc=$?
  [ $rc = 0 ] && mark TESTS10 PASS "$(tail -1 "$L/test_mpm_main.log")" || mark TESTS10 FAIL "rc=$rc $(tail -1 "$L/test_mpm_main.log")"
fi
for n in 1 64 1024; do
  done_ POLICY10_$n && continue
  steps=200; min=30; [ $n = 1024 ] && { steps=60; min=60; }
  gpu mm10_policy_$n timing $min "$L/policy_$n.log" -- "$PY" -P "$P/policy_step_ab.py" --worlds $n --fraction 0.02 \
      --steps $steps --warmup 10 --rounds 3 --hash-steps 40 --json "$O/policy_$n.json"
  rc=$?
  [ $rc = 0 ] && [ -s "$O/policy_$n.json" ] && mark POLICY10_$n PASS "$(grep '^{' "$L/policy_$n.log" | tail -1 | cut -c1-400)" \
      || mark POLICY10_$n FAIL "rc=$rc (see $L/policy_$n.log)"
done

cp -R "$L" "$O/"; cp "$S/batch_10.sh" "$S/copt_docs.py" "$STATUS" "$O/" 2>/dev/null
ok=1; grep -qE " FAIL " "$STATUS" && ok=0
if [ $ok = 1 ] && ! done_ DOCS10; then
  python3 "$S/copt_docs.py" "$WT" "$WT/runs/mpm_main/batch_07" "$O" > "$L/docs.log" 2>&1 && mark DOCS10 PASS || { mark DOCS10 FAIL "$(tail -2 "$L/docs.log")"; ok=0; }
  cat "$L/docs.log"
fi
git -C "$WT" add metalsim/physics/mpm/coupled.py metalsim/physics/mpm/probes/policy_step_ab.py \
    metalsim/physics/mpm/probes/one_world_reset.py tests/test_mpm_main.py runs/mpm_main/batch_10
[ $ok = 1 ] && git -C "$WT" add metalsim/physics/mpm/MPM.md docs/GUIDE.md CHANGELOG.md \
    "research_notes/MPM in main simulator/coupling_opt.md"
git -C "$WT" diff --cached --quiet || git -C "$WT" commit -q -s -F - <<'EOF'
MPMCoupling: one-world masked reset; policy-step A/B per process; docs

With one world Newton 1.5.2's solver has no separate FEM environments
and its masked reset refused grid warm starts, so mpm.reset(mask)
raised; MPMCoupling now launches the same reset steps with the mask
read on the device (bitwise the solver's full reset when selected,
nothing otherwise; probes/one_world_reset.py, new test).
policy_step_ab.py runs every A / B run in its own process (the
1,024-world run had run out of memory with every scene alive in one
process). Batch 10: the A/B at 1 / 64 / 1,024 worlds; MPM.md section
11, GUIDE, the research note and CHANGELOG get the measured numbers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$WT" push -q origin coupling-opt 2>&1 | grep -v "^remote:" | tail -2
if [ $ok = 1 ]; then
  ( cd "$WT" && gh pr edit 33 --title "MPMCoupling: launch_step and capturable per-world reset" --body-file "$S/pr33_body.md" && gh pr ready 33 ) > "$L/pr.log" 2>&1
  cat "$L/pr.log"; mark PR10 DONE "$(tail -1 "$L/pr.log")"
else
  say "a check failed: PR #33 stays a draft"
fi

# ---------------------------------------------------------------- part 2: deterministic-mode cost (PR #29)
B8=$W/scratch/b08b
OD=$W/runs/mpm_main/batch_10_det
mkdir -p "$OD"
if ! grep -qE "^DOCS8 PASS" "$B8/STATUS.txt" 2>/dev/null; then
  say "batch_08b run 1 has not reached DOCS8: skipping the deterministic-mode cost"; cat "$STATUS"; exit 0
fi
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
if ! done_ DETCOST10; then
  mkdir -p "$W/scripts/diagnostics"
  cp "$S/det_cost_g1.py" "$W/scripts/diagnostics/det_cost_g1.py"
  export GCWD=$W GPYP=$W/upstream/mujoco_warp-detccd:$W/upstream/warp-tripwire-git:$W
  env PYTHONPATH="$GPYP" "$PY" -P -c "import warp, mujoco_warp, metalsim; print(warp.__file__, mujoco_warp.__file__, metalsim.__file__)" | tee "$L/det_imports.log"
  fails=0
  for r in 1 2 3; do
    for n in 64 1024; do
      for d in 0 1; do
        [ -s "$OD/g1_${n}_${d}_r$r.json" ] || gpu mm10_g1_${n}_${d}_r$r timing 20 "$L/g1_${n}_${d}_r$r.log" -- "$PY" -P \
            "$W/scripts/diagnostics/det_cost_g1.py" --worlds $n --deterministic $d --frames 100 --warmup 20 \
            --json "$OD/g1_${n}_${d}_r$r.json" || fails=$((fails + 1))
        [ -s "$OD/box_${n}_${d}_r$r.json" ] || gpu mm10_box_${n}_${d}_r$r timing 20 "$L/box_${n}_${d}_r$r.log" -- "$PY" -P \
            "$W/metalsim/physics/mpm/probes/scale_coupled.py" --worlds $n --deterministic $d --time --steps 40 --warmup 10 \
            --json "$OD/box_${n}_${d}_r$r.json" || fails=$((fails + 1))
      done
    done
  done
  unset GCWD GPYP
  if [ $fails = 0 ] && python3 "$S/det_docs.py" "$W" "$OD" "$B8/pr29_body.md" > "$L/det_docs.log" 2>&1; then
    cat "$L/det_docs.log"; mark DETCOST10 PASS "$(grep -E '^\| (G1|coupled)' "$L/det_docs.log" | cut -d'|' -f2,3,6 | tr '\n' ';')"
  else
    cat "$L/det_docs.log" 2>/dev/null; mark DETCOST10 FAIL "$fails failed runs (see $L)"
  fi
fi
cp -R "$L" "$OD/logs"; cp "$S/batch_10.sh" "$S/det_docs.py" "$STATUS" "$OD/" 2>/dev/null
git -C "$W" add scripts/diagnostics/det_cost_g1.py runs/mpm_main/batch_10_det
done_ DETCOST10 && git -C "$W" add docs/GAPS.md "research_notes/MPM in main simulator/deterministic_ccd_divergence.md"
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: cost of the deterministic mode (batch 10)

BatchSimOptions(deterministic=True) against the default on the G1
flat velocity task and the coupled box scene at 64 and 1,024 worlds,
one process per run, three interleaved repetitions (timing class,
16-inch); the numbers in the research note and GAPS.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
done_ DETCOST10 && ( cd "$W" && gh pr edit 29 --body-file "$B8/pr29_body.md" ) > "$L/pr29.log" 2>&1
say "=== batch_10 end $(date)"; cat "$STATUS"
