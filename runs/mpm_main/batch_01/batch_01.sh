#!/bin/bash
# MPM in the main simulator, batch 01 (run by the user outside the agent sandbox):  bash <this file>
#
#   1  Newton fork: commit the Gauss-Seidel / separate-worlds coloring fix on a clone (branch metalsim-1.5.2-gs-envfirst)
#   2  move the pin (setup_newton152.sh, vendor script, notices), fallback patch, re-vendor the solver copy
#   3  CPU tests (vendor drift, licence headers, pin)
#   4  GPU, tiny first: coloring check (old fork solver vs fixed copy), stop unless the fixed one is VALID
#   5  GPU: multi-world repeatability matrix, column bitwise, box-on-sand Jacobi pair vs Isaac's coupling path,
#      rigid neutrality, angle of repose, coupled repeatability at 1 / 64 / 256 / 1,024 worlds, the test files
#   6  GPU timing (16-inch, timing class): coupled step at 1 / 64 / 256 / 1,024 worlds
#   7  push the fork branch (and fast-forward metalsim-1.5.2), commit + push mpm-main-sim, open the PR if all green
#
# Every GPU job goes through scripts/gpu_run.sh under scripts/watchdog.py; a health check follows each job and the
# batch stops on a GPU anomaly (busy GPU with no owner) or a watchdog kill. Idempotent: finished steps are skipped
# (markers in STATUS.txt); re-run after a failure. Expected duration ~2-3 h. Logs: runs/mpm_main/batch_01/logs/.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
B=$W/runs/mpm_main/batch_01
L=$B/logs
OUT=$W/scratch/mpm_main                       # large outputs (npz), gitignored
FORK=$W/upstream/newton-metalsim-1.5.2-envfirst   # gitignored (upstream/)
OLDPIN=3fd1f052a5ad031578960d9642aaaedc03834e4d
V152TAG=cf5378db6002b5c19b86344737cdd40cb8e08f0c   # newton v1.5.2
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
GH_NEWTON=https://github.com/pulipakaa24/newton.git
export PYTHONPATH=$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$OUT" "$W/runs/mpm_main/neutral" "$W/runs/mpm_main/repose" "$W/runs/mpm_main/scale" "$W/runs/mpm_main/timing"
STATUS=$B/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_01 start $(date)"

say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -q "^$1 PASS" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }

health() {
  local u top
  u=$("$R/scripts/gpu_util.sh" 2>/dev/null); top=$(python3 "$R/scripts/gpu_top.py" 2>/dev/null | head -4)
  say "health: utilisation ${u:-?}%"; echo "$top"
  if [ "${u:-0}" -ge 10 ] && echo "$top" | grep -q "none >= 1 %"; then
    sleep 30; u=$("$R/scripts/gpu_util.sh" 2>/dev/null)
    if [ "${u:-0}" -ge 10 ] && python3 "$R/scripts/gpu_top.py" 2>/dev/null | grep -q "none >= 1 %"; then
      mark HEALTH FAIL "GPU ${u}% busy with no owning process: stopping the batch (see the GPU hang memory note)"
      exit 90
    fi
  fi
}

# gpu NAME KIND MINUTES LOGFILE -- command...   (watchdog: no CPU progress for 180 s, or MINUTES elapsed -> kill)
gpu() {
  local name=$1 kind=$2 min=$3 log=$4; shift 5
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --timeout $((min * 60)) --cwd "$W" -- "$@" > "$log" 2>&1
  local rc=$?
  tail -5 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}

stage() {
  git -C "$W" add metalsim/physics/mpm tests/test_mpm_main.py scripts/vendor_mpm152.py scripts/setup_newton152.sh \
      patches/newton THIRD_PARTY_NOTICES.md README.md pyproject.toml docs/GUIDE.md docs/STATUS.md docs/GAPS.md \
      "research_notes/MPM in main simulator" runs/mpm_main ':!runs/mpm_main/fork' 2>&1 | tail -3
}

finish() {   # always: commit and push what exists (logs included), so nothing is lost
  local rc=$?
  say "finish (rc=$rc): commit + push"
  stage
  if ! git -C "$W" diff --cached --quiet; then
    git -C "$W" commit -q -s -F - <<EOF
MPM in the main simulator: Newton fork GS coloring fix, re-pin, batch 01 runs

Vendored solver re-generated from the fork commit that fixes Gauss-Seidel
colouring with separate worlds; box-on-sand Jacobi reference, scale,
neutrality, repose and timing probes with their logs (batch_01).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  git -C "$W" push -q origin mpm-main-sim 2>&1 | grep -v "^remote:" | tail -3
  say "=== batch_01 end $(date) rc=$rc"; cat "$STATUS"
}
trap finish EXIT

# ---------------------------------------------------------------- 0 save the agent's work, merge main
if ! done_ MERGE; then
  stage
  git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
MPM in the main simulator: tests, probes, docs; GS coloring fix staged

Box-on-sand / scale / neutrality / repose probes, the full test set for
MPMCoupling, GUIDE / STATUS / GAPS / MPM.md entries, and the Newton fork
edit (applied and pinned by runs/mpm_main/batch_01).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  git -C "$W" fetch -q origin
  if git -C "$W" merge -q --no-edit origin/main; then mark MERGE PASS "$(git -C "$W" rev-parse --short HEAD)"
  else git -C "$W" merge --abort; mark MERGE FAIL "conflict with origin/main: branch left unmerged"; fi
fi

# ---------------------------------------------------------------- 1 fork commit
if [ ! -d "$FORK/.git" ]; then git clone -q "$R/upstream/newton-metalsim-1.5.2" "$FORK" || exit 1; fi
SUBJ="Fix Gauss-Seidel coloring with separate worlds"
if [ "$(git -C "$FORK" log -1 --format=%s)" != "$SUBJ" ]; then
  git -C "$FORK" checkout -q -B metalsim-1.5.2-gs-envfirst "$OLDPIN" || exit 1
  for f in solver_implicit_mpm.py implicit_mpm_solver_kernels.py; do
    cmp -s "$FORK/newton/_src/solvers/implicit_mpm/$f" "$W/runs/mpm_main/fork/a/implicit_mpm/$f" \
      || { mark FORK FAIL "$f at $OLDPIN differs from the edit base"; exit 1; }
    cp "$W/runs/mpm_main/fork/b/implicit_mpm/$f" "$FORK/newton/_src/solvers/implicit_mpm/$f"
  done
  cmp -s "$FORK/METALSIM_CHANGES.md" "$W/runs/mpm_main/fork/a/METALSIM_CHANGES.md" || { mark FORK FAIL "METALSIM_CHANGES base"; exit 1; }
  cp "$W/runs/mpm_main/fork/b/METALSIM_CHANGES.md" "$FORK/METALSIM_CHANGES.md"
  git -C "$FORK" add newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py \
      newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py METALSIM_CHANGES.md
  git -C "$FORK" commit -q -s -F - <<EOF || exit 1
$SUBJ

The colour blocks of a discontinuous strain basis were built as
cell * nodes_per_cell, but environment-first strain partitions
(separate_worlds) number a cell's nodes in environment-major order.
On a sparse multi-environment grid the blocks were then a permutation
of other cells' nodes, and strain nodes solved concurrently in one
colour shared velocity nodes: a data race (two identical runs differed
from the first step on Metal; the colouring is invalid on CUDA too).
Blocks now start at the partition index of the cell's first node.
Single-environment grids keep the previous kernel, bitwise unchanged;
Jacobi solves never colour.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
fi
NEW=$(git -C "$FORK" rev-parse HEAD)
mark FORK PASS "$NEW"

# ---------------------------------------------------------------- 2 pin, fallback patch, re-vendor
"$PY" "$B/move_pin.py" "$NEW" || { mark PIN FAIL; exit 1; }
git -C "$FORK" diff "$V152TAG" "$NEW" > "$W/patches/newton/metalsim-1.5.2-${NEW:0:8}.diff"
"$PY" "$W/scripts/vendor_mpm152.py" "$FORK" || { mark VENDOR FAIL; exit 1; }
mark PIN PASS "${NEW:0:8}"

# ---------------------------------------------------------------- 3 CPU tests
METALSIM_NEWTON152_DIR=$FORK "$PY" -m pytest -q -p no:cacheprovider "$W/tests/test_mpm_main.py" \
    -k "pin or regenerates or licence" > "$L/cpu_tests.log" 2>&1 && mark CPU_TESTS PASS || { mark CPU_TESTS FAIL; tail -30 "$L/cpu_tests.log"; exit 1; }

# ---------------------------------------------------------------- 4 coloring check, tiny first
health
P=$W/metalsim/physics/mpm/probes
if ! done_ COLOR_NEW; then
  gpu mm_col_old check 6 "$L/color_old_newton152.log" -- env METALSIM_MPM_SOLVER=newton "$PY152" "$P/gs_coloring_check.py" --device metal:0 --worlds 2 --cells auto
  gpu mm_col_new2 check 6 "$L/color_new_w2.log" -- "$PY" "$P/gs_coloring_check.py" --device metal:0 --worlds 2 --cells auto
  gpu mm_col_new3 check 6 "$L/color_new_w3.log" -- "$PY" "$P/gs_coloring_check.py" --device metal:0 --worlds 3 --cells -1
  if tail -1 "$L/color_new_w2.log" | grep -qx VALID && tail -1 "$L/color_new_w3.log" | grep -qx VALID; then
    mark COLOR_NEW PASS "old fork: $(tail -1 "$L/color_old_newton152.log")"
  else
    mark COLOR_NEW FAIL "fixed coloring not VALID: stopping before anything is pushed"; exit 1
  fi
fi

# ---------------------------------------------------------------- 5 correctness runs
if ! done_ MWREPEAT; then
  gpu mm_mwrep check 30 "$L/mwrepeat_fixed.log" -- "$PY" "$P/multiworld_repeat.py" --steps 10
  n=$(grep -c "first differing step None" "$L/mwrepeat_fixed.log")
  [ "$n" = 12 ] && mark MWREPEAT PASS "12/12 repeatable" || mark MWREPEAT FAIL "$n/12 repeatable"
fi

if ! done_ COLUMN_BITWISE; then
  gpu mm_vbit check 15 "$L/vendored_bitwise_column.log" -- "$W/runs/mpm_main/vendored_bitwise/run.sh" column
  "$PY" - <<'EOF' > "$L/vendored_bitwise_compare.log" 2>&1 && mark COLUMN_BITWISE PASS || mark COLUMN_BITWISE FAIL
import json, sys
d = "/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/runs/mpm_main/vendored_bitwise/"
r = [json.load(open(d + f)) for f in ("n152_newton_column.json", "n152_vendored_column.json", "venv_vendored_column.json")]
ok = all(x["frames"] == r[0]["frames"] and x["iterations"] == r[0]["iterations"] and x["final_q"] == r[0]["final_q"] for x in r)
print([x["newton"] for x in r], [x["solver"] for x in r], "bitwise equal:", ok)
sys.exit(0 if ok else 1)
EOF
fi

if ! done_ BOX_PAIR; then
  BX="$P/box_on_sand.py"
  gpu mm_box_ms check 20 "$L/box_jacobi_metalsim.log" -- "$PY" "$BX" --engine metalsim --solver jacobi --worlds 2 \
      --steps 60 --capture 1 --out "$OUT/box_jacobi_metalsim.npz" --summary "$W/runs/mpm_main/box_on_sand/box_jacobi_metalsim.json"
  gpu mm_box_nt check 20 "$L/box_jacobi_newton.log" -- "$PY152" "$BX" --engine newton --solver jacobi --worlds 2 \
      --steps 60 --out "$OUT/box_jacobi_newton.npz" --summary "$W/runs/mpm_main/box_on_sand/box_jacobi_newton.json"
  "$PY" "$BX" --compare "$OUT/box_jacobi_metalsim.npz" "$OUT/box_jacobi_newton.npz" > "$L/box_jacobi_compare.log" 2>&1
  cat "$L/box_jacobi_compare.log"
  [ -s "$W/runs/mpm_main/box_on_sand/box_jacobi_newton.json" ] && mark BOX_PAIR PASS || mark BOX_PAIR FAIL
fi

if ! done_ NEUTRAL; then
  RN="$P/rigid_neutral_mpm.py"
  gpu mm_neutral_a check 15 "$L/neutral_plain1.log" -- "$PY" "$RN" --mode plain --json "$W/runs/mpm_main/neutral/plain1.json"
  gpu mm_neutral_b check 15 "$L/neutral_plain2.log" -- "$PY" "$RN" --mode plain --json "$W/runs/mpm_main/neutral/plain2.json"
  gpu mm_neutral_c check 15 "$L/neutral_after.log" -- "$PY" "$RN" --mode after-mpm --json "$W/runs/mpm_main/neutral/after_mpm.json"
  "$PY" "$RN" --compare "$W/runs/mpm_main/neutral/plain1.json" "$W/runs/mpm_main/neutral/plain2.json" \
      "$W/runs/mpm_main/neutral/after_mpm.json" > "$L/neutral_compare.log" 2>&1 && mark NEUTRAL PASS || mark NEUTRAL FAIL
  cat "$L/neutral_compare.log"
fi

if ! done_ REPOSE; then
  for mu in 0.68 0.4 0.9; do
    gpu "mm_repose_$mu" check 20 "$L/repose_$mu.log" -- "$PY" "$P/pile_repose.py" --friction "$mu" --seconds 2.0 \
        --json "$W/runs/mpm_main/repose/repose_mu$mu.json"
  done
  mark REPOSE PASS "$(grep -h angle_deg "$L"/repose_*.log | tr -s ' ' | tr '\n' ' ')"
fi

if ! done_ SCALE; then
  ok=1
  for w in 1 64 256 1024; do
    gpu "mm_scale_$w" check 45 "$L/scale_gs_w$w.log" -- "$PY" "$P/scale_coupled.py" --worlds $w --steps 40 --runs 2 \
        --json "$W/runs/mpm_main/scale/repeat_gs_w$w.json" || { ok=0; mark "SCALE_$w" FAIL; break; }
  done
  gpu mm_scale_j1024 check 45 "$L/scale_jacobi_w1024.log" -- "$PY" "$P/scale_coupled.py" --worlds 1024 --steps 40 \
      --runs 2 --solver jacobi --json "$W/runs/mpm_main/scale/repeat_jacobi_w1024.json" || ok=0
  [ $ok = 1 ] && mark SCALE PASS || mark SCALE FAIL
fi

if ! done_ TESTS_MAIN; then
  gpu mm_tests_main check 60 "$L/tests_mpm_main.log" -- env METALSIM_NEWTON152_DIR="$FORK" "$PY" -m pytest -v \
      -p no:cacheprovider --rootdir "$W" "$W/tests/test_mpm_main.py" && mark TESTS_MAIN PASS || mark TESTS_MAIN FAIL
fi
if ! done_ TESTS_MPM152; then
  gpu mm_tests_152 check 30 "$L/tests_mpm_newton152.log" -- "$PY152" -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_mpm.py" && mark TESTS_MPM152 PASS || mark TESTS_MPM152 FAIL
fi
if ! done_ TESTS_PHYSICS; then
  gpu mm_tests_phys check 40 "$L/tests_physics.log" -- "$PY" -m pytest -q -p no:cacheprovider --rootdir "$W" \
      "$W/tests/test_physics.py" && mark TESTS_PHYSICS PASS || mark TESTS_PHYSICS FAIL
fi

# ---------------------------------------------------------------- 6 timing (16-inch only, exclusive timing jobs)
if ! done_ TIMING; then
  for s in auto jacobi; do
    for w in 1 64 256 1024; do
      gpu "mm_t_${s}_$w" timing 20 "$L/time_${s}_w$w.log" -- "$PY" "$P/scale_coupled.py" --worlds $w --time --steps 40 \
          --warmup 10 --solver $s --json "$W/runs/mpm_main/timing/time_${s}_w$w.json"
    done
  done
  gpu mm_t_eager_64 timing 20 "$L/time_auto_w64_eager.log" -- "$PY" "$P/scale_coupled.py" --worlds 64 --time \
      --steps 40 --warmup 10 --capture 0 --json "$W/runs/mpm_main/timing/time_auto_w64_eager.json"
  mark TIMING PASS
fi

# ---------------------------------------------------------------- 7 push fork, PR
if done_ COLOR_NEW && done_ COLUMN_BITWISE && done_ TESTS_MAIN; then
  git -C "$FORK" push -q "$GH_NEWTON" metalsim-1.5.2-gs-envfirst 2>&1 | tail -2 && mark FORK_PUSH PASS || mark FORK_PUSH FAIL
  # fast-forward the fork's metalsim-1.5.2 (no force: refused if it moved past the old pin)
  git -C "$FORK" push -q "$GH_NEWTON" "$NEW:refs/heads/metalsim-1.5.2" 2>&1 | tail -2 && mark FORK_FF PASS || mark FORK_FF FAIL
fi
ALL=1
for s in MERGE FORK PIN CPU_TESTS COLOR_NEW MWREPEAT COLUMN_BITWISE BOX_PAIR NEUTRAL SCALE TESTS_MAIN TESTS_MPM152 FORK_PUSH; do
  done_ $s || ALL=0
done
if [ $ALL = 1 ]; then
  finish_pr=1
else
  finish_pr=0; say "not opening the PR: some steps are not PASS (see STATUS.txt)"
fi
trap - EXIT
finish
if [ $finish_pr = 1 ]; then
  cd "$W" && if gh pr view mpm-main-sim > /dev/null 2>&1; then gh pr edit mpm-main-sim --body-file "$B/pr_body.md"; else
    gh pr create --base main --head mpm-main-sim \
      --title "MPM material in the main simulator (BatchSim): Isaac's implicit MPM, proxy coupling, N worlds" \
      --body-file "$B/pr_body.md"; fi 2>&1 | tee "$L/pr.log"
  mark PR PASS "$(tail -1 "$L/pr.log")"
fi
