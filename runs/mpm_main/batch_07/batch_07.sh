#!/bin/bash
# batch 07 (run outside the agent sandbox):  bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b07/batch_07.sh
#
# Coupling-layer optimisation (items 1 + 2 of research_notes/MPM in main simulator/coupling_opt.md):
#   0  wait for batch_06 to finish (it shares the GPU queue and the repository)
#   1  BRANCH   new worktree scratch/wt-copt on branch coupling-opt from origin/main; apply scratch/coupling_opt/
#               coupling_opt.diff (MPMCoupling.launch_step, capturable reset, BatchSim.launch_reset, scale_coupled
#               --split / --rigid-only fix, new test) + probes/policy_step_ab.py + the research note; a second worktree
#               scratch/wt-main (origin/main, detached) for the bitwise tree comparison
#   2  GPU checks (16-inch): bitwise main vs branch (hash_trajectory.py, 8 worlds, resets); tests/test_mpm_main.py
#   3  GPU timing (16-inch, timing class): scale_coupled --split and --rigid-only --time at 1 / 64 / 256 / 1,024
#      worlds; policy_step_ab at 1 / 64 / 1,024 worlds (2 % of the worlds reset per step)
#   4  commit code + note + logs on coupling-opt, push, DRAFT PR (numbers go into the docs next round)
# Resumable (scratch/b07/STATUS.txt). Expected ~60-90 min after batch_06.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b07
C=$W/scratch/coupling_opt
L=$S/logs
WT=$W/scratch/wt-copt
WM=$W/scratch/wt-main
PY=$R/.venv/bin/python
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_07 start $(date)"

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
gpu() {   # gpu NAME KIND MIN LOG CWD -- cmd...
  local name=$1 kind=$2 min=$3 log=$4 cwd=$5; shift 6
  say "gpu job $name ($kind, <= $min min, cwd $cwd) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --timeout $((min * 60)) --cwd "$cwd" -- env PYTHONPATH="$cwd" "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -4 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}

# ---------------------------------------------------------------- 0 wait for batch_06
for i in $(seq 1 240); do
  grep -q "=== batch_06 end" "$W/scratch/b06/logs/batch.log" 2>/dev/null && break
  [ $i = 1 ] && say "waiting for batch_06 to finish"
  sleep 60
done
grep -q "=== batch_06 end" "$W/scratch/b06/logs/batch.log" || { say "batch_06 not finished after 4 h"; exit 1; }

# ---------------------------------------------------------------- 1 worktrees
git -C "$W" fetch -q origin || exit 1
if ! done_ BRANCH; then
  [ -d "$WT" ] || git -C "$W" worktree add -q -B coupling-opt "$WT" origin/main || { mark BRANCH FAIL "worktree add"; exit 1; }
  git -C "$WT" apply "$C/coupling_opt.diff" || { mark BRANCH FAIL "coupling_opt.diff does not apply to origin/main ($(git -C "$W" rev-parse --short origin/main))"; exit 1; }
  cp "$C/src/policy_step_ab.py" "$WT/metalsim/physics/mpm/probes/policy_step_ab.py"
  cp "$C/NOTE.md" "$WT/research_notes/MPM in main simulator/coupling_opt.md"
  [ -d "$WM" ] || git -C "$W" worktree add -q --detach "$WM" origin/main || { mark BRANCH FAIL "main worktree"; exit 1; }
  mark BRANCH PASS "coupling-opt on $(git -C "$W" rev-parse --short origin/main)"
fi
O=$WT/runs/mpm_main/batch_07
mkdir -p "$O"
P=$WT/metalsim/physics/mpm/probes
health

# ---------------------------------------------------------------- 2 checks
if ! done_ BITWISE_TREES; then
  gpu mm7_hash_main check 15 "$L/hash_main.log" "$WM" -- "$PY" -P "$C/src/hash_trajectory.py" --root "$WM" --worlds 8 \
      --steps 60 --json "$O/hash_main.json"
  gpu mm7_hash_copt check 15 "$L/hash_copt.log" "$WT" -- "$PY" -P "$C/src/hash_trajectory.py" --root "$WT" --worlds 8 \
      --steps 60 --json "$O/hash_copt.json"
  "$PY" -c "
import json, sys
a, b = (json.load(open(p))['hashes'] for p in ('$O/hash_main.json', '$O/hash_copt.json'))
d = next((i for i in range(len(a)) if a[i] != b[i]), None)
print('first differing step:', d, 'of', len(a)); sys.exit(0 if d is None and len(a) == len(b) == 60 else 1)" > "$L/hash_cmp.log" 2>&1
  [ $? = 0 ] && mark BITWISE_TREES PASS "60 steps with resets, main == coupling-opt" || mark BITWISE_TREES FAIL "$(cat "$L/hash_cmp.log")"
fi
if ! done_ TESTS_MPM7; then
  gpu mm7_tests check 75 "$L/test_mpm_main.log" "$WT" -- "$PY" -P -m pytest -q tests/test_mpm_main.py
  rc=$?
  [ $rc = 0 ] && mark TESTS_MPM7 PASS "$(tail -1 "$L/test_mpm_main.log")" || mark TESTS_MPM7 FAIL "rc=$rc $(tail -1 "$L/test_mpm_main.log")"
fi

# ---------------------------------------------------------------- 3 timing (16-inch only)
if ! done_ TIMING7; then
  for n in 1 64 256 1024; do
    steps=100; [ $n = 1024 ] && steps=40
    gpu mm7_split_$n timing 20 "$L/split_$n.log" "$WT" -- "$PY" -P "$P/scale_coupled.py" --worlds $n --split \
        --steps $steps --warmup 10 --json "$O/split_$n.json"
    gpu mm7_rigid_$n timing 15 "$L/rigid_$n.log" "$WT" -- "$PY" -P "$P/scale_coupled.py" --worlds $n --rigid-only --time \
        --steps $steps --warmup 10 --json "$O/rigid_$n.json"
  done
  for n in 1 64 1024; do
    steps=200; [ $n = 1024 ] && steps=60
    gpu mm7_policy_$n timing 30 "$L/policy_$n.log" "$WT" -- "$PY" -P "$P/policy_step_ab.py" --worlds $n --fraction 0.02 \
        --steps $steps --warmup 10 --rounds 3 --hash-steps 40 --json "$O/policy_$n.json"
  done
  "$PY" - "$O" > "$L/timing_summary.log" 2>&1 <<'EOF'
import json, pathlib, sys
o = pathlib.Path(sys.argv[1])
for n in (1, 64, 256, 1024):
    try:
        s, r = json.load(open(o / f"split_{n}.json")), json.load(open(o / f"rigid_{n}.json"))
        print(f"{n:5d} worlds: full {s['full_ms_median']:.3f} ms = rigid {s['rigid_ms_median']:.3f} + coupling "
              f"{s['coupling_ms_median']:.3f} ({100 * s['coupling_share_of_full']:.2f} %) + mpm {s['mpm_ms_median']:.3f}; "
              f"rigid-only scene {r['ms_per_step_median']:.3f} ms")
    except Exception as e:
        print(n, "split/rigid missing:", e)
for n in (1, 64, 1024):
    try:
        p = json.load(open(o / f"policy_{n}.json"))
        print(f"{n:5d} worlds policy step: A {p['A_ms_median']:.3f} ms, B {p['B_ms_median']:.3f} ms, saving "
              f"{p['saving_ms_per_policy_step']:.3f} ms ({p['speedup']:.3f}x), bitwise {p['bitwise_equal']}")
    except Exception as e:
        print(n, "policy missing:", e)
EOF
  cat "$L/timing_summary.log"
  mark TIMING7 DONE "$(tr '\n' ';' < "$L/timing_summary.log" | cut -c1-600)"
fi

# ---------------------------------------------------------------- 4 commit, push, draft PR
cp -R "$L" "$O/"
cp "$S/batch_07.sh" "$STATUS" "$O/" 2>/dev/null
cp "$C/src/hash_trajectory.py" "$O/"
git -C "$WT" add metalsim/physics/mpm/coupled.py metalsim/physics/batch.py metalsim/physics/mpm/probes/scale_coupled.py \
    metalsim/physics/mpm/probes/policy_step_ab.py tests/test_mpm_main.py \
    "research_notes/MPM in main simulator/coupling_opt.md" runs/mpm_main/batch_07
git -C "$WT" diff --cached --quiet || git -C "$WT" commit -q -s -F - <<'EOF'
MPMCoupling: launch_step and a capturable per-world reset

The coupled step and per-world resets can now be captured into one
policy-step graph: MPMCoupling.launch_step(), launch_reset() reading a
persistent device mask (optionally BatchSim's, copied on the device),
reset(mask) taking a torch mask without a host sync and replaying a
reset graph, BatchSim.launch_reset(). The coupled step is split into
its sections (same launches, same order). scale_coupled.py: --split
per-section timings, --rigid-only --time fixed. probes/policy_step_ab.py:
eager vs one graph per policy step, bitwise and timed. Research note
(items 3-4 not done: at most ~1 % of the step). Batch 07 logs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$WT" push -q -u origin coupling-opt 2>&1 | grep -v "^remote:" | tail -2
( cd "$WT" && { gh pr view coupling-opt --json state -q .state 2>/dev/null | grep -q OPEN || \
    gh pr create --draft --base main --head coupling-opt \
    --title "MPMCoupling: launch_step and capturable per-world reset (draft: numbers to docs)" --body-file "$C/pr_body.md"; } ) \
    > "$L/pr.log" 2>&1
cat "$L/pr.log"; mark PR7 DONE "$(tail -1 "$L/pr.log")"
say "=== batch_07 end $(date)"; cat "$STATUS"
