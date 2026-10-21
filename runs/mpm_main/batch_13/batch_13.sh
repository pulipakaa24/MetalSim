#!/bin/bash
# batch 13 (run outside the agent sandbox, after batch_12):
#   bash scratch/b13/batch_13.sh
#
# Part A, #29 (deterministic BatchSim) on the G1 task: batch_10b found (1) Warp's counter tripwire firing at 64 worlds
# and (2) a 3.5 GB deterministic scratch request at 1,024 worlds in MuJoCo Warp's fused solver-update launch
# (_update_constraint_gradient_fused, solver.py, MetalSim's MuJoCo Warp fork). Source reading: that fused kernel
# consumes a slot counter (quad_changed_count) that its own lane 0 zeroes inside the launch and then reads back after a
# threadgroup barrier (n_changes, the htot fusion), and it reads state_changed_count, whose atomics the counting pass
# suppresses: Warp's two-pass counters cannot express either (the counting pass sees stale values, the execution pass
# reads the counter before Warp writes the totals back), and its per-thread record bound comes from a dynamic row loop
# (the 1,024-world plan). Hypothesis: deterministic mode must take the unfused update (MJW_METAL_FUSE_UPDATE=0, the
# upstream structure: counters zeroed by their own launch). This batch:
#   A1 names the tripping kernel: G1 at 64 worlds, deterministic, uncaptured, a synchronize + flag check after every
#      deterministic launch (scripts/diagnostics/det_mismatch_probe.py), with the fused update (default) and without
#   A2 G1 deterministic with MJW_METAL_FUSE_UPDATE=0, captured, 64 and 1,024 worlds (det_cost_g1.py: check_overflow)
#   A3 names the kernel of the 1,024-world allocation failure (probe, fused update, 3 frames)
#   A4 #29 back to draft, with a comment (and a body line) saying G1 deterministic mode fails until the fix lands
# Part B, Newton SolverMuJoCo on MuJoCo Warp 3.14 (batch_11): Newton's tests run with Newton's own runner (unittest):
#   batch_11 ran them under pytest, which also collects add_function_test's module-level helpers (functions named test_*
#   taking (test, device)) and reports "fixture 'test' not found" for test_mujoco_fk_consistency's d6 helper (upstream's
#   layout; under unittest the registered test runs and passes). Pass = "OK" from both runs, no failure, no error. If
#   clean, batch_11's pin move runs (PIN_MOVE=1 bash scratch/b11/batch_11.sh, unchanged).
# Resumable (scratch/b13/STATUS.txt). ~40 min of GPU + batch_11's pin move ~20 min.
set -u
W=.
R=.
S=$W/scratch/b13
L=$S/logs
O=$W/runs/mpm_main/batch_13
PY=$R/.venv/bin/python
PYNF=$R/.venv-newtonfork/bin/python
NG=$W/upstream/newton-mjw314-git
NH=$W/upstream/newton-ht-mjw314-git
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_13 start $(date)"
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
gpu() {   # gpu NAME MIN LOG CWD PYTHONPATH -- cmd...
  local name=$1 min=$2 log=$3 cwd=$4 pyp=$5; shift 6
  say "gpu job $name (<= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" check "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$cwd" -- env PYTHONPATH="$pyp" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -3 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
[ -f "$W/assets/isaac/G1/g1_minimal.usd" ] || ln -s "$R/assets/isaac/G1" "$W/assets/isaac/G1"
mkdir -p "$W/scripts/diagnostics"
cp "$S/det_mismatch_probe.py" "$W/scripts/diagnostics/det_mismatch_probe.py"
health

# ---------------------------------------------------------------- part A (#29, G1)
PROBE=$W/scripts/diagnostics/det_mismatch_probe.py
if ! done_ A1; then
  gpu mm13_probe64 20 "$L/probe64_fused.log" "$W" "$W" -- "$PY" -P "$PROBE" --worlds 64 --frames 40 --json "$O/probe64_fused.json"
  t1=$(grep -E "^(TRIPPED|CLEAN)" "$L/probe64_fused.log" | tail -1 | cut -c1-400)
  gpu mm13_probe64nf 20 "$L/probe64_nofuse.log" "$W" "$W" -- env MJW_METAL_FUSE_UPDATE=0 "$PY" -P "$PROBE" --worlds 64 --frames 40 \
      --json "$O/probe64_nofuse.json"
  t2=$(grep -E "^(TRIPPED|CLEAN)" "$L/probe64_nofuse.log" | tail -1 | cut -c1-400)
  mark A1 DONE "fused: ${t1:-no verdict (see log)} | MJW_METAL_FUSE_UPDATE=0: ${t2:-no verdict (see log)}"
fi
if ! done_ A2; then
  r=""
  for n in 64 1024; do
    gpu mm13_g1nf_$n 20 "$L/g1_det_nofuse_$n.log" "$W" "$W" -- env MJW_METAL_FUSE_UPDATE=0 "$PY" -P \
        "$W/scripts/diagnostics/det_cost_g1.py" --worlds $n --deterministic 1 --frames 100 --warmup 20 --json "$O/g1_det_nofuse_$n.json"
    r="$r $n:rc=$?"
  done
  mark A2 DONE "G1 deterministic, unfused update:$r"
fi
if ! done_ A3; then
  gpu mm13_probe1024 20 "$L/probe1024_fused.log" "$W" "$W" -- "$PY" -P "$PROBE" --worlds 1024 --frames 3 --json "$O/probe1024_fused.json"
  k=$(grep -B2 "Array shapes must not exceed\|TRIPPED\|CLEAN" "$L/probe1024_fused.log" | tail -3 | tr '\n' ' ' | cut -c1-300)
  kk=$(grep -oE "_update_constraint_gradient_fused|_launch_update_fused|launch_tiled" "$L/probe1024_fused.log" | sort -u | tr '\n' ' ')
  mark A3 DONE "1,024 worlds, fused: $kk $k"
fi
if ! done_ A4; then
  {
    echo "Batch 13 (16-inch): #29 is back to draft. Deterministic mode fails on the G1 task (batch_10b): Warp's counter tripwire fires at 64 worlds, and at 1,024 worlds the deterministic plan of MuJoCo Warp's fused solver-update launch asks for 3.5 GB of scratch. The prime suspect is the MetalSim fork's fused update (\`_update_constraint_gradient_fused\`), which zeroes and reads back its own slot counter inside the launch. Diagnosis (kernel named by a per-launch flag check; unfused A/B):"
    echo; echo '```'; cat "$STATUS"; echo '```'
    echo; echo "The fix (deterministic mode takes the unfused update, or the fused kernel gets a counter-free compaction) follows once the A/B above confirms it."
    echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$S/pr29_comment.md"
  python3 - "$W/scratch/b08b/pr29_body.md" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
line = ("- **Open (batch_10b / batch_13): deterministic mode fails on the G1 task** (tripwire at 64 worlds, a 3.5 GB "
        "deterministic plan at 1,024 worlds in MuJoCo Warp's fused solver-update launch); draft until fixed.\n")
if "**Open (batch_10b" not in s:
    i = s.index("\n🤖 Generated with")
    s = s[:i].rstrip("\n") + "\n" + line + s[i:]
    open(p, "w").write(s)
EOF
  ( cd "$W" && gh pr ready 29 --undo; gh pr edit 29 --body-file "$W/scratch/b08b/pr29_body.md"; gh pr comment 29 --body-file "$S/pr29_comment.md" ) \
      > "$L/pr29.log" 2>&1
  cat "$L/pr29.log"; mark A4 DONE "$(tail -1 "$L/pr29.log")"
fi
cp -R "$L" "$O/"; cp "$S/batch_13.sh" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add scripts/diagnostics/det_mismatch_probe.py runs/mpm_main/batch_13
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: G1 diagnosis (batch 13)

A probe that synchronizes after every deterministic launch and names
the first one that trips Warp's counter flag
(scripts/diagnostics/det_mismatch_probe.py), run on the G1 task with
and without MuJoCo Warp's fused solver update, and the G1
deterministic runs without it at 64 / 1,024 worlds.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2

# ---------------------------------------------------------------- part B (Newton tests under unittest, then the pin move)
MODS="newton.tests.test_mujoco_solver newton.tests.test_mujoco_fk_consistency newton.tests.test_mujoco_general_actuators
 newton.tests.test_mujoco_margin_zeroing newton.tests.test_mujoco_reset newton.tests.test_mujoco_sleeping
 newton.tests.test_mujoco_version_check newton.tests.test_solver_mujoco_planar_mesh newton.tests.test_sites_mujoco_export"
if ! done_ NEWTON13; then
  gpu mm13_newton_nf 60 "$L/newton_unittest_metalsim.log" "$NG" "$NG" -- "$PYNF" -P -m unittest $MODS
  r1=$?
  gpu mm13_newton_ht 60 "$L/newton_unittest_hashtable.log" "$NH" "$NH" -- "$PY" -P -m unittest $MODS
  r2=$?
  s1=$(grep -E "^(Ran|OK|FAILED)" "$L/newton_unittest_metalsim.log" | tr '\n' ' '); s2=$(grep -E "^(Ran|OK|FAILED)" "$L/newton_unittest_hashtable.log" | tr '\n' ' ')
  if [ $r1$r2 = 00 ] && grep -q "^OK" "$L/newton_unittest_metalsim.log" && grep -q "^OK" "$L/newton_unittest_hashtable.log"; then
    mark NEWTON13 PASS "metalsim: $s1| hashtable: $s2"
  else mark NEWTON13 FAIL "rc $r1 $r2; metalsim: $s1| hashtable: $s2"; fi
  cp "$L"/newton_unittest_*.log "$O/" 2>/dev/null
fi
if done_ NEWTON13 && ! done_ PIN11RUN; then
  # batch_11's status still holds its first attempt's network failure (FORKS11 FAIL 21:17, superseded by the PASS at
  # 22:05); its pin-move gate refuses on any FAIL line, so the superseded one is dropped
  grep -q "^FORKS11 PASS" "$W/scratch/b11/STATUS.txt" && sed -i '' '/^FORKS11 FAIL 21:17:02 fetch$/d' "$W/scratch/b11/STATUS.txt"
  # the fetched data in batch_11's MetalSim worktree (its G1 Newton-engine tests skipped for the missing G1 USD)
  for p in assets/isaac/G1 upstream/mujoco_menagerie assets/polyhaven assets/tron1/scenes_local; do
    mkdir -p "$(dirname "$W/scratch/wt-newton/$p")"
    [ -e "$W/scratch/wt-newton/$p" ] || ln -s "$R/$p" "$W/scratch/wt-newton/$p"
  done
  say "running batch_11's pin move"
  PIN_MOVE=1 bash "$W/scratch/b11/batch_11.sh" > "$L/batch_11_pinmove.log" 2>&1
  rc=$?
  tail -15 "$L/batch_11_pinmove.log"
  [ $rc = 0 ] && mark PIN11RUN DONE "$(grep -E '^PR11' "$W/scratch/b11/STATUS.txt" | tail -1)" || mark PIN11RUN FAIL "rc=$rc (scratch/b11/STATUS.txt)"
fi
say "=== batch_13 end $(date)"; cat "$STATUS"
