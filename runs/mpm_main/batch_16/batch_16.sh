#!/bin/bash
# batch 16 (run outside the agent sandbox):  bash scratch/b16/batch_16.sh
#
# A  POUR14's failure is not #46's: batch_14 ran the Franka-Pour tests in .venv-newton152 BEFORE its own repair of that
#    environment (FIX_N152 came after (a) in the batch), i.e. on stock Newton 1.5.2 (upstream/newton-1.5.2, traceback
#    paths), whose geometry/hashtable.py uses a 64-bit wp.atomic_cas that Apple GPUs lack; the metalsim-1.5.2 fork
#    replaced it with Metal protocols (hashtable.py "#if defined(__METAL_VERSION__)", MPM.md 8d). #46 moves the Newton
#    1.7 pins of .venv / .venv-newtonfork and does not touch .venv-newton152. Re-run of both Franka-Pour files on the
#    repaired environment; the result is posted on #46.
# B  #29, G1 deterministic under capture: batch_15 showed each of MetalSim's two capture-time optimisations is enough to
#    trip Warp's counter tripwire (no early exit alone: trips; unfused alone (batch_13 A2): trips; both off: passes at
#    64 worlds). Repeat of the four combinations at 64 worlds in one build, then both off at 1,024 worlds.
# C  cost of forcing both off in deterministic mode, timing class, one process per run, 3 interleaved repetitions:
#    coupled box (scale_coupled --time --deterministic 1) at 64 / 1,024 worlds, both on vs both off; G1 deterministic with
#    both off at 64 / 1,024 worlds (its default-mode numbers are batch_10b's g1_*_0 runs).
# Resumable (scratch/b16/STATUS.txt). ~1 h.
set -u
W=.
R=.
S=$W/scratch/b16
L=$S/logs
O=$W/runs/mpm_main/batch_16
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_16 start $(date)"
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
gpu() {   # gpu NAME KIND MIN LOG -- cmd...   (cwd and PYTHONPATH = the det worktree)
  local name=$1 kind=$2 min=$3 log=$4; shift 5
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$W" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -2 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
for p in assets/isaac/G1 upstream/mujoco_menagerie assets/polyhaven assets/tron1/scenes_local; do [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"; done
health

# ---------------------------------------------------------------- A Franka-Pour on the repaired .venv-newton152
if ! done_ POUR16; then
  "$PY152" -P -c "import newton, os; from newton._src.solvers.implicit_mpm import solve_rheology as sr; print('newton', os.path.dirname(newton.__file__), 'fork API', hasattr(sr, '_color_block_count'))" | tee "$L/n152_env.log"
  gpu mm16_pour check 60 "$L/franka_pour_tests.log" -- "$PY152" -P -m pytest -q -rfEs tests/test_franka_pour_terms.py tests/test_franka_pour_resume.py
  rc=$?; s=$(grep -E "passed|failed|error" "$L/franka_pour_tests.log" | tail -1)
  if [ $rc = 0 ] && ! grep -q "^SKIPPED" "$L/franka_pour_tests.log" && grep -q "newton-metalsim-1.5.2" "$L/n152_env.log"; then mark POUR16 PASS "$s"
  else mark POUR16 FAIL "rc=$rc $s"; fi
  {
    echo "Not blocking #46. batch_14's Franka-Pour failure (\`reduce_buffered_contacts_kernel ... atomic_cas(uint64)\`) came from \`.venv-newton152\`, which at that point imported stock Newton 1.5.2 (\`upstream/newton-1.5.2\`). Stock Newton's hash table uses a 64-bit \`atomic_cas\` that Apple GPUs lack; the metalsim-1.5.2 fork replaces it with Metal protocols. #46 moves only the Newton 1.7 pins (\`.venv\`, \`.venv-newtonfork\`). Batch 16 re-ran both Franka-Pour files after \`.venv-newton152\` was repaired (batch_14 FIX_N152):"
    echo; echo '```'; cat "$L/n152_env.log"; grep "^POUR16" "$STATUS"; echo '```'
    echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$S/pr46_comment.md"
  ( cd "$W" && gh pr comment 46 --body-file "$S/pr46_comment.md" ) > "$L/pr46.log" 2>&1
fi

# ---------------------------------------------------------------- B G1 deterministic: both causes
G1=$W/scripts/diagnostics/det_cost_g1.py
run_g1() {   # run_g1 TAG WORLDS ENV...
  local tag=$1 n=$2; shift 2
  gpu mm16_${tag}_$n check 25 "$L/g1_${tag}_$n.log" -- env "$@" "$PY" -P "$G1" --worlds $n --deterministic 1 --frames 40 --warmup 10 \
      --json "$O/g1_${tag}_$n.json"
  local rc=$? why
  why=$(grep -hoE "phase mismatch|Array shapes must not exceed[^,]*|counter overflow" "$L/g1_${tag}_$n.log" | head -1)
  echo "$tag/$n:rc=$rc${why:+($why)}"
}
if ! done_ G1CAUSES16; then
  r1=$(run_g1 both_on 64 X=1)
  r2=$(run_g1 no_early_exit 64 MJW_METAL_ICB_EARLY_EXIT=0)
  r3=$(run_g1 unfused 64 MJW_METAL_FUSE_UPDATE=0)
  r4=$(run_g1 both_off 64 MJW_METAL_ICB_EARLY_EXIT=0 MJW_METAL_FUSE_UPDATE=0)
  mark G1CAUSES16 DONE "64 worlds: $r1 $r2 $r3 $r4"
fi
if ! done_ G1BIG16; then
  r=$(run_g1 both_off 1024 MJW_METAL_ICB_EARLY_EXIT=0 MJW_METAL_FUSE_UPDATE=0)
  mark G1BIG16 DONE "$r"
fi

# ---------------------------------------------------------------- C cost of forcing both off (timing class)
if ! done_ COST16; then
  for rep in 1 2 3; do
    for n in 64 1024; do
      for v in "on:X=1" "off:MJW_METAL_ICB_EARLY_EXIT=0 MJW_METAL_FUSE_UPDATE=0"; do
        tag=${v%%:*}; envs=${v#*:}
        f="$O/box_${n}_${tag}_r$rep.json"
        [ -s "$f" ] || gpu mm16_box_${n}_${tag}_r$rep timing 20 "$L/box_${n}_${tag}_r$rep.log" -- env $envs "$PY" -P \
            "$W/metalsim/physics/mpm/probes/scale_coupled.py" --worlds $n --deterministic 1 --time --steps 40 --warmup 10 --json "$f"
      done
      f="$O/g1_${n}_off_r$rep.json"
      [ -s "$f" ] || gpu mm16_g1t_${n}_r$rep timing 20 "$L/g1t_${n}_r$rep.log" -- env MJW_METAL_ICB_EARLY_EXIT=0 MJW_METAL_FUSE_UPDATE=0 \
          "$PY" -P "$G1" --worlds $n --deterministic 1 --frames 100 --warmup 20 --json "$f"
    done
  done
  python3 - "$O" "$W/runs/mpm_main/batch_10_det" > "$L/cost_summary.log" 2>&1 <<'EOF'
import glob, json, statistics, sys
o, b10 = sys.argv[1], sys.argv[2]
def med(pattern, key):
    v = [json.load(open(p))[key] for p in sorted(glob.glob(pattern))]
    return (statistics.median(v), len(v)) if v else (None, 0)
for n in (64, 1024):
    on, k1 = med(f"{o}/box_{n}_on_r*.json", "ms_per_step_median")
    off, k2 = med(f"{o}/box_{n}_off_r*.json", "ms_per_step_median")
    if on and off:
        print(f"coupled box, deterministic, {n} worlds: early exit + fused on {on:.2f} ms, both off {off:.2f} ms ({100 * (off / on - 1):+.1f} %), n={k1}/{k2}")
    g0, k3 = med(f"{b10}/g1_{n}_0_r*.json", "mean_step_ms")
    goff, k4 = med(f"{o}/g1_{n}_off_r*.json", "mean_step_ms")
    if g0 and goff:
        print(f"G1 env step, {n} worlds: default mode {g0:.2f} ms (batch_10b), deterministic with both off {goff:.2f} ms ({100 * (goff / g0 - 1):+.1f} %), n={k3}/{k4}")
EOF
  cat "$L/cost_summary.log"
  mark COST16 DONE "$(tr '\n' ';' < "$L/cost_summary.log")"
fi
cp -R "$L" "$O/"; cp "$S/batch_16.sh" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add runs/mpm_main/batch_16
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: G1 causes and the cost of the fix options (batch 16)

Captured G1 deterministic runs with MuJoCo Warp's ICB early exit and
fused update in all four combinations (64 worlds), both off at 1,024
worlds, and the timing of both off on the coupled box and G1 scenes.
Franka-Pour tests re-run on the repaired .venv-newton152.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
say "=== batch_16 end $(date)"; cat "$STATUS"
