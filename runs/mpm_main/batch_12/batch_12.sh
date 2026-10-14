#!/bin/bash
# batch 12 (run outside the agent sandbox):  bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b12/batch_12.sh
#
# #29 shown green before merge. batch_08b's rigid suite had 20 failed + 3 errors ("PREEXISTING": the same on the old
# pins), all one cause: the worktree lacked the fetched, gitignored G1 USD (assets/isaac/G1/g1_minimal.usd; every
# failure is UsdStage::Open "Failed to open layer"), so test_solver_presets, test_terrain, test_ppo_warp_rollout and
# test_deformable's G1 cloth scenes could not build the G1. The worktree now links the main checkout's fetched data:
# assets/isaac/G1, upstream/mujoco_menagerie, assets/polyhaven, assets/tron1/scenes_local (symlinks, never committed).
# This re-runs every suite of batch_08b phase C on the pinned shared checkouts (phase D moved them) with -rs, so each
# skip prints its reason; logs + a skip-reason summary are committed on batchsim-deterministic and posted on #29.
# Pass = no failure and no error in any suite. ~20 min of GPU (check class).
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b12
L=$S/logs
O=$W/runs/mpm_main/batch_12
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_12 start $(date)"
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
gpu() {   # gpu NAME MIN LOG -- cmd...   (cwd and PYTHONPATH = the det worktree)
  local name=$1 min=$2 log=$3; shift 4
  say "gpu job $name (<= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" check "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$W" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -2 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
# fetched data from the main checkout (gitignored; the same files a fresh install fetches)
for p in assets/isaac/G1 upstream/mujoco_menagerie assets/polyhaven assets/tron1/scenes_local; do
  [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"
  ls "$W/$p/" > /dev/null 2>&1 || { mark DATA FAIL "$p missing in the main checkout too"; exit 1; }
done
[ -f "$W/assets/isaac/G1/g1_minimal.usd" ] || { mark DATA FAIL "no G1 USD"; exit 1; }
"$PY" -P -c "import warp, mujoco_warp, metalsim; print(warp.__file__, mujoco_warp.__file__, metalsim.__file__)" | tee "$L/imports.log"
health

RIGID="tests/test_physics.py tests/test_capacity.py tests/test_contact_tuning.py tests/test_plane_convex_contacts.py
 tests/test_solver_presets.py tests/test_terrain.py tests/test_g1_task_terms.py tests/test_g1_fast_factorization.py
 tests/test_g1_parity.py tests/test_lift_env.py tests/test_lift_ctrl_ordering.py tests/test_sensors_contact.py
 tests/test_sensors_state.py tests/test_interop.py tests/test_warp_policy.py tests/test_ppo_warp_rollout.py
 tests/test_tron1.py tests/test_deformable.py"
RENDER=$(ls tests/test_render_*.py tests/test_sensors_rt.py tests/test_lidar_*.py tests/test_camera_*.py tests/test_area_lights.py \
    tests/test_radar_lite.py tests/test_replicator*.py tests/test_scene_usd.py tests/test_materialx.py \
    tests/test_sensor_transparency.py tests/test_metal_conv.py 2>/dev/null | tr '\n' ' ')
N152="tests/test_mpm.py tests/test_franka_pour_terms.py tests/test_franka_pour_resume.py"
RIGID=$(echo $RIGID); ALL_LISTED=" $RIGID $RENDER $N152 tests/test_batch_deterministic.py "
REST=$(for f in tests/test_*.py; do case "$ALL_LISTED" in (*" $f "*) ;; (*) printf "%s " "$f";; esac; done)

suite() {   # suite NAME MIN PYEXE files...
  local name=$1 min=$2 exe=$3; shift 3
  done_ "$name" && return 0
  gpu "mm12_$name" "$min" "$L/$name.log" -- "$exe" -P -m pytest -q -rfEs "$@"
  local rc=$? s; s=$(grep -E "passed|failed|error" "$L/$name.log" | tail -1)
  if [ $rc = 0 ] || [ $rc = 5 ]; then mark "$name" PASS "$s"; else mark "$name" FAIL "rc=$rc $s"; fi
}
suite rigid12 60 "$PY" $RIGID
suite det12 20 "$PY" tests/test_batch_deterministic.py
suite render12 60 "$PY" $RENDER
suite rest12 60 "$PY" $REST
suite n152_12 60 "$PY152" $N152

# skip reasons, per suite
{
  for n in rigid12 det12 render12 rest12 n152_12; do
    echo "== $n: $(grep -E 'passed|failed|error' "$L/$n.log" | tail -1)"
    grep -E "^SKIPPED" "$L/$n.log" | sed -E 's/^SKIPPED \[[0-9]+\] [^ ]+: //' | sort | uniq -c | sort -rn
  done
} > "$L/skip_reasons.txt"
cat "$L/skip_reasons.txt"
cp -R "$L" "$O/"; cp "$S/batch_12.sh" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add runs/mpm_main/batch_12
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: every suite on the pinned checkouts (batch 12)

batch_08b's rigid suite failed 20 + 3 in this worktree only because
the fetched G1 USD (gitignored) was absent; with the main checkout's
fetched data linked, the rigid, deterministic, render, remaining and
.venv-newton152 suites re-run on the moved pins, with skip reasons.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
{
  echo "Batch 12 (16-inch): every suite of batch_08b phase C again, on the moved pins, with the fetched data (G1 USD, Menagerie, Poly Haven, TRON1 scenes) linked. batch_08b's 20 rigid failures + 3 errors were all \`UsdStage::Open\` on the missing, gitignored G1 USD in this worktree."
  echo; echo '```'; cat "$STATUS"; echo; cat "$L/skip_reasons.txt"; echo '```'
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr comment 29 --body-file "$S/pr29_comment.md" ) > "$L/pr.log" 2>&1; cat "$L/pr.log"
say "=== batch_12 end $(date)"; cat "$STATUS"
