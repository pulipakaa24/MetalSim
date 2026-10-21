#!/bin/bash
# batch 08 (run outside the agent sandbox), in two runs:
#   bash scratch/b08b/batch_08b.sh             # phases A-C
#   PIN_MOVE=1 bash scratch/b08b/batch_08b.sh  # + phase D
#
# batch_08b replaces batch_08 (stopped at FORKS8 on 2026-10-13 19:57: the regenerated fallback diffs differed from the
# committed ones only in the abbreviated blob ids of their "index" lines, 8 hex digits here against 7 in the files;
# every diff is now made with --abbrev=7). The current pins are read from origin/main at run time.
# Pins for the deterministic BatchSim (PR #29). Nothing on this Mac's shared checkouts or the forks' `metalsim`
# branches changes before phase D, which runs only with PIN_MOVE=1 and only if phase C found no regression.
#   A  FORKS8   Warp: branch metalsim-pin-tripwire = fork metalsim (31cf79d7, coupled-speed's pin) + the tripwire commit
#               (cherry-pick of c26bd28a) + a METALSIM_CHANGES.md row; MuJoCo Warp: metalsim-det-ccd (on f1b35e2) + a
#               METALSIM_CHANGES.md row (no AI co-author there). Both pushed as feature branches. Whole-branch diffs
#               regenerated (method checked against the current pins' committed diffs).
#   B  DOCS8    batchsim-deterministic: merge origin/main (checked clean), pins in setup_warp.sh / README / GUIDE /
#               STATUS / MPM.md, patches/*/metalsim-<new>.diff, GAPS / CHANGELOG / research note (scratch/b08b/pin_docs.py)
#   C  tests with the new pins' code first on PYTHONPATH (the forks' clones in upstream/, this Mac's shared checkouts
#      untouched): Warp deterministic unit tests; Newton's determinism tests (.venv-newton152, .venv-newtonfork);
#      .venv-newton152 suites (test_mpm, Franka-Pour); .venv: deterministic, rigid, render and remaining suites.
#      A failing test file is re-run on the current pins: passing there = REGRESSION (phase D refused).
#   D  (PIN_MOVE=1) forks' metalsim fast-forwarded to the new pins; this Mac's shared checkouts synced through the
#      GPU queue (scripts/fork_sync.sh, rebuilds libwarp: a native header changed); smoke tests on the shared
#      checkouts; push; PR #29 out of draft.
# Resumable (scratch/b08b/STATUS.txt). Phase C ~4 h, phase D ~30 min.
set -u
W=.
R=.
S=$W/scratch/b08b
L=$S/logs
O=$W/runs/mpm_main/batch_08b
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
PYNF=$R/.venv-newtonfork/bin/python
WARPG=$W/upstream/warp-tripwire-git
MJW=$W/upstream/mujoco_warp-detccd
TRIPOLD=c26bd28ae591acb53a09bae7e2f38aa178cbeaa5
CCD=cfa290b8831dfaf6f2e71ba0c5cd6d3f0c42c97d
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_08b start $(date) PIN_MOVE=${PIN_MOVE:-0}"
git -C "$W" fetch -q origin || exit 1
# the current pins, from origin/main
OLDW=$(git -C "$W" show origin/main:scripts/setup_warp.sh | sed -n 's/^PIN=\([0-9a-f]\{40\}\).*/\1/p')
OLDM=$(git -C "$W" show origin/main:README.md | grep -o 'mujoco_warp@[0-9a-f]\{40\}' | head -1 | cut -d@ -f2)
[ ${#OLDW} = 40 ] && [ ${#OLDM} = 40 ] || { echo "cannot read the current pins from origin/main ($OLDW / $OLDM)"; exit 1; }
echo "current pins (origin/main $(git -C "$W" rev-parse --short origin/main)): Warp $OLDW, MuJoCo Warp $OLDM"

say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE|PREEXISTING|REGRESSION)" "$STATUS"; }
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
gpu() {   # gpu NAME MIN LOG PYTHONPATH -- cmd...   (check class, cwd = the det worktree)
  local name=$1 min=$2 log=$3 pyp=$4; shift 5
  say "gpu job $name (<= $min min, PYTHONPATH=$pyp) -> $log"
  "$R/scripts/gpu_run.sh" "$name" check "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$pyp" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -3 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
commit_w() {   # commit_w SUBJECT BODY: commit what is staged in the det worktree
  git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
$1

$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}

[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "worktree not on batchsim-deterministic"; exit 1; }

# ---------------------------------------------------------------- A forks
if ! done_ FORKS8; then
  git -C "$WARPG" fetch -q fork || { mark FORKS8 FAIL "warp fetch"; exit 1; }
  fw=$(git -C "$WARPG" rev-parse fork/metalsim)
  [ "$fw" = "$OLDW" ] || { mark FORKS8 FAIL "Warp fork metalsim is $fw, not $OLDW: re-plan"; exit 1; }
  if git -C "$WARPG" rev-parse -q --verify metalsim-pin-tripwire > /dev/null && \
     ! git -C "$WARPG" merge-base --is-ancestor "$OLDW" metalsim-pin-tripwire; then
    mark FORKS8 FAIL "branch metalsim-pin-tripwire is not on the current Warp pin $OLDW (pushed earlier): re-plan"; exit 1
  fi
  if ! git -C "$WARPG" log --format=%s -1 metalsim-pin-tripwire 2>/dev/null | grep -q "METALSIM_CHANGES.md: counter phase-mismatch"; then
    git -C "$WARPG" checkout -q -B metalsim-pin-tripwire "$OLDW" || { mark FORKS8 FAIL "warp checkout"; exit 1; }
    git -C "$WARPG" cherry-pick "$TRIPOLD" > "$L/cherry_pick.log" 2>&1 || { git -C "$WARPG" cherry-pick --abort; mark FORKS8 FAIL "cherry-pick (see $L/cherry_pick.log)"; exit 1; }
    python3 "$S/add_row.py" warp "$WARPG/METALSIM_CHANGES.md" "$(git -C "$WARPG" rev-parse --short=8 HEAD)" || { mark FORKS8 FAIL "warp row"; exit 1; }
    git -C "$WARPG" add METALSIM_CHANGES.md
    git -C "$WARPG" commit -q -s -F - <<'EOF'
METALSIM_CHANGES.md: counter phase-mismatch tripwire

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  for f in warp/_src/metal_deterministic.py warp/native/metal_deterministic.h warp/tests/test_metal_deterministic.py; do
    cmp -s "$WARPG/$f" "$W/upstream/warp-tripwire/$f" || { mark FORKS8 FAIL "$f differs from the tested copy"; exit 1; }
  done
  mkdir -p "$WARPG/warp/bin" && cp "$R"/upstream/warp-innate/warp/bin/*.dylib "$WARPG/warp/bin/"
  git -C "$WARPG" push -q -u fork metalsim-pin-tripwire 2>&1 | grep -v "^remote:" | tail -2
  git -C "$MJW" fetch -q fork || { mark FORKS8 FAIL "mjw fetch"; exit 1; }
  fm=$(git -C "$MJW" rev-parse fork/metalsim)
  git -C "$MJW" merge-base --is-ancestor "$fm" metalsim-det-ccd || { mark FORKS8 FAIL "MuJoCo Warp fork metalsim $fm is not an ancestor of metalsim-det-ccd: re-plan"; exit 1; }
  git -C "$MJW" checkout -q metalsim-det-ccd
  if ! git -C "$MJW" log --format=%s -1 | grep -q "METALSIM_CHANGES.md: deterministic CCD"; then
    python3 "$S/add_row.py" mjw "$MJW/METALSIM_CHANGES.md" "${CCD:0:7}" || { mark FORKS8 FAIL "mjw row"; exit 1; }
    git -C "$MJW" add METALSIM_CHANGES.md
    git -C "$MJW" commit -q -s -m "METALSIM_CHANGES.md: deterministic CCD split"   # no AI co-author (MuJoCo Warp CLA rule)
  fi
  git -C "$MJW" push -q fork metalsim-det-ccd 2>&1 | grep -v "^remote:" | tail -2
  NEWW=$(git -C "$WARPG" rev-parse HEAD); NEWM=$(git -C "$MJW" rev-parse HEAD); TRIPN=$(git -C "$WARPG" rev-parse HEAD~1)
  git -C "$WARPG" ls-remote fork refs/heads/metalsim-pin-tripwire | grep -q "$NEWW" || { mark FORKS8 FAIL "warp push"; exit 1; }
  git -C "$MJW" ls-remote fork refs/heads/metalsim-det-ccd | grep -q "$NEWM" || { mark FORKS8 FAIL "mjw push"; exit 1; }
  printf "NEWW=%s\nNEWM=%s\nTRIPN=%s\nCCD=%s\nOLDW=%s\nOLDM=%s\n" "$NEWW" "$NEWM" "$TRIPN" "$CCD" "$OLDW" "$OLDM" > "$O/pins.txt"
  # whole-branch diff method: regenerating the current pins' diffs must give the committed files
  git -C "$WARPG" diff --abbrev=7 ce15f6bb2545e2d6c4a4c1fac8e40bb800db7265 "$OLDW" > "$S/check_old_warp.diff"
  git -C "$W" show "origin/main:patches/warp/metalsim-${OLDW:0:8}.diff" > "$S/main_old_warp.diff"
  git -C "$MJW" diff --abbrev=7 88af9cc "$OLDM" > "$S/check_old_mjw.diff"
  git -C "$W" show "origin/main:patches/mujoco_warp/metalsim-${OLDM:0:7}.diff" > "$S/main_old_mjw.diff"
  cmp -s "$S/check_old_warp.diff" "$S/main_old_warp.diff" && cmp -s "$S/check_old_mjw.diff" "$S/main_old_mjw.diff" \
      || { diff "$S/check_old_warp.diff" "$S/main_old_warp.diff" | head -20; diff "$S/check_old_mjw.diff" "$S/main_old_mjw.diff" | head -20
           mark FORKS8 FAIL "whole-branch diff method does not reproduce the committed diffs"; exit 1; }
  git -C "$WARPG" diff --abbrev=7 ce15f6bb2545e2d6c4a4c1fac8e40bb800db7265 "$NEWW" > "$S/new_warp.diff"
  git -C "$MJW" diff --abbrev=7 88af9cc "$NEWM" > "$S/new_mjw.diff"
  mark FORKS8 PASS "Warp ${NEWW:0:8} (code ${TRIPN:0:8}), MuJoCo Warp ${NEWM:0:7} (code ${CCD:0:7})"
fi
. "$O/pins.txt"

# ---------------------------------------------------------------- B merge main, pins and docs
if ! done_ DOCS8; then
  git -C "$W" fetch -q origin || exit 1
  if ! git -C "$W" merge-base --is-ancestor origin/main HEAD; then
    git -C "$W" merge -q --no-edit -m "Merge origin/main into batchsim-deterministic" origin/main \
        || { git -C "$W" merge --abort; mark DOCS8 FAIL "merge with origin/main conflicts"; exit 1; }
  fi
  git -C "$W" rm -q "patches/warp/metalsim-${OLDW:0:8}.diff" "patches/mujoco_warp/metalsim-${OLDM:0:7}.diff"
  cp "$S/new_warp.diff" "$W/patches/warp/metalsim-${NEWW:0:8}.diff"
  cp "$S/new_mjw.diff" "$W/patches/mujoco_warp/metalsim-${NEWM:0:7}.diff"
  python3 "$S/pin_docs.py" "$W" "$OLDW" "$NEWW" "$OLDM" "$NEWM" "$TRIPN" "$CCD" > "$L/pin_docs.log" 2>&1 \
      || { cat "$L/pin_docs.log"; mark DOCS8 FAIL "pin_docs (see $L/pin_docs.log)"; exit 1; }
  cat "$L/pin_docs.log"
  git -C "$W" add "patches/warp/metalsim-${NEWW:0:8}.diff" "patches/mujoco_warp/metalsim-${NEWM:0:7}.diff" \
      scripts/setup_warp.sh README.md docs/GUIDE.md docs/STATUS.md docs/GAPS.md CHANGELOG.md metalsim/physics/mpm/MPM.md \
      "research_notes/MPM in main simulator/deterministic_ccd_divergence.md"
  commit_w "Pin Warp ${NEWW:0:8} and MuJoCo Warp ${NEWM:0:7}: deterministic BatchSim" \
"MuJoCo Warp ${OLDM:0:7} -> ${NEWM:0:7} (fork branch metalsim-det-ccd):
deterministic-mode CCD as a counter-free solve kernel plus a
contact-write kernel. Warp ${OLDW:0:8} -> ${NEWW:0:8} (fork branch
metalsim-pin-tripwire, on coupled-speed's ${OLDW:0:8}): a slot-counter
call that differs between the counting and the execution pass raises.
Whole-branch fallback diffs regenerated; README, GUIDE, STATUS, MPM.md,
GAPS, CHANGELOG and the research note updated (batch_06 results)."
  mark DOCS8 PASS "$(git -C "$W" rev-parse --short HEAD)"
fi

# ---------------------------------------------------------------- C tests with the new pins' code
NEWPY=$MJW:$WARPG:$W          # .venv: both forks' clones first
NEWPY_N=$WARPG:$W             # Newton venvs: the Warp clone only (their MuJoCo Warp stays theirs)
env PYTHONPATH="$NEWPY" "$PY" -P -c "import warp, mujoco_warp, metalsim; print(warp.__file__); print(mujoco_warp.__file__); print(metalsim.__file__)" > "$L/imports_new.log" 2>&1
cat "$L/imports_new.log"
grep -q warp-tripwire-git "$L/imports_new.log" && grep -q mujoco_warp-detccd "$L/imports_new.log" && grep -q "$W/metalsim" "$L/imports_new.log" \
    || { mark IMPORTS8 FAIL "clones not first on PYTHONPATH"; exit 1; }
RIGID="tests/test_physics.py tests/test_capacity.py tests/test_contact_tuning.py tests/test_plane_convex_contacts.py
 tests/test_solver_presets.py tests/test_terrain.py tests/test_g1_task_terms.py tests/test_g1_fast_factorization.py
 tests/test_g1_parity.py tests/test_lift_env.py tests/test_lift_ctrl_ordering.py tests/test_sensors_contact.py
 tests/test_sensors_state.py tests/test_interop.py tests/test_warp_policy.py tests/test_ppo_warp_rollout.py
 tests/test_tron1.py tests/test_deformable.py"
RENDER=$(cd "$W" && ls tests/test_render_*.py tests/test_sensors_rt.py tests/test_lidar_*.py tests/test_camera_*.py \
    tests/test_area_lights.py tests/test_radar_lite.py tests/test_replicator*.py tests/test_scene_usd.py tests/test_materialx.py \
    tests/test_sensor_transparency.py tests/test_metal_conv.py 2>/dev/null | tr '\n' ' ')
N152="tests/test_mpm.py tests/test_franka_pour_terms.py tests/test_franka_pour_resume.py"
RIGID=$(echo $RIGID); ALL_LISTED=" $RIGID $RENDER $N152 tests/test_batch_deterministic.py "
REST=$(cd "$W" && for f in tests/test_*.py; do case "$ALL_LISTED" in (*" $f "*) ;; (*) printf "%s " "$f";; esac; done)
echo "rest: $REST"
health
run_suite() {   # run_suite NAME MIN PYEXE PYTHONPATH files...
  local name=$1 min=$2 exe=$3 pyp=$4; shift 4
  done_ "$name" && return 0
  gpu "mm8_$name" "$min" "$L/$name.log" "$pyp" -- "$exe" -P -m pytest -q -rfE "$@"
  local rc=$?
  local summary; summary=$(grep -E "passed|failed|error" "$L/$name.log" | tail -1)
  if [ $rc = 0 ] || [ $rc = 5 ]; then mark "$name" PASS "$summary"; return 0; fi
  # classify: re-run the failing files on the current pins
  local files; files=$(grep -E "^(FAILED|ERROR) " "$L/$name.log" | sed -E 's/^(FAILED|ERROR) ([^:]+).*/\2/' | sort -u | tr '\n' ' ')
  say "$name failed ($summary); re-running on the current pins: $files"
  local base=$W   # current pins: this Mac's shared checkouts (only the branch's metalsim on PYTHONPATH)
  gpu "mm8_${name}_old" "$min" "$L/${name}_oldpins.log" "$base" -- "$exe" -P -m pytest -q -rfE $files
  local new_f old_f
  new_f=$(grep -E "^(FAILED|ERROR) " "$L/$name.log" | sed -E 's/ - .*//' | sort -u)
  old_f=$(grep -E "^(FAILED|ERROR) " "$L/${name}_oldpins.log" | sed -E 's/ - .*//' | sort -u)
  local reg; reg=$(comm -23 <(echo "$new_f") <(echo "$old_f") | tr '\n' ';')
  if [ -n "$reg" ]; then mark "$name" REGRESSION "new-pin-only failures: $reg"
  else mark "$name" PREEXISTING "$summary; same failures on the current pins: $(echo "$new_f" | tr '\n' ';')"; fi
}
if ! done_ WDET8; then
  gpu mm8_warp_det 30 "$L/warp_det.log" "$NEWPY_N" -- "$PY" -m unittest discover -s "$WARPG/warp/tests/deterministic" -t "$WARPG"
  r1=$?
  gpu mm8_warp_metal_det 15 "$L/warp_metal_det.log" "$NEWPY_N" -- "$PY" -m unittest warp.tests.test_metal_deterministic
  r2=$?
  [ $r1$r2 = 00 ] && mark WDET8 PASS "$(grep -h '^Ran' "$L/warp_det.log" "$L/warp_metal_det.log" | tr '\n' ' ')" \
                   || mark WDET8 FAIL "rc $r1 $r2 (see $L/warp_det.log, $L/warp_metal_det.log)"
fi
for v in 152 nf; do
  name=newton_det_$v; exe=$PY152; [ $v = nf ] && exe=$PYNF
  done_ "$name" && continue
  [ -x "$exe" ] || { mark "$name" DONE "no $exe"; continue; }
  nd=$(env PYTHONPATH="$NEWPY_N" "$exe" -P -c "import newton, os; print(os.path.dirname(os.path.dirname(newton.__file__)))" 2>/dev/null | tail -1)
  gpu "mm8_$name" 45 "$L/$name.log" "$NEWPY_N" -- "$exe" -P -m pytest -q -rfE -p no:cacheprovider --rootdir "$nd" -c /dev/null \
      "$nd/newton/tests/determinism/test_solver_determinism.py"
  rc=$?
  s=$(grep -E "passed|failed|error|skipped" "$L/$name.log" | tail -1)
  if [ $rc = 0 ]; then mark "$name" PASS "$s"
  else
    gpu "mm8_${name}_old" 45 "$L/${name}_oldpins.log" "" -- "$exe" -P -m pytest -q -rfE -p no:cacheprovider --rootdir "$nd" -c /dev/null \
        "$nd/newton/tests/determinism/test_solver_determinism.py"
    [ $? = 0 ] && mark "$name" REGRESSION "$s (passes on the current Warp)" || mark "$name" PREEXISTING "$s; fails on the current Warp too: $(grep -E 'passed|failed' "$L/${name}_oldpins.log" | tail -1)"
  fi
done
run_suite det8 30 "$PY" "$NEWPY" tests/test_batch_deterministic.py
run_suite n152_8 90 "$PY152" "$NEWPY_N" $N152
run_suite rigid8 120 "$PY" "$NEWPY" $RIGID
run_suite render8 120 "$PY" "$NEWPY" $RENDER
run_suite rest8 150 "$PY" "$NEWPY" $REST
cp -R "$L" "$O/"; cp "$S/batch_08b.sh" "$S/pin_docs.py" "$S/add_row.py" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add runs/mpm_main/batch_08b
commit_w "Deterministic BatchSim: batch 08b tests on the new pins" "Phase C of batch_08b (STATUS.txt): Warp and Newton determinism tests, .venv-newton152 and .venv suites with the new pins' code."
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
if grep -qE " (REGRESSION|FAIL) " "$STATUS"; then
  say "phase C found a regression or failure: no pin move"; cat "$STATUS"; exit 2
fi
mark PHASEC DONE "no regression"
[ "${PIN_MOVE:-0}" = 1 ] || { say "=== batch_08b phase C end $(date); run again with PIN_MOVE=1 for phase D"; cat "$STATUS"; exit 0; }

# ---------------------------------------------------------------- D pin move on the forks and this Mac
if ! done_ FORKMOVE8; then
  git -C "$WARPG" push -q fork "$NEWW:refs/heads/metalsim" 2>&1 | grep -v "^remote:" | tail -2
  git -C "$MJW" push -q fork "$NEWM:refs/heads/metalsim" 2>&1 | grep -v "^remote:" | tail -2
  git -C "$WARPG" ls-remote fork refs/heads/metalsim | grep -q "$NEWW" && git -C "$MJW" ls-remote fork refs/heads/metalsim | grep -q "$NEWM" \
      && mark FORKMOVE8 PASS "fork metalsim: Warp ${NEWW:0:8}, MuJoCo Warp ${NEWM:0:7}" \
      || { mark FORKMOVE8 FAIL "fast-forward of a fork's metalsim refused"; exit 1; }
fi
if ! done_ SYNC8; then
  ( cd "$R" && scripts/fork_sync.sh ) > "$L/fork_sync.log" 2>&1
  cat "$L/fork_sync.log"
  hw=$(git -C "$R/upstream/warp-innate" rev-parse HEAD); hm=$(git -C "$R/upstream/mujoco_warp" rev-parse HEAD)
  [ "$hw" = "$NEWW" ] && [ "$hm" = "$NEWM" ] && mark SYNC8 PASS "shared checkouts: warp-innate ${hw:0:8}, mujoco_warp ${hm:0:7}" \
      || { mark SYNC8 FAIL "shared checkouts at ${hw:0:8} / ${hm:0:7}"; exit 1; }
fi
if ! done_ SMOKE8; then
  gpu mm8_smoke_det 30 "$L/smoke_det.log" "$W" -- "$PY" -P -m pytest -q -rfE tests/test_batch_deterministic.py
  r1=$?
  gpu mm8_smoke_warp 15 "$L/smoke_warp.log" "" -- "$PY" -m unittest warp.tests.test_metal_deterministic
  r2=$?
  [ $r1$r2 = 00 ] && mark SMOKE8 PASS "on the shared checkouts: $(grep -E 'passed' "$L/smoke_det.log" | tail -1); warp $(grep '^Ran' "$L/smoke_warp.log")" \
                   || { mark SMOKE8 FAIL "rc $r1 $r2"; exit 1; }
fi
cp -R "$L" "$O/"; cp "$STATUS" "$O/"
git -C "$W" add runs/mpm_main/batch_08b
commit_w "Deterministic BatchSim: pins moved (batch 08b phase D)" "Forks' metalsim fast-forwarded, this Mac's shared checkouts synced, smoke tests on them (STATUS.txt)."
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
{
  echo "Batch 08b (16-inch): pins moved to Warp \`${NEWW:0:8}\` (tripwire on coupled-speed's \`${OLDW:0:8}\`) and MuJoCo Warp \`${NEWM:0:7}\` (deterministic CCD split); main merged in."
  echo; echo '```'; cat "$STATUS"; echo '```'
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr comment 29 --body-file "$S/pr29_comment.md" && gh pr edit 29 --title "Deterministic BatchSim (opt-in): Warp RUN_TO_RUN for MuJoCo Warp, CCD split, counter tripwire" --body-file "$S/pr29_body.md" && gh pr ready 29 ) \
    > "$L/pr.log" 2>&1
cat "$L/pr.log"; mark PR8 DONE "$(tail -1 "$L/pr.log")"
say "=== batch_08b end $(date)"; cat "$STATUS"
