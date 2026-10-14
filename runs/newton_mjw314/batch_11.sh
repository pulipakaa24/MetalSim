#!/bin/bash
# batch 11 (run outside the agent sandbox), in two runs:
#   bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b11/batch_11.sh               # A-B
#   PIN_MOVE=1 bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b11/batch_11.sh    # + C
#
# Newton 1.7's SolverMuJoCo against MetalSim's MuJoCo Warp fork 3.14: contact_force_fn takes contact_adhesion_in, which
# Newton's convert_mjw_contacts_to_newton_kernel did not pass (upstream Newton fixed it in 90f809a, its MuJoCo Warp
# 3.14 update), so no SolverMuJoCo kernel compiled and the Newton-engine / Newton-bridge tests skipped in .venv.
#   A  FORKS11  clone of upstream/newton; branches metalsim (9f317390) and metalsim-hashtable-metal (a4be3604) each get
#               the same commit (scratch/b11/edit_newton.py: the contact part of 90f809a; the sim extra to 3.14);
#               pushed as feature branches metalsim-mjw314 / metalsim-hashtable-metal-mjw314 (the forks' branches and
#               this Mac's environments untouched); the whole-branch fallback diffs regenerated (--full-index, method
#               checked against the committed ones)
#   B  GPU tests with the patched trees on PYTHONPATH: Newton's MuJoCo-solver tests (cpu + metal) for both branches;
#      MetalSim's Newton tests in .venv before (baseline: expected skips) and after (no "SolverMuJoCo unavailable")
#   C  (PIN_MOVE=1, only if B passed) forks' branches fast-forwarded; this Mac: upstream/newton fast-forwarded and
#      .venv's newton reinstalled at the new pin, both through the GPU queue; MetalSim worktree scratch/wt-newton
#      (branch newton-mjw314 from origin/main): pins, patches, README / MPM.md / CHANGELOG (scratch/b11/newton_docs.py);
#      MetalSim's Newton tests again on the real environments; commit, push, PR
# Resumable (scratch/b11/STATUS.txt). A+B ~45 min, C ~20 min.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b11
L=$S/logs
NG=$W/upstream/newton-mjw314-git
NH=$W/upstream/newton-ht-mjw314-git
WTN=$W/scratch/wt-newton
PY=$R/.venv/bin/python
PYNF=$R/.venv-newtonfork/bin/python
PIN_NF=9f317390da07432dfb5813bd0db6a47eeb7cb330
PIN_HT=a4be3604bc218bb366d12164fa3e1025f333bdae
BASE=45458023f24629c1feffb1f336594808f537db14
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_11 start $(date) PIN_MOVE=${PIN_MOVE:-0}"

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
  say "gpu job $name (<= $min min, cwd $cwd, PYTHONPATH=$pyp) -> $log"
  "$R/scripts/gpu_run.sh" "$name" check "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$cwd" -- env PYTHONPATH="$pyp" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -3 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
newton_commit() {   # newton_commit DIR: the fix as one commit
  python3 "$S/edit_newton.py" "$1" && python3 "$S/fork_pyproject.py" "$1" || return 1
  git -C "$1" add newton/_src/solvers/mujoco/kernels.py newton/_src/solvers/mujoco/solver_mujoco.py pyproject.toml
  git -C "$1" commit -q -s -F - <<'EOF'
Pass contact adhesion to MuJoCo Warp 3.14's contact_force_fn

MuJoCo Warp 3.14 added contact_adhesion_in to contact_force_fn, so
convert_mjw_contacts_to_newton_kernel no longer compiled against it,
and with it no SolverMuJoCo kernel (they share a module). This is the
contact part of upstream's MuJoCo Warp 3.14 update (90f809a,
newton-physics/newton#4392): the kernel takes mj_contact_adhesion and
passes it on, the launch passes mj_contact.adhesion. The sim extra asks
for MuJoCo / MuJoCo Warp 3.14 as upstream does.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}
TESTFILES="test_mujoco_solver.py test_mujoco_fk_consistency.py test_mujoco_general_actuators.py test_mujoco_margin_zeroing.py
 test_mujoco_reset.py test_mujoco_sleeping.py test_mujoco_version_check.py test_solver_mujoco_planar_mesh.py test_sites_mujoco_export.py"
MS_TESTS="tests/test_render_newton_bridge.py tests/test_render_newton_perworld.py tests/test_newton_bridge_poses.py tests/test_g1_newton_engine.py"

# ---------------------------------------------------------------- A fork commits
if ! done_ FORKS11; then
  [ -d "$NG/.git" ] || git clone -q "$R/upstream/newton" "$NG" || { mark FORKS11 FAIL "clone"; exit 1; }
  git -C "$NG" remote get-url fork > /dev/null 2>&1 || git -C "$NG" remote add fork https://github.com/pulipakaa24/newton.git
  git -C "$NG" fetch -q fork || { mark FORKS11 FAIL "fetch"; exit 1; }
  [ "$(git -C "$NG" rev-parse fork/metalsim)" = "$PIN_NF" ] && [ "$(git -C "$NG" rev-parse fork/metalsim-hashtable-metal)" = "$PIN_HT" ] \
      || { mark FORKS11 FAIL "fork branches moved (metalsim $(git -C "$NG" rev-parse --short fork/metalsim), hashtable $(git -C "$NG" rev-parse --short fork/metalsim-hashtable-metal)): re-plan"; exit 1; }
  if ! git -C "$NG" log --format=%s -1 metalsim-mjw314 2>/dev/null | grep -q "contact adhesion"; then
    git -C "$NG" checkout -q -B metalsim-mjw314 "$PIN_NF" && newton_commit "$NG" || { mark FORKS11 FAIL "metalsim commit"; exit 1; }
  fi
  [ -d "$NH/.git" ] || [ -f "$NH/.git" ] || git -C "$NG" worktree add -q "$NH" -B metalsim-hashtable-metal-mjw314 "$PIN_HT" \
      || { mark FORKS11 FAIL "hashtable worktree"; exit 1; }
  if ! git -C "$NH" log --format=%s -1 | grep -q "contact adhesion"; then
    git -C "$NH" checkout -q -B metalsim-hashtable-metal-mjw314 "$PIN_HT" && newton_commit "$NH" || { mark FORKS11 FAIL "hashtable commit"; exit 1; }
  fi
  git -C "$NG" checkout -q metalsim-mjw314
  # the tested copies (CPU runs in the sandbox) must equal what was committed
  for f in kernels.py solver_mujoco.py; do
    cmp -s "$NG/newton/_src/solvers/mujoco/$f" "$W/upstream/newton-mjw314/newton/_src/solvers/mujoco/$f" \
      && cmp -s "$NH/newton/_src/solvers/mujoco/$f" "$W/upstream/newton-ht-mjw314/newton/_src/solvers/mujoco/$f" \
      || { mark FORKS11 FAIL "$f differs from the CPU-tested copy"; exit 1; }
  done
  git -C "$NG" push -q -u fork metalsim-mjw314 metalsim-hashtable-metal-mjw314 2>&1 | grep -v "^remote:" | tail -2
  NEW_NF=$(git -C "$NG" rev-parse metalsim-mjw314); NEW_HT=$(git -C "$NG" rev-parse metalsim-hashtable-metal-mjw314)
  git -C "$NG" ls-remote fork refs/heads/metalsim-mjw314 | grep -q "$NEW_NF" && git -C "$NG" ls-remote fork refs/heads/metalsim-hashtable-metal-mjw314 | grep -q "$NEW_HT" \
      || { mark FORKS11 FAIL "push"; exit 1; }
  # whole-branch fallback diffs: the method must reproduce the committed ones
  git -C "$NG" diff --full-index "$BASE" "$PIN_NF" > "$S/check_nf.diff"
  git -C "$NG" diff --full-index "$BASE" "$PIN_HT" > "$S/check_ht.diff"
  git -C "$W" fetch -q origin
  git -C "$W" show "origin/main:patches/newton/metalsim-${PIN_NF:0:8}.diff" > "$S/main_nf.diff"
  git -C "$W" show "origin/main:patches/newton/metalsim-hashtable-metal-${PIN_HT:0:8}.diff" > "$S/main_ht.diff"
  cmp -s "$S/check_nf.diff" "$S/main_nf.diff" && cmp -s "$S/check_ht.diff" "$S/main_ht.diff" \
      || { mark FORKS11 FAIL "fallback diff method does not reproduce patches/newton"; exit 1; }
  git -C "$NG" diff --full-index "$BASE" "$NEW_NF" > "$S/new_nf.diff"
  git -C "$NG" diff --full-index "$BASE" "$NEW_HT" > "$S/new_ht.diff"
  printf "NEW_NF=%s\nNEW_HT=%s\n" "$NEW_NF" "$NEW_HT" > "$S/pins.txt"
  mark FORKS11 PASS "metalsim-mjw314 ${NEW_NF:0:8}, metalsim-hashtable-metal-mjw314 ${NEW_HT:0:8}"
fi
. "$S/pins.txt"

# ---------------------------------------------------------------- B tests with the patched trees
health
if ! done_ NEWTONTESTS11; then
  r=0
  gpu mm11_newton_nf 45 "$L/newton_mujoco_metalsim.log" "$NG" "$NG" -- "$PYNF" -P -m pytest -q -rfE -p no:cacheprovider \
      -c /dev/null --rootdir "$NG" $(for f in $TESTFILES; do printf "%s " "$NG/newton/tests/$f"; done) || r=1
  gpu mm11_newton_ht 45 "$L/newton_mujoco_hashtable.log" "$NH" "$NH" -- "$PY" -P -m pytest -q -rfE -p no:cacheprovider \
      -c /dev/null --rootdir "$NH" $(for f in $TESTFILES; do printf "%s " "$NH/newton/tests/$f"; done) || r=1
  # the only accepted error: pytest collecting test_mujoco_fk_consistency's add_function_test helper (fixture 'test')
  bad=$(grep -hE "^(FAILED|ERROR) " "$L/newton_mujoco_metalsim.log" "$L/newton_mujoco_hashtable.log" | grep -v "test_mujoco_fk_kernel_matches_mujoco_warp_d6_three_angular")
  if [ -z "$bad" ]; then mark NEWTONTESTS11 PASS "$(grep -h ' passed' "$L/newton_mujoco_metalsim.log" "$L/newton_mujoco_hashtable.log" | tr '\n' ';')"
  else mark NEWTONTESTS11 FAIL "$(echo "$bad" | head -5 | tr '\n' ';')"; fi
fi
if ! done_ MSTESTS11; then
  [ -d "$WTN" ] || git -C "$W" worktree add -q -B newton-mjw314 "$WTN" origin/main || { mark MSTESTS11 FAIL "worktree"; exit 1; }
  gpu mm11_ms_before 30 "$L/metalsim_newton_tests_before.log" "$WTN" "$WTN" -- "$PY" -P -m pytest -q -rfEs $MS_TESTS
  gpu mm11_ms_after 30 "$L/metalsim_newton_tests_after.log" "$WTN" "$NH:$WTN" -- "$PY" -P -m pytest -q -rfEs $MS_TESTS
  rc=$?
  if [ $rc = 0 ] && ! grep -q "SolverMuJoCo unavailable" "$L/metalsim_newton_tests_after.log"; then
    mark MSTESTS11 PASS "before: $(tail -1 "$L/metalsim_newton_tests_before.log"); after: $(tail -1 "$L/metalsim_newton_tests_after.log")"
  else mark MSTESTS11 FAIL "rc=$rc $(tail -1 "$L/metalsim_newton_tests_after.log")"; fi
fi
if grep -qE " FAIL " "$STATUS"; then say "a check failed: no pin move"; cat "$STATUS"; exit 2; fi
[ "${PIN_MOVE:-0}" = 1 ] || { say "=== batch_11 A-B end $(date); run again with PIN_MOVE=1 for the pin move"; cat "$STATUS"; exit 0; }

# ---------------------------------------------------------------- C pins
if ! done_ FORKMOVE11; then
  git -C "$NG" push -q fork "$NEW_NF:refs/heads/metalsim" "$NEW_HT:refs/heads/metalsim-hashtable-metal" 2>&1 | grep -v "^remote:" | tail -2
  git -C "$NG" ls-remote fork refs/heads/metalsim | grep -q "$NEW_NF" && git -C "$NG" ls-remote fork refs/heads/metalsim-hashtable-metal | grep -q "$NEW_HT" \
      && mark FORKMOVE11 PASS || { mark FORKMOVE11 FAIL "fast-forward refused"; exit 1; }
fi
if ! done_ SYNC11; then
  cat > "$S/sync_newton.sh" <<EOF
#!/bin/bash
set -ex
cd $R
git -C upstream/newton remote get-url origin | grep -q pulipakaa24/newton
git -C upstream/newton fetch -q origin metalsim
git -C upstream/newton merge -q --ff-only $NEW_NF
$PY -m pip install -q --no-deps --force-reinstall "newton @ git+https://github.com/pulipakaa24/newton@$NEW_HT"
$PYNF -P -c "import newton; print('newtonfork', newton.__file__)"
$PY -P -c "import newton; print('venv', newton.__file__)"
EOF
  "$R/scripts/gpu_run.sh" sync_newton11 check 15 -- bash "$S/sync_newton.sh" > "$L/sync_newton.log" 2>&1
  cat "$L/sync_newton.log"
  [ "$(git -C "$R/upstream/newton" rev-parse HEAD)" = "$NEW_NF" ] && grep -q "\"commit_id\":\"$NEW_HT\"" "$R"/.venv/lib/python3.12/site-packages/newton-*.dist-info/direct_url.json \
      && mark SYNC11 PASS || { mark SYNC11 FAIL "see $L/sync_newton.log"; exit 1; }
fi
if ! done_ DOCS11; then
  git -C "$WTN" rm -q "patches/newton/metalsim-${PIN_NF:0:8}.diff" "patches/newton/metalsim-hashtable-metal-${PIN_HT:0:8}.diff"
  cp "$S/new_nf.diff" "$WTN/patches/newton/metalsim-${NEW_NF:0:8}.diff"
  cp "$S/new_ht.diff" "$WTN/patches/newton/metalsim-hashtable-metal-${NEW_HT:0:8}.diff"
  python3 "$S/newton_docs.py" "$WTN" "$PIN_NF" "$NEW_NF" "$PIN_HT" "$NEW_HT" > "$L/docs.log" 2>&1 && mark DOCS11 PASS \
      || { cat "$L/docs.log"; mark DOCS11 FAIL "newton_docs"; exit 1; }
  cat "$L/docs.log"
fi
if ! done_ MSFINAL11; then
  gpu mm11_ms_final 30 "$L/metalsim_newton_tests_final.log" "$WTN" "$WTN" -- "$PY" -P -m pytest -q -rfEs $MS_TESTS
  rc=$?
  [ $rc = 0 ] && ! grep -q "SolverMuJoCo unavailable" "$L/metalsim_newton_tests_final.log" \
      && mark MSFINAL11 PASS "$(tail -1 "$L/metalsim_newton_tests_final.log")" || { mark MSFINAL11 FAIL "rc=$rc"; exit 1; }
fi
O=$WTN/runs/newton_mjw314
mkdir -p "$O"
cp -R "$L" "$O/"; cp -R "$S/cpu_logs" "$O/"; cp "$S/batch_11.sh" "$S/edit_newton.py" "$S/fork_pyproject.py" "$S/newton_docs.py" "$STATUS" "$O/" 2>/dev/null
git -C "$WTN" add -A patches/newton pyproject.toml scripts/setup_newton.sh README.md metalsim/physics/mpm/MPM.md CHANGELOG.md runs/newton_mjw314
git -C "$WTN" diff --cached --quiet || git -C "$WTN" commit -q -s -F - <<EOF
Pin Newton 1.7 ${NEW_NF:0:8} / ${NEW_HT:0:8}: SolverMuJoCo on MuJoCo Warp 3.14

Both 1.7 fork branches pass MuJoCo Warp 3.14's contact adhesion to
contact_force_fn (the contact part of upstream Newton's 90f809a);
before, no SolverMuJoCo kernel compiled against MetalSim's MuJoCo Warp
fork and the Newton-engine / Newton-bridge tests skipped in .venv.
metalsim ${PIN_NF:0:8} -> ${NEW_NF:0:8} (.venv-newtonfork),
metalsim-hashtable-metal ${PIN_HT:0:8} -> ${NEW_HT:0:8} (.venv);
fallback diffs regenerated. Batch 11: Newton's MuJoCo-solver tests and
MetalSim's Newton tests (STATUS.txt).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$WTN" push -q -u origin newton-mjw314 2>&1 | grep -v "^remote:" | tail -2
( cd "$WTN" && { gh pr view newton-mjw314 --json state -q .state 2>/dev/null | grep -q OPEN || \
    gh pr create --base main --head newton-mjw314 --title "Newton 1.7: SolverMuJoCo on MuJoCo Warp 3.14 (pins)" --body-file "$S/pr_body.md"; } ) \
    > "$L/pr.log" 2>&1
cat "$L/pr.log"; mark PR11 DONE "$(tail -1 "$L/pr.log")"
say "=== batch_11 end $(date)"; cat "$STATUS"
