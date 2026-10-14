#!/bin/bash
# batch 06 (run outside the agent sandbox):  bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b06/batch_06.sh
#
# Deterministic BatchSim, box-box divergence fix (research_notes/MPM in main simulator/deterministic_ccd_divergence.md):
#   1  MJWFORK  MuJoCo Warp fork: clone the shared checkout (f1b35e2, fork/metalsim) into the worktree's upstream/
#               (gitignored), branch metalsim-det-ccd, apply patches/mujoco_warp/0017-deterministic-ccd-split.diff,
#               commit (no AI co-author: MuJoCo Warp's CLA rule), push to github.com/pulipakaa24/mujoco_warp.
#               The shared checkout upstream/mujoco_warp and .venv are NOT touched; jobs below put the clone first on
#               PYTHONPATH.
#   1b WARPFORK Warp fork: clone the shared checkout (95f6f121, fork/metalsim) the same way, branch
#               metalsim-det-tripwire, apply patches/warp/det-phase-mismatch-tripwire.diff (phase-0 / phase-1 counter
#               call check, Metal), commit, push to github.com/pulipakaa24/warp; warp/bin (the built dylibs, untracked)
#               is copied from the shared checkout (no native source changes: the header is compiled at run time).
#   2  GPU (16-inch, each job under gpu_run + watchdog, tiny first): Warp's Metal deterministic unit tests (incl. the
#      new tripwire tests); stack 4 worlds; stack 256 worlds in two cold caches + warm; contact-slot audit at 256
#      worlds; tests/test_batch_deterministic.py; tripwire A/B: stack 256 with the single CCD kernel
#      (MJW_CCD_DET_SPLIT=0) must now fail with a phase-mismatch error instead of diverging silently.
#   3  commit the BatchSim / probe / test / doc changes and the logs on batchsim-deterministic, push, comment on PR #29.
# Resumable (scratch/b06/STATUS.txt). Expected ~60-90 min.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b06
L=$S/logs
O=$W/runs/mpm_main/batch_06
PY=$R/.venv/bin/python
MJW=$W/upstream/mujoco_warp-detccd
PATCH=$W/patches/mujoco_warp/0017-deterministic-ccd-split.diff
BASE=f1b35e2b3ddc80b197b0c665bf6d71523d175daf
WARPG=$W/upstream/warp-tripwire-git
WPATCH=$W/patches/warp/det-phase-mismatch-tripwire.diff
WBASE=95f6f121b7e67025958335c7c15e338ac04a798a
export PYTHONPATH=$MJW:$WARPG:$W
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_06 start $(date)"

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
gpu() {
  local name=$1 kind=$2 min=$3 log=$4; shift 5
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$PYTHONPATH" "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -4 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
commit_msg() {   # commit_msg SUBJECT BODY  (commits what is staged in the worktree)
  git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
$1

$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}

[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "worktree not on batchsim-deterministic"; exit 1; }

# ---------------------------------------------------------------- 1 MuJoCo Warp fork branch
if ! done_ MJWFORK; then
  [ -d "$MJW/.git" ] || git clone -q "$R/upstream/mujoco_warp" "$MJW" || { mark MJWFORK FAIL "clone"; exit 1; }
  git -C "$MJW" remote get-url fork > /dev/null 2>&1 || git -C "$MJW" remote add fork https://github.com/pulipakaa24/mujoco_warp.git
  if ! git -C "$MJW" log --format=%s -1 metalsim-det-ccd 2>/dev/null | grep -q "deterministic"; then
    git -C "$MJW" checkout -q -B metalsim-det-ccd "$BASE" || { mark MJWFORK FAIL "checkout $BASE"; exit 1; }
    git -C "$MJW" apply "$PATCH" || { mark MJWFORK FAIL "patch does not apply to $BASE"; exit 1; }
    git -C "$MJW" add mujoco_warp/_src/collision_convex.py
    # MuJoCo Warp's AGENTS.md: no AI co-author on commits (CLA check)
    git -C "$MJW" commit -q -s -F - <<'EOF'
Repeatable CCD in deterministic mode: split solve and contact write

Under Warp's RUN_TO_RUN mode a kernel that consumes atomic_add return values runs twice: phase 0 with global array stores suppressed and every counter call returning 0, to count each thread's calls, then phase 1 with the slots. ccd_kernel picks its EPA / multicontact scratch row with such a counter (nccd) and then reserves contact slots in write_contact. In phase 0 every thread runs EPA on row 0 with its polytope stores dropped, reading stale memory, so the number of contacts it counts can differ from the number it writes in phase 1; a contact then lands in another world's slot (two stacked boxes on 256 worlds: one world with 0 box-box contacts, one with 2, different worlds per run).

In deterministic mode convex_narrowphase now runs non-heightfield CCD as a solve kernel without a slot counter (scratch row = thread id, launch width capped at naccdmax), which stages each collision's contacts, followed by a write kernel whose write_contact calls depend only on the staged count. The default path is unchanged. The heightfield kernel is not split; in deterministic mode a primitive against a heightfield raises (it takes GJK/EPA), while meshes take the plane path. MJW_CCD_DET_SPLIT=0 keeps the single kernel for diagnostics.
EOF
  fi
  cmp -s "$MJW/mujoco_warp/_src/collision_convex.py" "$W/upstream/mjw-detccd/mujoco_warp/_src/collision_convex.py" \
      || { mark MJWFORK FAIL "committed file differs from the tested copy"; exit 1; }
  git -C "$MJW" push -q -u fork metalsim-det-ccd 2>&1 | grep -v "^remote:" | tail -2
  sha=$(git -C "$MJW" rev-parse HEAD)
  git -C "$MJW" ls-remote fork refs/heads/metalsim-det-ccd | grep -q "$sha" || { mark MJWFORK FAIL "push"; exit 1; }
  mark MJWFORK PASS "$sha"
fi
git -C "$MJW" rev-parse HEAD > "$O/mujoco_warp_sha.txt"

# ---------------------------------------------------------------- 1b Warp fork branch (tripwire)
if ! done_ WARPFORK; then
  [ -d "$WARPG/.git" ] || git clone -q "$R/upstream/warp-innate" "$WARPG" || { mark WARPFORK FAIL "clone"; exit 1; }
  git -C "$WARPG" remote get-url fork > /dev/null 2>&1 || git -C "$WARPG" remote add fork https://github.com/pulipakaa24/warp.git
  if ! git -C "$WARPG" log --format=%s -1 metalsim-det-tripwire 2>/dev/null | grep -q "phase mismatch"; then
    git -C "$WARPG" checkout -q -B metalsim-det-tripwire "$WBASE" || { mark WARPFORK FAIL "checkout $WBASE"; exit 1; }
    git -C "$WARPG" apply "$WPATCH" || { mark WARPFORK FAIL "patch does not apply to $WBASE"; exit 1; }
    git -C "$WARPG" add warp/_src/metal_deterministic.py warp/native/metal_deterministic.h warp/tests/test_metal_deterministic.py
    git -C "$WARPG" commit -q -s -F - <<'EOF'
Flag counter phase mismatch in Metal deterministic mode

Consumed-return counters run a counting pass with global stores suppressed and then an execution pass that takes the scanned slots. The deterministic execution guide lists counter calls that depend on stores made earlier in the same launch as unsupported ("Side effects in the counting pass"), but nothing detected it: the slots silently went to the wrong calls. MuJoCo Warp's CCD kernel is such a kernel (EPA scratch picked by a counter), and in MetalSim it wrote contacts into other worlds' slots.

Phase 1 now checks each counter call against the record phase 0 made (same call number, destination and count), and the writeback compares each thread's call counts of the two phases. A mismatch sets bit 2 of the sticky deterministic flag; launches outside a capture raise a RuntimeError naming the kernel, as for an overflow. The CUDA path is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  for f in warp/_src/metal_deterministic.py warp/native/metal_deterministic.h warp/tests/test_metal_deterministic.py; do
    cmp -s "$WARPG/$f" "$W/upstream/warp-tripwire/$f" || { mark WARPFORK FAIL "committed $f differs from the tested copy"; exit 1; }
  done
  mkdir -p "$WARPG/warp/bin" && cp "$R"/upstream/warp-innate/warp/bin/*.dylib "$WARPG/warp/bin/" || { mark WARPFORK FAIL "bin"; exit 1; }
  git -C "$WARPG" push -q -u fork metalsim-det-tripwire 2>&1 | grep -v "^remote:" | tail -2
  wsha=$(git -C "$WARPG" rev-parse HEAD)
  git -C "$WARPG" ls-remote fork refs/heads/metalsim-det-tripwire | grep -q "$wsha" || { mark WARPFORK FAIL "push"; exit 1; }
  mark WARPFORK PASS "$wsha"
fi
git -C "$WARPG" rev-parse HEAD > "$O/warp_sha.txt"

# ---------------------------------------------------------------- 2 GPU validation
P=$W/metalsim/physics/mpm/probes
health
if ! done_ WARPTESTS; then
  gpu mm6_warp_det check 15 "$L/warp_test_metal_deterministic.log" -- "$PY" -m unittest -v warp.tests.test_metal_deterministic
  rc=$?
  "$PY" -c "import warp, mujoco_warp; print(warp.__file__); print(mujoco_warp.__file__)" > "$L/imports.log" 2>&1; cat "$L/imports.log"
  grep -q warp-tripwire-git "$L/imports.log" && grep -q mujoco_warp-detccd "$L/imports.log" || { mark WARPTESTS FAIL "fork clones not first on PYTHONPATH"; exit 1; }
  [ $rc = 0 ] && grep -q "test_phase_mismatch_tripwire" "$L/warp_test_metal_deterministic.log" \
      && mark WARPTESTS PASS "$(grep -E '^(Ran|OK)' "$L/warp_test_metal_deterministic.log" | tr '\n' ' ')" \
      || { mark WARPTESTS FAIL "rc=$rc (see $L/warp_test_metal_deterministic.log)"; exit 1; }
fi
if ! done_ TINY; then
  gpu mm6_tiny check 15 "$L/stack4.log" -- "$PY" "$P/rigid_det_diag.py" --scene stack --worlds 4 --steps 160 \
      --json "$O/stack4.json"
  rc=$?
  [ $rc = 0 ] && grep -q ccd_det_kernels_builder "$L/stack4.log" && mark TINY PASS || { mark TINY FAIL "rc=$rc (see $L/stack4.log)"; exit 1; }
fi
if ! done_ STACK256; then
  C=$S/cache; mkdir -p "$C"
  gpu mm6_s256a check 25 "$L/stack256_cold_a.log" -- env WARP_CACHE_PATH="$C/a" "$PY" "$P/rigid_det_diag.py" --scene stack \
      --worlds 256 --steps 200 --json "$O/stack256_cold_a.json"; ra=$?
  gpu mm6_s256b check 25 "$L/stack256_cold_b.log" -- env WARP_CACHE_PATH="$C/b" "$PY" "$P/rigid_det_diag.py" --scene stack \
      --worlds 256 --steps 200 --json "$O/stack256_cold_b.json"; rb=$?
  gpu mm6_s256w check 25 "$L/stack256_warm_b.log" -- env WARP_CACHE_PATH="$C/b" "$PY" "$P/rigid_det_diag.py" --scene stack \
      --worlds 256 --steps 200 --json "$O/stack256_warm_b.json"; rw=$?
  "$PY" "$P/rigid_det_diag.py" --compare "$O/stack256_cold_a.json" "$O/stack256_cold_b.json" > "$L/cmp_ab.log" 2>&1; cab=$?
  "$PY" "$P/rigid_det_diag.py" --compare "$O/stack256_cold_b.json" "$O/stack256_warm_b.json" > "$L/cmp_bw.log" 2>&1; cbw=$?
  if [ $ra$rb$rw$cab$cbw = 00000 ]; then mark STACK256 PASS "in-process and cross-process identical (cold a, cold b, warm b)"
  else mark STACK256 FAIL "rc a=$ra b=$rb warm=$rw cmp_ab=$cab cmp_bw=$cbw"; fi
fi
if ! done_ AUDIT6; then
  gpu mm6_audit check 25 "$L/contact_audit_ccd.log" -- "$PY" "$P/contact_audit.py" --worlds 256 --first 130 --last 160 \
      --json "$O/contact_audit_ccd.json"
  "$PY" -c "
import json, sys
d = json.load(open('$O/contact_audit_ccd.json'))
bad = [(s['step'], s['bb_count_hist'], s['n_worlds_state_differ']) for s in d['summary']
       if s['n_worlds_state_differ'] or any(len([c for c in h if c]) > 1 for h in s['bb_count_hist'])]
print('irregular steps:', bad[:10]); sys.exit(1 if bad else 0)" > "$L/audit_check.log" 2>&1
  [ $? = 0 ] && mark AUDIT6 PASS "every world the same box-box count at every step 130-160, no state difference" \
             || mark AUDIT6 FAIL "$(cat "$L/audit_check.log" | tail -1)"
fi
if ! done_ TESTS_DET6; then
  gpu mm6_tests check 60 "$L/test_batch_deterministic.log" -- "$PY" -m pytest -q tests/test_batch_deterministic.py
  rc=$?
  [ $rc = 0 ] && mark TESTS_DET6 PASS "$(tail -1 "$L/test_batch_deterministic.log")" \
              || mark TESTS_DET6 FAIL "rc=$rc $(tail -1 "$L/test_batch_deterministic.log")"
fi

if ! done_ TRIPWIRE_AB; then
  gpu mm6_trip check 25 "$L/stack256_single_ccd_kernel.log" -- env MJW_CCD_DET_SPLIT=0 "$PY" "$P/rigid_det_diag.py" \
      --scene stack --worlds 256 --steps 200 --json "$O/stack256_single_ccd_kernel.json"
  rc=$?
  if grep -q "phase mismatch" "$L/stack256_single_ccd_kernel.log"; then mark TRIPWIRE_AB PASS "single CCD kernel: phase mismatch raised (rc=$rc)"
  else mark TRIPWIRE_AB FAIL "single CCD kernel: no phase-mismatch error (rc=$rc)"; fi
fi

# ---------------------------------------------------------------- 3 commit, push, PR comment
cp -R "$L" "$O/"
cp "$S/batch_06.sh" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add metalsim/physics/batch.py tests/test_batch_deterministic.py metalsim/physics/mpm/probes/rigid_det_diag.py \
    metalsim/physics/mpm/probes/ccd_split_equiv.py patches/mujoco_warp/0017-deterministic-ccd-split.diff \
    patches/warp/det-phase-mismatch-tripwire.diff \
    "research_notes/MPM in main simulator/deterministic_ccd_divergence.md" docs/GAPS.md docs/GUIDE.md runs/mpm_main/batch_06
commit_msg "Deterministic BatchSim: root cause and fix of the box-box divergence" \
"The stacked-boxes divergence (2 of 256 worlds from step 139) is MuJoCo
Warp's CCD kernel under Warp's two-phase deterministic counters: phase 0
runs EPA on scratch row 0 with its stores suppressed, so it can count a
different number of contacts than phase 1 writes, and a contact lands in
another world's slot (batch_05 audit: one world 0 box-box contacts, one
2; native CCD off repeats). Fix on the MuJoCo Warp fork, branch
metalsim-det-ccd (patches/mujoco_warp/0017): in deterministic mode a
counter-free solve kernel stages the contacts and a write kernel writes
them. BatchSim requires that branch in deterministic mode, refuses
heightfield GJK/EPA contacts, and check_determinism now covers MuJoCo
Warp's module=\"unique\" kernels. Warp fork branch metalsim-det-tripwire
(patches/warp/det-phase-mismatch-tripwire.diff) flags a counter phase
mismatch; check_overflow reports it. Batch 06: GPU validation (STATUS)."
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
{
  echo "Batch 06 (16-inch): box-box divergence root-caused and fixed on the MuJoCo Warp fork, branch \`metalsim-det-ccd\` ($(cat "$O/mujoco_warp_sha.txt" | cut -c1-10)); see research_notes/MPM in main simulator/deterministic_ccd_divergence.md. Warp fork branch \`metalsim-det-tripwire\` ($(cut -c1-10 "$O/warp_sha.txt")) adds the phase-mismatch tripwire."
  echo; echo '```'; cat "$STATUS"; echo '```'
  echo; echo "Still a draft until the MuJoCo Warp pin (README, setup, patches/mujoco_warp) moves to that branch."
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr comment 29 --body-file "$S/pr29_comment.md" ) > "$L/pr_comment.log" 2>&1; cat "$L/pr_comment.log"
say "=== batch_06 end $(date)"; cat "$STATUS"
