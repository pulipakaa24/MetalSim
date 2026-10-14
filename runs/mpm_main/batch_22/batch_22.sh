#!/bin/bash
# batch 22 (run outside the agent sandbox):  bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b22/batch_22.sh
#
# Two follow-ups of #29, on branch mpm-simd-gs-removal (from main dad44063 = 11dc7bb3 + NOW.md):
#   1. simd_gs removal (edits already in the worktree, uncommitted): simd_gs.py, newton_compat.install(simd=),
#      record_mpm.py --simd, loop_breakdown.py --simd, ab_timing.py +simd / +simdgraph, xfail test_simd_sweep_bitwise;
#      THIRD_PARTY_NOTICES, MPM.md, DECISIONS.md row marked removed, CHANGELOG. GPU: tests/test_mpm.py only
#      (.venv-newton152; gpu_history fresh_13_test_mpm 2.1-2.7 min). PR opened (not draft) only if green.
#   2. fused-update check (research note section 9), no code change: G1 64 worlds, captured, DEFAULT mode, 60 steps,
#      seed 0; A fused, B and C unfused (MJW_METAL_FUSE_UPDATE=0; B-C is the run-to-run floor); one process per arm
#      (scratch/b22/fused_cap_check.py; gpu_history mm19_fix_g1_64 0.3 min, mm18 0.5 min -> ~1 min per arm with
#      compile). Decision (scratch/b22/fused_eval.py) appended to the note, committed on the same branch, and posted
#      as a #29 follow-up comment.
# ~7 min. Resumable (scratch/b22/STATUS.txt).
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b22
L=$S/logs
O=$W/runs/mpm_main/batch_22
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
NOTE="$W/research_notes/MPM in main simulator/deterministic_g1_capture.md"
NEWM=8eb51fe
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O/fused"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_22 start $(date)"
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
gpu() {   # gpu NAME KIND MIN LOG PYTHONPATH -- cmd...   (cwd = this worktree)
  local name=$1 kind=$2 min=$3 log=$4 pyp=$5; shift 6
  say "gpu job $name ($kind, <= $min min) -> $log"
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$pyp" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -2 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = mpm-simd-gs-removal ] || { say "worktree not on mpm-simd-gs-removal"; exit 1; }
[ "$(git -C "$R/upstream/mujoco_warp" rev-parse --short=7 HEAD)" = "$NEWM" ] || { say "shared MuJoCo Warp checkout is not $NEWM: re-plan"; exit 1; }
for p in assets/isaac/G1 upstream/mujoco_menagerie; do [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"; done
health

# ---------------------------------------------------------------- 1. test_mpm.py on the removal
if ! done_ TMPM22; then
  gpu mm22_test_mpm check 15 "$L/test_mpm.log" "$W" -- "$PY152" -P -m pytest -q -rfEs tests/test_mpm.py
  rc=$?
  [ $rc = 0 ] && mark TMPM22 PASS "$(tail -1 "$L/test_mpm.log")" || mark TMPM22 FAIL "rc=$rc $(tail -1 "$L/test_mpm.log")"
fi

# ---------------------------------------------------------------- 2. fused-update check, arms A / B / C
if ! done_ FUSED22; then
  for arm in A B C; do
    [ -s "$O/fused/$arm.json" ] && continue
    ev=X=1; [ "$arm" != A ] && ev=MJW_METAL_FUSE_UPDATE=0
    gpu mm22_fused_$arm check 10 "$L/fused_$arm.log" "$W" -- env "$ev" "$PY" -P "$S/fused_cap_check.py" --root "$W" \
        --worlds 64 --steps 60 --out "$O/fused/$arm" || { mark FUSED22 FAIL "arm $arm"; exit 1; }
  done
  "$PY" "$S/fused_eval.py" "$O/fused" "$W/runs/mpm_main/batch_18/fix_g1_64.json" "$W/runs/mpm_main/batch_19/fix_g1_64.json" \
      "$NOTE" > "$L/fused_eval.log" 2>&1 || { cat "$L/fused_eval.log"; mark FUSED22 FAIL "eval"; exit 1; }
  cat "$L/fused_eval.log"
  mark FUSED22 DONE "$(tail -1 "$L/fused_eval.log")"
fi
DEC=$(grep "^FUSED22 DONE" "$STATUS" | tail -1 | awk '{print $NF}')

# ---------------------------------------------------------------- commits
if ! done_ COMMIT22; then
  git -C "$W" add -A metalsim/physics/mpm/simd_gs.py metalsim/physics/mpm/newton_compat.py metalsim/parity/isaac_side/record_mpm.py \
      metalsim/physics/mpm/probes/ab_timing.py metalsim/physics/mpm/probes/loop_breakdown.py tests/test_mpm.py \
      THIRD_PARTY_NOTICES.md metalsim/physics/mpm/MPM.md docs/DECISIONS.md CHANGELOG.md
  git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
MPM: remove the opt-in SIMD Gauss-Seidel sweep (simd_gs)

simd_gs.py ran each colored Gauss-Seidel sweep with 8 SIMD lanes per strain node. It was opt-in (newton_compat.install(simd=True)) and never the default. MPM.md's interleaved A/B (2026-10-09, granular example) measured it at 1.0x eager and +10 % with the chunk graph, then bitwise equal to Newton's serial loop. Since FMA contraction on Metal (Warp branch metalsim-fp-contract) its lane-shuffled sums contract differently, so it no longer reproduces the oracle bit for bit (test_simd_sweep_bitwise had become an xfail): the +10 % now comes with different numbers, and the all-core launches plus chunk graph stay the accelerated default.

Removed: simd_gs.py, install(simd=), record_mpm.py --simd, loop_breakdown.py --simd, ab_timing.py's +simd / +simdgraph variants (now rejected with a message), the xfail test. THIRD_PARTY_NOTICES (derived files: fused_gs.py only), MPM.md, CHANGELOG; the DECISIONS.md row stays, marked removed. Nothing else imported it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  mkdir -p "$O"; cp -R "$L" "$O/"; cp "$S/batch_22.sh" "$S/fused_cap_check.py" "$S/fused_eval.py" "$STATUS" "$O/" 2>/dev/null
  git -C "$W" add "$NOTE" runs/mpm_main/batch_22
  git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
Research note: the fused Newton update in default mode (batch 22, $DEC)

Section 9's check of the deterministic_g1_capture note: G1, 64
worlds, captured, default mode, seed 0; fused update vs unfused
twice (the run-to-run floor). Decision: $DEC. No code change.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  mark COMMIT22 DONE "$(git -C "$W" log --oneline -2 | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- #29 follow-up comment (the fused decision)
if ! done_ C29_22; then
  {
    echo "Follow-up (batch 22, 16-inch): the fused Newton update in **default** mode, research note \`deterministic_g1_capture.md\` section 9. Decision: **$DEC**."
    echo; grep "^\*\*\|^- " "$L/fused_eval.log"
    echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$S/c29.md"
  ( cd "$W" && gh pr comment 29 --body-file "$S/c29.md" ) > "$L/c29.log" 2>&1 && mark C29_22 DONE "$(tail -1 "$L/c29.log")" \
      || mark C29_22 FAIL "$(tail -1 "$L/c29.log")"
fi

# ---------------------------------------------------------------- push + PR (only if test_mpm passed)
if done_ TMPM22 && ! done_ PR22; then
  git -C "$W" push -q -u origin mpm-simd-gs-removal 2>&1 | grep -v "^remote:" | tail -2
  [ "$(git -C "$W" ls-remote origin refs/heads/mpm-simd-gs-removal | cut -c1-40)" = "$(git -C "$W" rev-parse HEAD)" ] \
      || { mark PR22 FAIL "push"; exit 1; }
  {
    echo "Removes the opt-in SIMD Gauss-Seidel sweep (\`metalsim/physics/mpm/simd_gs.py\`, \`newton_compat.install(simd=True)\`) and its xfail test. It was never the default. MPM.md measured it at +10 % with the chunk graph (1.0x eager) when it was bitwise equal to Newton's serial loop; since FMA contraction on Metal its lane-shuffled sums no longer are (\`test_simd_sweep_bitwise\` had become an xfail), so the speedup now comes with different numbers. Dependents removed with it: \`record_mpm.py --simd\`, \`loop_breakdown.py --simd\`, \`ab_timing.py\`'s \`+simd\` / \`+simdgraph\` variants (rejected with a message). Docs: THIRD_PARTY_NOTICES, MPM.md, CHANGELOG; the DECISIONS.md row stays, marked removed."
    echo
    echo "Also on this branch (no code change): the result of the fused-update check from #29's research note (section 9), decision **$DEC**; see the #29 comment."
    echo
    echo "Tests: \`tests/test_mpm.py\` on the 16-inch: $(grep '^TMPM22 PASS' "$STATUS" | tail -1 | cut -d' ' -f4-)."
    echo
    echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  } > "$S/pr_body.md"
  ( cd "$W" && gh pr create --base main --head mpm-simd-gs-removal --title "MPM: remove the opt-in SIMD Gauss-Seidel sweep (simd_gs)" \
      --body-file "$S/pr_body.md" ) > "$L/pr.log" 2>&1
  cat "$L/pr.log"
  grep -q "github.com/.*/pull/" "$L/pr.log" && mark PR22 PASS "$(grep -o 'https://github.com/[^ ]*/pull/[0-9]*' "$L/pr.log" | tail -1)" \
      || mark PR22 FAIL "gh pr create"
fi
done_ TMPM22 || say "test_mpm.py failed: commits are local, no push, no PR (see $L/test_mpm.log)"
say "=== batch_22 end $(date)"; cat "$STATUS"
