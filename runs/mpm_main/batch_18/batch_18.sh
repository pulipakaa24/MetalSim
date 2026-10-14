#!/bin/bash
# batch 18 (run outside the agent sandbox):  bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b18/batch_18.sh
#
# #29, G1 deterministic: discriminate the mechanism of the _solve_done trip, test the fix, decide.
# Research and design: research_notes/MPM in main simulator/deterministic_g1_capture.md (prior art, flag semantics,
# mechanisms M1-M4 and their predictions). Every run records per physics step the deterministic launches that set
# Warp's flag (diagnostic Warp build upstream/warp-diag: bit 2 no phase-0 record, 16 beyond a nonzero phase-0 count,
# 8 other destination/count, 4 fewer calls) and the worlds that reached the iteration cap
# (scripts/diagnostics/det_step_probe.py).
#   E1 pre-fix  stack 64 worlds cap 2, uncaptured / captured;  G1 64 / 1,024 captured
#   E2 fix      the same with MuJoCo Warp's _solve_done fix (upstream/mjw-solvedone), + G1 det_cost_g1 64 / 1,024
#               (batch_16's failing configuration, BatchSim.check_overflow)
#   verdict (scratch/b18/evaluate.py, appended to the note):
#     M1_CONFIRMED  -> MuJoCo Warp fork branch metalsim-det-solvedone (commit, push); tests/test_batch_deterministic.py
#                      with it; G1 deterministic timing (3 interleaved reps); MuJoCo Warp pin move (fork metalsim
#                      fast-forward, this Mac's checkout through the queue, patches, README / GUIDE / STATUS / GAPS /
#                      CHANGELOG); smoke on the shared checkout; #29 body + comment, out of draft
#     otherwise     -> #29 comment with the verdict and the per-run evidence; the fix stays uncommitted
# Resumable (scratch/b18/STATUS.txt). E1+E2 ~45 min; the M1_CONFIRMED path +~60 min.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b18
L=$S/logs
O=$W/runs/mpm_main/batch_18
PY=$R/.venv/bin/python
DIAGW=$W/upstream/warp-diag
MJFIX=$W/upstream/mjw-solvedone
MJW=$W/upstream/mujoco_warp-detccd
NOTE="$W/research_notes/MPM in main simulator/deterministic_g1_capture.md"
OLDM=4d05cb7d4453d82c94d19ab9fb392df2c62b1acd
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_18 start $(date)"
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
gpu() {   # gpu NAME KIND MIN LOG PYTHONPATH -- cmd...   (cwd = the det worktree)
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
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
for p in assets/isaac/G1 upstream/mujoco_menagerie; do [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"; done
cp "$S/det_step_probe.py" "$W/scripts/diagnostics/det_step_probe.py"
[ -f "$DIAGW/warp/bin/libwarp.dylib" ] || cp "$R"/upstream/warp-innate/warp/bin/*.dylib "$DIAGW/warp/bin/"
health

# ---------------------------------------------------------------- E1 / E2 probes
P=$W/scripts/diagnostics/det_step_probe.py
probe() {   # probe TAG PYTHONPATH args...
  local tag=$1 pyp=$2; shift 2
  [ -s "$O/$tag.json" ] && return 0
  gpu mm18_$tag check 30 "$L/$tag.log" "$pyp" -- "$PY" -P "$P" "$@" --json "$O/$tag.json"
}
if ! done_ PROBES18; then
  PRE=$DIAGW:$W; FIX=$MJFIX:$DIAGW:$W
  probe pre_stack_eager "$PRE" --scene stack --worlds 64 --iterations 2 --capture 0 --steps 20
  probe pre_stack_capt "$PRE" --scene stack --worlds 64 --iterations 2 --capture 1 --steps 40
  probe pre_g1_64 "$PRE" --scene g1 --worlds 64 --capture 1 --steps 60
  probe pre_g1_1024 "$PRE" --scene g1 --worlds 1024 --capture 1 --steps 60
  probe fix_stack_eager "$FIX" --scene stack --worlds 64 --iterations 2 --capture 0 --steps 20
  probe fix_stack_capt "$FIX" --scene stack --worlds 64 --iterations 2 --capture 1 --steps 40
  probe fix_g1_64 "$FIX" --scene g1 --worlds 64 --capture 1 --steps 60
  probe fix_g1_1024 "$FIX" --scene g1 --worlds 1024 --capture 1 --steps 60
  r64=0; r1024=0
  gpu mm18_g1fix_64 check 25 "$L/g1fix_64.log" "$FIX" -- "$PY" -P "$W/scripts/diagnostics/det_cost_g1.py" --worlds 64 --deterministic 1 \
      --frames 40 --warmup 10 --json "$O/g1fix_64.json" || r64=$?
  gpu mm18_g1fix_1024 check 25 "$L/g1fix_1024.log" "$FIX" -- "$PY" -P "$W/scripts/diagnostics/det_cost_g1.py" --worlds 1024 --deterministic 1 \
      --frames 40 --warmup 10 --json "$O/g1fix_1024.json" || r1024=$?
  echo "$r64 $r1024" > "$S/g1fix_rc.txt"
  mark PROBES18 DONE "g1 fix rc $r64 $r1024"
fi
if ! done_ VERDICT18; then
  read r64 r1024 < "$S/g1fix_rc.txt"
  python3 "$S/evaluate.py" "$O" "$r64" "$r1024" "$NOTE" > "$L/verdict.log" 2>&1
  cat "$L/verdict.log"
  mark VERDICT18 DONE "$(tail -1 "$L/verdict.log")"
fi
VERDICT=$(grep "^VERDICT18" "$STATUS" | tail -1 | awk '{print $4}')
say "verdict: $VERDICT"

if [ "$VERDICT" = M1_CONFIRMED ]; then
  # ---------------------------------------------------------------- fork commit
  if ! done_ FORK18; then
    git -C "$MJW" fetch -q fork || { mark FORK18 FAIL "fetch"; exit 1; }
    [ "$(git -C "$MJW" rev-parse fork/metalsim)" = "$OLDM" ] || { mark FORK18 FAIL "fork metalsim moved from ${OLDM:0:7}: re-plan"; exit 1; }
    if ! git -C "$MJW" log --format=%s -1 metalsim-det-solvedone 2>/dev/null | grep -q "METALSIM_CHANGES.md: _solve_done"; then
      git -C "$MJW" checkout -q -B metalsim-det-solvedone "$OLDM" && git -C "$MJW" apply "$S/solvedone.diff" \
          || { mark FORK18 FAIL "patch"; exit 1; }
      cmp -s "$MJW/mujoco_warp/_src/solver.py" "$MJFIX/mujoco_warp/_src/solver.py" || { mark FORK18 FAIL "differs from the tested copy"; exit 1; }
      git -C "$MJW" add mujoco_warp/_src/solver.py
      git -C "$MJW" commit -q -s -F - <<'EOF'
Keep the solver iteration count in a local in _solve_done

_solve_done incremented solver_niter in place and read it back to test the iteration cap, then took a slot from the nsolving counter (consumed for the Metal early exit, and assigned even without it). Under Warp's deterministic mode a kernel with a consumed counter runs a counting pass with stores and non-counter atomics suppressed, so there the read-back saw the old count: a world reaching the cap made the counter call in the execution pass only, and the slots no longer matched the calls (Warp fork's tripwire: phase mismatch). The count now lives in a local, and the counter's return is consumed only with the early exit. The CG variant gets the same local count.
EOF
      sha=$(git -C "$MJW" rev-parse --short=7 HEAD)
      "$PY" - "$MJW/METALSIM_CHANGES.md" "$sha" <<'EOF'
import sys
p, sha = sys.argv[1:3]
lines = open(p).read().split("\n")
last = max(i for i, l in enumerate(lines) if l.startswith("| "))
lines.insert(last + 1, f"| `{sha}` | `_solve_done` keeps the solver's iteration count in a local (it read back an in-place increment, which Warp's deterministic counting pass suppresses: a world at the iteration cap then made the early exit's counter call in one pass only); the counter's return is consumed only with the early exit. |")
open(p, "w").write("\n".join(lines))
EOF
      git -C "$MJW" add METALSIM_CHANGES.md
      git -C "$MJW" commit -q -s -m "METALSIM_CHANGES.md: _solve_done iteration count"
    fi
    git -C "$MJW" push -q fork metalsim-det-solvedone 2>&1 | grep -v "^remote:" | tail -2
    NEWM=$(git -C "$MJW" rev-parse metalsim-det-solvedone)
    git -C "$MJW" ls-remote fork refs/heads/metalsim-det-solvedone | grep -q "$NEWM" && mark FORK18 PASS "$NEWM" || { mark FORK18 FAIL "push"; exit 1; }
  fi
  NEWM=$(grep "^FORK18 PASS" "$STATUS" | tail -1 | awk '{print $4}')
  # ---------------------------------------------------------------- tests with the fork commit
  if ! done_ TESTS18; then
    git -C "$MJW" checkout -q metalsim-det-solvedone
    gpu mm18_tests check 40 "$L/test_batch_deterministic.log" "$MJW:$W" -- "$PY" -P -m pytest -q -rfEs tests/test_batch_deterministic.py
    rc=$?
    [ $rc = 0 ] && mark TESTS18 PASS "$(tail -1 "$L/test_batch_deterministic.log")" || { mark TESTS18 FAIL "rc=$rc"; exit 1; }
  fi
  # ---------------------------------------------------------------- G1 deterministic timing (timing class)
  if ! done_ TIMING18; then
    for rep in 1 2 3; do for n in 64 1024; do
      f="$O/g1det_${n}_r$rep.json"
      [ -s "$f" ] || gpu mm18_t_${n}_r$rep timing 20 "$L/g1det_${n}_r$rep.log" "$MJW:$W" -- "$PY" -P "$W/scripts/diagnostics/det_cost_g1.py" \
          --worlds $n --deterministic 1 --frames 100 --warmup 20 --json "$f"
    done; done
    python3 - "$O" "$W/runs/mpm_main/batch_10_det" > "$S/cost_summary.txt" <<'EOF'
import glob, json, statistics, sys
o, b10 = sys.argv[1], sys.argv[2]
def med(p, k):
    v = [json.load(open(f))[k] for f in sorted(glob.glob(p))]
    return statistics.median(v) if v else None
print("  | scene | worlds | default | deterministic | cost |")
print("  |---|---|---|---|---|")
for n in (64, 1024):
    d0, d1 = med(f"{b10}/g1_{n}_0_r*.json", "mean_step_ms"), med(f"{o}/g1det_{n}_r*.json", "mean_step_ms")
    if d0 and d1:
        print(f"  | G1 flat env step | {n:,} | {d0:.2f} ms | {d1:.2f} ms | {100 * (d1 / d0 - 1):+.1f} % |")
    b0, b1 = med(f"{b10}/box_{n}_0_r*.json", "ms_per_step_median"), med(f"{b10}/box_{n}_1_r*.json", "ms_per_step_median")
    if b0 and b1:
        print(f"  | coupled box step | {n:,} | {b0:.2f} ms | {b1:.2f} ms | {100 * (b1 / b0 - 1):+.1f} % |")
EOF
    cat "$S/cost_summary.txt"
    mark TIMING18 DONE "$(tr '\n' ' ' < "$S/cost_summary.txt" | cut -c1-500)"
  fi
  # ---------------------------------------------------------------- MuJoCo Warp pin move
  if ! done_ PIN18; then
    git -C "$MJW" diff --abbrev=7 88af9cc "$OLDM" > "$S/check_old.diff"
    git -C "$W" show "HEAD:patches/mujoco_warp/metalsim-${OLDM:0:7}.diff" > "$S/committed_old.diff"
    cmp -s "$S/check_old.diff" "$S/committed_old.diff" || { mark PIN18 FAIL "fallback diff method"; exit 1; }
    git -C "$MJW" push -q fork "$NEWM:refs/heads/metalsim" 2>&1 | grep -v "^remote:" | tail -2
    git -C "$MJW" ls-remote fork refs/heads/metalsim | grep -q "$NEWM" || { mark PIN18 FAIL "fork metalsim fast-forward refused"; exit 1; }
    ( cd "$R" && FORKS=mujoco_warp scripts/fork_sync.sh ) > "$L/fork_sync.log" 2>&1; cat "$L/fork_sync.log"
    [ "$(git -C "$R/upstream/mujoco_warp" rev-parse HEAD)" = "$NEWM" ] || { mark PIN18 FAIL "shared checkout not at ${NEWM:0:7}"; exit 1; }
    git -C "$W" rm -q "patches/mujoco_warp/metalsim-${OLDM:0:7}.diff"
    git -C "$MJW" diff --abbrev=7 88af9cc "$NEWM" > "$W/patches/mujoco_warp/metalsim-${NEWM:0:7}.diff"
    python3 "$S/mjw_pin.py" "$W" "$OLDM" "$NEWM" > "$L/mjw_pin.log" 2>&1 || { cat "$L/mjw_pin.log"; mark PIN18 FAIL "docs"; exit 1; }
    mark PIN18 PASS "MuJoCo Warp ${OLDM:0:7} -> ${NEWM:0:7}"
  fi
  if ! done_ SMOKE18; then
    gpu mm18_smoke check 40 "$L/smoke_shared.log" "$W" -- "$PY" -P -m pytest -q -rfEs tests/test_batch_deterministic.py
    rc=$?
    [ $rc = 0 ] && mark SMOKE18 PASS "$(tail -1 "$L/smoke_shared.log")" || mark SMOKE18 FAIL "rc=$rc"
  fi
fi

# ---------------------------------------------------------------- record, commit, #29
cp -R "$L" "$O/"; cp "$S/batch_18.sh" "$S/evaluate.py" "$S/solvedone.diff" "$STATUS" "$O/" 2>/dev/null
cp -R "$DIAGW/warp/native/metal_deterministic.h" "$O/warp_diag_metal_deterministic.h" 2>/dev/null
git -C "$W" add scripts/diagnostics/det_step_probe.py "research_notes/MPM in main simulator/deterministic_g1_capture.md" \
    runs/mpm_main/batch_18 metalsim/physics/mpm/probes/rigid_det_diag.py tests/test_batch_deterministic.py
if done_ PIN18; then
  git -C "$W" add -A patches/mujoco_warp README.md docs/GUIDE.md docs/STATUS.md docs/GAPS.md CHANGELOG.md
fi
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
Deterministic BatchSim: G1, _solve_done at the iteration cap (batch 18, $VERDICT)

Research note (prior art, Warp's two-pass semantics and flag bits,
mechanisms with their predictions) and the per-step probe that
records, per physics step, the deterministic launches that trip
Warp's tripwire and the worlds at the iteration cap. Verdict:
$VERDICT. With M1 confirmed: MuJoCo Warp pin to the _solve_done fix,
test_iteration_cap_repeats, deterministic-mode cost on G1.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
python3 "$S/pr29_resolved.py" "$W/scratch/b08b/pr29_body.md" "$VERDICT" "$S/cost_summary.txt" "$(grep '^FORK18 PASS' "$STATUS" | tail -1 | awk '{print substr($4,1,7)}')" \
    && cp "$W/scratch/b08b/pr29_body.md" "$O/pr29_body.md"
{
  echo "Batch 18 (16-inch), verdict **$VERDICT**. Research note: \`research_notes/MPM in main simulator/deterministic_g1_capture.md\` (prior art, flag semantics, mechanisms and the per-run evidence)."
  echo; cat "$L/verdict.log" | grep "^- "
  echo; echo '```'; cat "$STATUS"; echo '```'
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr edit 29 --body-file "$W/scratch/b08b/pr29_body.md"; gh pr comment 29 --body-file "$S/pr29_comment.md"
  if [ "$VERDICT" = M1_CONFIRMED ] && done_ SMOKE18; then gh pr ready 29; fi ) > "$L/pr29.log" 2>&1
cat "$L/pr29.log"
say "=== batch_18 end $(date)"; cat "$STATUS"
