#!/bin/bash
# batch 19 (run outside the agent sandbox):  bash scratch/b19/batch_19.sh
#
# #29, G1 deterministic: close batch_18's gaps (research note section 7), then ship.
#   The MuJoCo Warp fix copy (upstream/mjw-solvedone) now has two changes against pin 4d05cb7:
#     _solve_done keeps the iteration count in a local (batch_18's fix);
#     the Newton update takes the unfused launches in Warp's deterministic mode. The fused plan at 1,024 worlds
#     under capture asked for 3,534,564,096 scratch elements: all three of batch_18's 1,024 crashes.
#   P  probes (diagnostic Warp first on PYTHONPATH):
#        stack, 64 worlds, cap 2, 200 steps (free fall about 110 steps, then contact), pin / fix x captured / uncaptured;
#        G1 1,024 worlds on the pin with MJW_METAL_FUSE_UPDATE=0; G1 64 / 1,024 with the fix;
#        det_cost_g1 64 / 1,024 with the fix (check_overflow)
#   T  test_iteration_cap_repeats on the pin: M1 predicts a counter phase-mismatch failure (otherwise the test is vacuous)
#   V  scratch/b19/evaluate.py (accepts flag 6 = 2|4; appends section 8 to the note) -> M1_CONFIRMED / INCONCLUSIVE
#   M1_CONFIRMED -> fork branch metalsim-det-solvedone (3 commits, no AI co-author), test_batch_deterministic.py,
#        G1 timing (default vs deterministic, interleaved, 3 reps), pin move, README / GAPS / CHANGELOG, smoke on the
#        shared checkout, #29 body + comment, out of draft
#   otherwise -> #29 comment with the evidence; nothing pushed to the fork
# Time (gpu_history: b18 probe arms 0.2-0.8 min, det_cost 0.3 min, mm10_tests 4.7 min, G1 timing 0.6 min per run):
#   P+T+V ~15 min, the M1_CONFIRMED path ~30 min more.  Resumable (scratch/b19/STATUS.txt).
set -u
W=.
R=.
S=$W/scratch/b19
L=$S/logs
O=$W/runs/mpm_main/batch_19
O18=$W/runs/mpm_main/batch_18
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
echo "=== batch_19 start $(date)"
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
[ "$(git -C "$R/upstream/mujoco_warp" rev-parse HEAD)" = "$OLDM" ] || { say "shared MuJoCo Warp checkout is not the pin ${OLDM:0:7}: re-plan"; exit 1; }
for p in assets/isaac/G1 upstream/mujoco_menagerie; do [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"; done
[ -f "$DIAGW/warp/bin/libwarp.dylib" ] || cp "$R"/upstream/warp-innate/warp/bin/*.dylib "$DIAGW/warp/bin/"
health

# ---------------------------------------------------------------- P: probes
P=$W/scripts/diagnostics/det_step_probe.py
probe() {   # probe TAG PYTHONPATH ENV args...   (ENV: one VAR=value, X=1 for none)
  local tag=$1 pyp=$2 ev=$3; shift 3
  [ -s "$O/$tag.json" ] && return 0
  gpu mm19_$tag check 15 "$L/$tag.log" "$pyp" -- env "$ev" "$PY" -P "$P" "$@" --json "$O/$tag.json"
}
if ! done_ PROBES19; then
  PRE=$DIAGW:$W; FIX=$MJFIX:$DIAGW:$W
  probe pre_stack_capt "$PRE" X=1 --scene stack --worlds 64 --iterations 2 --capture 1 --steps 200
  probe fix_stack_capt "$FIX" X=1 --scene stack --worlds 64 --iterations 2 --capture 1 --steps 200
  probe pre_stack_eager "$PRE" X=1 --scene stack --worlds 64 --iterations 2 --capture 0 --steps 200
  probe fix_stack_eager "$FIX" X=1 --scene stack --worlds 64 --iterations 2 --capture 0 --steps 200
  probe pre_g1_1024u "$PRE" MJW_METAL_FUSE_UPDATE=0 --scene g1 --worlds 1024 --capture 1 --steps 60
  probe fix_g1_64 "$FIX" X=1 --scene g1 --worlds 64 --capture 1 --steps 60
  probe fix_g1_1024 "$FIX" X=1 --scene g1 --worlds 1024 --capture 1 --steps 60
  r64=0; r1024=0
  gpu mm19_g1fix_64 check 15 "$L/g1fix_64.log" "$FIX" -- "$PY" -P "$W/scripts/diagnostics/det_cost_g1.py" --worlds 64 --deterministic 1 \
      --frames 40 --warmup 10 --json "$O/g1fix_64.json" || r64=$?
  gpu mm19_g1fix_1024 check 15 "$L/g1fix_1024.log" "$FIX" -- "$PY" -P "$W/scripts/diagnostics/det_cost_g1.py" --worlds 1024 --deterministic 1 \
      --frames 40 --warmup 10 --json "$O/g1fix_1024.json" || r1024=$?
  echo "$r64 $r1024" > "$S/g1fix_rc.txt"
  mark PROBES19 DONE "g1 fix rc $r64 $r1024"
fi
# ---------------------------------------------------------------- T: the regression test must fail on the pin
if ! done_ TESTPRE19; then
  gpu mm19_testpre check 15 "$L/testpre.log" "$W" -- "$PY" -P -m pytest -q -rfE tests/test_batch_deterministic.py \
      -k test_iteration_cap_repeats
  rc=$?
  if [ $rc = 0 ]; then t=0; elif grep -q "counter phase mismatch" "$L/testpre.log"; then t=1; else t=2; fi
  echo "$t" > "$S/testpre.txt"
  mark TESTPRE19 DONE "rc=$rc class=$t (1 = failed on the phase mismatch, as M1 predicts)"
fi
# ---------------------------------------------------------------- V: verdict
if ! done_ VERDICT19; then
  read r64 r1024 < "$S/g1fix_rc.txt"; t=$(cat "$S/testpre.txt")
  python3 "$S/evaluate.py" "$O" "$O18" "$r64" "$r1024" "$t" "$NOTE" > "$L/verdict.log" 2>&1
  cat "$L/verdict.log"
  mark VERDICT19 DONE "$(tail -1 "$L/verdict.log")"
fi
VERDICT=$(grep "^VERDICT19" "$STATUS" | tail -1 | awk '{print $4}')
say "verdict: $VERDICT"

if [ "$VERDICT" = M1_CONFIRMED ]; then
  # ---------------------------------------------------------------- fork commits (MuJoCo Warp: no AI co-author, CLA)
  if ! done_ FORK19; then
    git -C "$MJW" fetch -q fork || { mark FORK19 FAIL "fetch"; exit 1; }
    [ "$(git -C "$MJW" rev-parse fork/metalsim)" = "$OLDM" ] || { mark FORK19 FAIL "fork metalsim moved from ${OLDM:0:7}: re-plan"; exit 1; }
    if ! git -C "$MJW" log --format=%s -1 metalsim-det-solvedone 2>/dev/null | grep -q "METALSIM_CHANGES.md: deterministic-mode"; then
      git -C "$MJW" checkout -q -B metalsim-det-solvedone "$OLDM" && git -C "$MJW" apply "$W/scratch/b18/solvedone.diff" \
          || { mark FORK19 FAIL "patch"; exit 1; }
      git -C "$MJW" add mujoco_warp/_src/solver.py
      git -C "$MJW" commit -q -s -F - <<'EOF'
Keep the solver iteration count in a local in _solve_done

_solve_done incremented solver_niter in place and read it back to test the iteration cap, then took a slot from the nsolving counter (consumed for the Metal early exit). Under Warp's deterministic mode a kernel with a consumed counter runs a counting pass with stores and non-counter atomics suppressed, so there the read-back saw the old count: a world reaching the cap made the counter call in the execution pass only, and the slots no longer matched the calls (Warp fork's tripwire: phase mismatch). The count now lives in a local, and the counter's return is consumed only with the early exit. The CG variant gets the same local count.
EOF
      cp "$MJFIX/mujoco_warp/_src/solver.py" "$MJW/mujoco_warp/_src/solver.py"
      [ "$(git -C "$MJW" diff --numstat | awk '{print $1+$2}')" -le 8 ] || { mark FORK19 FAIL "second change larger than expected"; exit 1; }
      git -C "$MJW" add mujoco_warp/_src/solver.py
      git -C "$MJW" commit -q -s -F - <<'EOF'
Take the unfused Newton update in Warp's deterministic mode

The fused per-iteration update zeroes and reads back its own change counter within the launch, which Warp's deterministic counting pass suppresses, and its deterministic plan at 1,024 worlds under graph capture asks for more than 2^31 scratch elements. In deterministic mode (wp.config.deterministic other than NOT_GUARANTEED) _fuse_update now returns False; the unfused launches measured the same time as the fused one there.
EOF
      "$PY" - "$MJW/METALSIM_CHANGES.md" "$(git -C "$MJW" rev-parse --short=7 HEAD~1)" "$(git -C "$MJW" rev-parse --short=7 HEAD)" <<'EOF'
import sys
p, a, b = sys.argv[1:4]
lines = open(p).read().split("\n")
last = max(i for i, l in enumerate(lines) if l.startswith("| "))
lines[last + 1:last + 1] = [
    f"| `{a}` | `_solve_done` keeps the solver's iteration count in a local (it read back an in-place increment, which Warp's deterministic counting pass suppresses: a world at the iteration cap then made the early exit's counter call in one pass only); the counter's return is consumed only with the early exit. |",
    f"| `{b}` | Warp's deterministic mode takes the unfused Newton update (the fused kernel reads back its own change counter within the launch; its deterministic plan at 1,024 worlds asked for more than 2^31 scratch elements). |"]
open(p, "w").write("\n".join(lines))
EOF
      git -C "$MJW" add METALSIM_CHANGES.md
      git -C "$MJW" commit -q -s -m "METALSIM_CHANGES.md: deterministic-mode _solve_done and update"
    fi
    cmp -s "$MJW/mujoco_warp/_src/solver.py" "$MJFIX/mujoco_warp/_src/solver.py" || { mark FORK19 FAIL "differs from the tested copy"; exit 1; }
    git -C "$MJW" push -q fork metalsim-det-solvedone 2>&1 | grep -v "^remote:" | tail -2
    NEWM=$(git -C "$MJW" rev-parse metalsim-det-solvedone)
    git -C "$MJW" ls-remote fork refs/heads/metalsim-det-solvedone | grep -q "$NEWM" && mark FORK19 PASS "$NEWM" || { mark FORK19 FAIL "push"; exit 1; }
  fi
  NEWM=$(grep "^FORK19 PASS" "$STATUS" | tail -1 | awk '{print $4}')
  # ---------------------------------------------------------------- tests with the fork commits
  if ! done_ TESTS19; then
    git -C "$MJW" checkout -q metalsim-det-solvedone
    gpu mm19_tests check 30 "$L/test_batch_deterministic.log" "$MJW:$W" -- "$PY" -P -m pytest -q -rfEs tests/test_batch_deterministic.py
    rc=$?
    [ $rc = 0 ] && mark TESTS19 PASS "$(tail -1 "$L/test_batch_deterministic.log")" || { mark TESTS19 FAIL "rc=$rc"; exit 1; }
  fi
  # ---------------------------------------------------------------- G1 timing, default vs deterministic, interleaved (timing class)
  if ! done_ TIMING19; then
    for rep in 1 2 3; do for n in 64 1024; do for det in 0 1; do
      f="$O/g1_${n}_d${det}_r$rep.json"
      [ -s "$f" ] || gpu mm19_t_${n}_d${det}_r$rep timing 15 "$L/g1_${n}_d${det}_r$rep.log" "$MJW:$W" -- "$PY" -P \
          "$W/scripts/diagnostics/det_cost_g1.py" --worlds $n --deterministic $det --frames 100 --warmup 20 --json "$f"
    done; done; done
    python3 - "$O" > "$O/cost_summary.txt" <<'EOF'
import glob, json, statistics, sys
o = sys.argv[1]
def med(p):
    v = [json.load(open(f))["mean_step_ms"] for f in sorted(glob.glob(p))]
    return statistics.median(v) if len(v) == 3 else None
print("  | scene | worlds | default | deterministic | cost |")
print("  |---|---|---|---|---|")
for n in (64, 1024):
    d0, d1 = med(f"{o}/g1_{n}_d0_r*.json"), med(f"{o}/g1_{n}_d1_r*.json")
    if d0 and d1:
        print(f"  | G1 flat env step (batch_19) | {n:,} | {d0:.2f} ms | {d1:.2f} ms | {100 * (d1 / d0 - 1):+.1f} % |")
print("  | coupled box step (batch_17, same session) | 64 | 23.87 ms | 24.32 ms | +1.9 % |")
print("  | coupled box step (batch_10 default, batch_17 deterministic) | 1,024 | 228.78 ms | 231.50 ms | +1.2 % |")
EOF
    cat "$O/cost_summary.txt"
    mark TIMING19 DONE "$(tr '\n' ' ' < "$O/cost_summary.txt" | cut -c1-500)"
  fi
  # ---------------------------------------------------------------- MuJoCo Warp pin move
  if ! done_ PIN19; then
    git -C "$MJW" diff --abbrev=7 88af9cc "$OLDM" > "$S/check_old.diff"
    git -C "$W" show "HEAD:patches/mujoco_warp/metalsim-${OLDM:0:7}.diff" > "$S/committed_old.diff"
    cmp -s "$S/check_old.diff" "$S/committed_old.diff" || { mark PIN19 FAIL "fallback diff method"; exit 1; }
    git -C "$MJW" push -q fork "$NEWM:refs/heads/metalsim" 2>&1 | grep -v "^remote:" | tail -2
    git -C "$MJW" ls-remote fork refs/heads/metalsim | grep -q "$NEWM" || { mark PIN19 FAIL "fork metalsim fast-forward refused"; exit 1; }
    ( cd "$R" && FORKS=mujoco_warp scripts/fork_sync.sh ) > "$L/fork_sync.log" 2>&1; cat "$L/fork_sync.log"
    [ "$(git -C "$R/upstream/mujoco_warp" rev-parse HEAD)" = "$NEWM" ] || { mark PIN19 FAIL "shared checkout not at ${NEWM:0:7}"; exit 1; }
    git -C "$W" rm -q "patches/mujoco_warp/metalsim-${OLDM:0:7}.diff"
    git -C "$MJW" diff --abbrev=7 88af9cc "$NEWM" > "$W/patches/mujoco_warp/metalsim-${NEWM:0:7}.diff"
    python3 "$S/mjw_pin.py" "$W" "$OLDM" "$NEWM" > "$L/mjw_pin.log" 2>&1 || { cat "$L/mjw_pin.log"; mark PIN19 FAIL "pin docs"; exit 1; }
    python3 "$S/det_docs19.py" "$W" "$NEWM" "$O/cost_summary.txt" > "$L/det_docs.log" 2>&1 || { cat "$L/det_docs.log"; mark PIN19 FAIL "docs"; exit 1; }
    mark PIN19 PASS "MuJoCo Warp ${OLDM:0:7} -> ${NEWM:0:7}"
  fi
  if ! done_ SMOKE19; then
    gpu mm19_smoke check 30 "$L/smoke_shared.log" "$W" -- "$PY" -P -m pytest -q -rfEs tests/test_batch_deterministic.py
    rc=$?
    [ $rc = 0 ] && mark SMOKE19 PASS "$(tail -1 "$L/smoke_shared.log")" || mark SMOKE19 FAIL "rc=$rc"
  fi
fi

# ---------------------------------------------------------------- record, commit, #29
cp -R "$L" "$O/"; cp "$S/batch_19.sh" "$S/evaluate.py" "$S/solvedone.diff" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add scripts/diagnostics/det_step_probe.py "research_notes/MPM in main simulator/deterministic_g1_capture.md" runs/mpm_main/batch_19
if done_ PIN19; then
  git -C "$W" add -A patches/mujoco_warp README.md docs/GUIDE.md docs/STATUS.md docs/GAPS.md CHANGELOG.md
fi
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
Deterministic BatchSim: G1 at the iteration cap, 1,024 worlds (batch 19, $VERDICT)

batch_18's gaps: the stack scene now reaches the iteration cap (200
steps), the 1,024-world runs take the unfused update (the fused
kernel's deterministic plan asked for more than 2^31 scratch
elements), flag 6 read as M1's signature, and test_iteration_cap_repeats
checked to fail on the old pin. With M1 confirmed: MuJoCo Warp pin
with the _solve_done fix and the unfused update in deterministic
mode, G1 deterministic-mode cost.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
python3 "$S/pr29_resolved.py" "$W/scratch/b08b/pr29_body.md" "$VERDICT" "$O/cost_summary.txt" "$(grep '^FORK19 PASS' "$STATUS" | tail -1 | awk '{print substr($4,1,7)}')" \
    && cp "$W/scratch/b08b/pr29_body.md" "$O/pr29_body.md"
{
  echo "Batch 19 (16-inch), verdict **$VERDICT**. Research note: \`research_notes/MPM in main simulator/deterministic_g1_capture.md\` (section 7: reading batch_18; section 8: these runs)."
  echo; grep "^- \|^M1 confirmed\|^Not confirmed" "$L/verdict.log"
  echo; echo '```'; cat "$STATUS"; echo '```'
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr edit 29 --body-file "$W/scratch/b08b/pr29_body.md"; gh pr comment 29 --body-file "$S/pr29_comment.md"
  if [ "$VERDICT" = M1_CONFIRMED ] && done_ SMOKE19; then gh pr ready 29; fi ) > "$L/pr29.log" 2>&1
cat "$L/pr29.log"
say "=== batch_19 end $(date)"; cat "$STATUS"
