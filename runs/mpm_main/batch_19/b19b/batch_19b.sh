#!/bin/bash
# batch 19b (run outside the agent sandbox):  bash scratch/b19b/batch_19b.sh
#
# #29: batch_19's INCONCLUSIVE was an evaluator bug. The classifier for test_iteration_cap_repeats on the old pin
# grepped for "counter phase mismatch", and pytest's repr truncation had cut that phrase out ("...-counter calls
# differ between its two passes" remained). This batch re-runs no probe and no test that already ran:
#   C  re-classify batch_19's testpre.log (scratch/b19b/classify_testpre.py, both phrasings), then re-run the
#      evaluator on batch_19's JSON files (into a scratch copy of the note: section 8 was already corrected by hand)
#   M1_CONFIRMED -> batch_19's planned path: fork branch metalsim-det-solvedone (3 commits, no AI co-author),
#      test_batch_deterministic.py (the test now reports the subprocess stderr in full), G1 timing (default vs
#      deterministic, interleaved, 3 reps), pin move + README / GAPS / CHANGELOG, smoke on the shared checkout,
#      #29 body + comment, out of draft
# Time (gpu_history: mm10_tests 4.7 min, G1 timing ~0.6 min per run x 12, b19 probes ~4 min not re-run): ~30 min.
# Resumable (scratch/b19b/STATUS.txt). Reuses scratch/b19's scripts and runs/mpm_main/batch_19 outputs.
set -u
W=.
R=.
S=$W/scratch/b19b
S19=$W/scratch/b19
L=$S/logs
O=$W/runs/mpm_main/batch_19
O18=$W/runs/mpm_main/batch_18
PY=$R/.venv/bin/python
MJFIX=$W/upstream/mjw-solvedone
MJW=$W/upstream/mujoco_warp-detccd
OLDM=4d05cb7d4453d82c94d19ab9fb392df2c62b1acd
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_19b start $(date)"
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
grep -q "^PROBES19 DONE .* g1 fix rc 0 0" "$S19/STATUS.txt" && grep -q "^TESTPRE19 DONE .* rc=1 " "$S19/STATUS.txt" \
    || { say "batch_19's probe / test markers are not as expected: re-plan"; exit 1; }
for p in assets/isaac/G1 upstream/mujoco_menagerie; do [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"; done

# ---------------------------------------------------------------- C: re-classify, re-evaluate (no GPU)
if ! done_ VERDICT19B; then
  t=$(python3 "$S/classify_testpre.py" "$S19/logs/testpre.log" 1)
  cp "$S19/fake/note2.md" "$S/note_scratch.md" 2>/dev/null || echo "# scratch" > "$S/note_scratch.md"
  sed -i '' '/^## 8\. Results (batch_19/,$d' "$S/note_scratch.md"
  read r64 r1024 < "$S19/g1fix_rc.txt"
  python3 "$S19/evaluate.py" "$O" "$O18" "$r64" "$r1024" "$t" "$S/note_scratch.md" > "$L/verdict.log" 2>&1
  cat "$L/verdict.log"
  mark VERDICT19B DONE "testpre class $t; $(tail -1 "$L/verdict.log")"
fi
VERDICT=$(grep "^VERDICT19B" "$STATUS" | tail -1 | awk '{print $NF}')
say "verdict: $VERDICT"

if [ "$VERDICT" = M1_CONFIRMED ]; then
  [ "$(git -C "$R/upstream/mujoco_warp" rev-parse HEAD)" = "$OLDM" ] || done_ PIN19B || { say "shared MuJoCo Warp checkout moved from ${OLDM:0:7}: re-plan"; exit 1; }
  health
  # ---------------------------------------------------------------- fork commits (MuJoCo Warp: no AI co-author, CLA)
  if ! done_ FORK19B; then
    git -C "$MJW" fetch -q fork || { mark FORK19B FAIL "fetch"; exit 1; }
    [ "$(git -C "$MJW" rev-parse fork/metalsim)" = "$OLDM" ] || { mark FORK19B FAIL "fork metalsim moved from ${OLDM:0:7}: re-plan"; exit 1; }
    if ! git -C "$MJW" log --format=%s -1 metalsim-det-solvedone 2>/dev/null | grep -q "METALSIM_CHANGES.md: deterministic-mode"; then
      git -C "$MJW" checkout -q -B metalsim-det-solvedone "$OLDM" && git -C "$MJW" apply "$W/scratch/b18/solvedone.diff" \
          || { mark FORK19B FAIL "patch"; exit 1; }
      git -C "$MJW" add mujoco_warp/_src/solver.py
      git -C "$MJW" commit -q -s -F - <<'EOF'
Keep the solver iteration count in a local in _solve_done

_solve_done incremented solver_niter in place and read it back to test the iteration cap, then took a slot from the nsolving counter (consumed for the Metal early exit). Under Warp's deterministic mode a kernel with a consumed counter runs a counting pass with stores and non-counter atomics suppressed, so there the read-back saw the old count: a world reaching the cap made the counter call in the execution pass only, and the slots no longer matched the calls (Warp fork's tripwire: phase mismatch). The count now lives in a local, and the counter's return is consumed only with the early exit. The CG variant gets the same local count.
EOF
      cp "$MJFIX/mujoco_warp/_src/solver.py" "$MJW/mujoco_warp/_src/solver.py"
      [ "$(git -C "$MJW" diff --numstat | awk '{print $1+$2}')" -le 8 ] || { mark FORK19B FAIL "second change larger than expected"; exit 1; }
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
    cmp -s "$MJW/mujoco_warp/_src/solver.py" "$MJFIX/mujoco_warp/_src/solver.py" || { mark FORK19B FAIL "differs from the tested copy"; exit 1; }
    git -C "$MJW" push -q fork metalsim-det-solvedone 2>&1 | grep -v "^remote:" | tail -2
    NEWM=$(git -C "$MJW" rev-parse metalsim-det-solvedone)
    git -C "$MJW" ls-remote fork refs/heads/metalsim-det-solvedone | grep -q "$NEWM" && mark FORK19B PASS "$NEWM" || { mark FORK19B FAIL "push"; exit 1; }
  fi
  NEWM=$(grep "^FORK19B PASS" "$STATUS" | tail -1 | awk '{print $4}')
  # ---------------------------------------------------------------- tests with the fork commits
  if ! done_ TESTS19B; then
    git -C "$MJW" checkout -q metalsim-det-solvedone
    gpu mm19b_tests check 30 "$L/test_batch_deterministic.log" "$MJW:$W" -- "$PY" -P -m pytest -q -rfEs tests/test_batch_deterministic.py
    rc=$?
    [ $rc = 0 ] && mark TESTS19B PASS "$(tail -1 "$L/test_batch_deterministic.log")" || { mark TESTS19B FAIL "rc=$rc"; exit 1; }
  fi
  # ---------------------------------------------------------------- G1 timing, default vs deterministic, interleaved (timing class)
  if ! done_ TIMING19B; then
    for rep in 1 2 3; do for n in 64 1024; do for det in 0 1; do
      f="$O/g1_${n}_d${det}_r$rep.json"
      [ -s "$f" ] || gpu mm19b_t_${n}_d${det}_r$rep timing 15 "$L/g1_${n}_d${det}_r$rep.log" "$MJW:$W" -- "$PY" -P \
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
        print(f"  | G1 flat env step (batch_19b) | {n:,} | {d0:.2f} ms | {d1:.2f} ms | {100 * (d1 / d0 - 1):+.1f} % |")
print("  | coupled box step (batch_17, same session) | 64 | 23.87 ms | 24.32 ms | +1.9 % |")
print("  | coupled box step (batch_10 default, batch_17 deterministic) | 1,024 | 228.78 ms | 231.50 ms | +1.2 % |")
EOF
    cat "$O/cost_summary.txt"
    mark TIMING19B DONE "$(tr '\n' ' ' < "$O/cost_summary.txt" | cut -c1-500)"
  fi
  # ---------------------------------------------------------------- MuJoCo Warp pin move
  if ! done_ PIN19B; then
    git -C "$MJW" diff --abbrev=7 88af9cc "$OLDM" > "$S/check_old.diff"
    git -C "$W" show "HEAD:patches/mujoco_warp/metalsim-${OLDM:0:7}.diff" > "$S/committed_old.diff"
    cmp -s "$S/check_old.diff" "$S/committed_old.diff" || { mark PIN19B FAIL "fallback diff method"; exit 1; }
    git -C "$MJW" push -q fork "$NEWM:refs/heads/metalsim" 2>&1 | grep -v "^remote:" | tail -2
    git -C "$MJW" ls-remote fork refs/heads/metalsim | grep -q "$NEWM" || { mark PIN19B FAIL "fork metalsim fast-forward refused"; exit 1; }
    ( cd "$R" && FORKS=mujoco_warp scripts/fork_sync.sh ) > "$L/fork_sync.log" 2>&1; cat "$L/fork_sync.log"
    [ "$(git -C "$R/upstream/mujoco_warp" rev-parse HEAD)" = "$NEWM" ] || { mark PIN19B FAIL "shared checkout not at ${NEWM:0:7}"; exit 1; }
    git -C "$W" rm -q "patches/mujoco_warp/metalsim-${OLDM:0:7}.diff"
    git -C "$MJW" diff --abbrev=7 88af9cc "$NEWM" > "$W/patches/mujoco_warp/metalsim-${NEWM:0:7}.diff"
    python3 "$S19/mjw_pin.py" "$W" "$OLDM" "$NEWM" > "$L/mjw_pin.log" 2>&1 || { cat "$L/mjw_pin.log"; mark PIN19B FAIL "pin docs"; exit 1; }
    python3 "$S19/det_docs19.py" "$W" "$NEWM" "$O/cost_summary.txt" > "$L/det_docs.log" 2>&1 || { cat "$L/det_docs.log"; mark PIN19B FAIL "docs"; exit 1; }
    mark PIN19B PASS "MuJoCo Warp ${OLDM:0:7} -> ${NEWM:0:7}"
  fi
  if ! done_ SMOKE19B; then
    gpu mm19b_smoke check 30 "$L/smoke_shared.log" "$W" -- "$PY" -P -m pytest -q -rfEs tests/test_batch_deterministic.py
    rc=$?
    [ $rc = 0 ] && mark SMOKE19B PASS "$(tail -1 "$L/smoke_shared.log")" || mark SMOKE19B FAIL "rc=$rc"
  fi
fi

# ---------------------------------------------------------------- record, commit, #29
mkdir -p "$O/b19b"; cp -R "$L" "$O/b19b/"; cp "$S/batch_19b.sh" "$S/classify_testpre.py" "$STATUS" "$O/b19b/" 2>/dev/null
git -C "$W" add "research_notes/MPM in main simulator/deterministic_g1_capture.md" tests/test_batch_deterministic.py runs/mpm_main/batch_19
if done_ PIN19B; then
  git -C "$W" add -A patches/mujoco_warp README.md docs/GUIDE.md docs/STATUS.md docs/GAPS.md CHANGELOG.md
fi
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<EOF
Deterministic BatchSim: G1 fix shipped (batch 19b, $VERDICT)

batch_19's INCONCLUSIVE was an evaluator bug: pytest's repr
truncation cut "counter phase mismatch" out of the old-pin test
failure, whose remaining text was the same error. Re-classified from
the log (nothing re-run): M1 confirmed. The test now reports the
subprocess's stderr in full; research note section 8 corrected.
With it: MuJoCo Warp pin with the _solve_done fix and the unfused
update in deterministic mode, G1 deterministic-mode cost.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
python3 "$S19/pr29_resolved.py" "$W/scratch/b08b/pr29_body.md" "$VERDICT" "$O/cost_summary.txt" "$(grep '^FORK19B PASS' "$STATUS" | tail -1 | awk '{print substr($4,1,7)}')" \
    && cp "$W/scratch/b08b/pr29_body.md" "$O/b19b/pr29_body.md"
{
  echo "Batch 19b (16-inch), verdict **$VERDICT**. batch_19's INCONCLUSIVE was an evaluator bug: the old-pin run of \`test_iteration_cap_repeats\` failed on Warp's counter phase mismatch, but pytest's repr truncation cut the phrase the classifier looked for. Re-classified from the same log; no probe re-run. Research note section 8 corrected."
  echo; grep "^- \|^M1 confirmed\|^Not confirmed" "$L/verdict.log"
  echo; echo '```'; cat "$STATUS"; echo '```'
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr edit 29 --body-file "$W/scratch/b08b/pr29_body.md"; gh pr comment 29 --body-file "$S/pr29_comment.md"
  if [ "$VERDICT" = M1_CONFIRMED ] && done_ SMOKE19B; then gh pr ready 29; fi ) > "$L/pr29.log" 2>&1
cat "$L/pr29.log"
say "=== batch_19b end $(date)"; cat "$STATUS"
