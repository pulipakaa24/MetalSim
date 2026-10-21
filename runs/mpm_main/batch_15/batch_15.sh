#!/bin/bash
# batch 15 (run outside the agent sandbox, after batch_14 with FIX_N152=1):
#   bash scratch/b15/batch_15.sh
#
# P1  batch_11's SYNC11 "FAIL" was its own check's false negative (it grepped "commit_id":"<sha>" without the space pip
#     writes); the sync had worked (upstream/newton at 1301e8da, .venv's newton at 9d6616c6). Verified here with
#     python's json + git, recorded as SYNC11 PASS with the evidence, then batch_11's PIN_MOVE=1 run resumes at DOCS11.
# P2  #29, G1 deterministic mode fails only under graph capture (batch_13: uncaptured clean at 64 and 1,024 worlds with
#     and without the fused update; captured fails at both). On Metal MuJoCo Warp has no conditional node
#     (graph_conditional off); what capture adds is (a) MetalSim's exact early exit of the solver loop: each
#     iteration's launches are an indirect-command-buffer range whose length _solve_done zeroes for the remaining
#     iterations (MJW_METAL_ICB_EARLY_EXIT), (b) capture memory reuse (WP_METAL_CAPTURE_REUSE), (c) replay through
#     indirect command buffers at all (WP_METAL_ICB). Captured G1 deterministic at 64 worlds with each switched off
#     (one process each), then the configurations that pass at 1,024 worlds.
# P3  guard (new branch newton152-guard from origin/main, PR): tests/test_mpm.py fails instead of skipping when
#     .venv-newton152 imports stock Newton / Warp (unless METALSIM_NEWTON152_STOCK=1), setup_newton152.sh refuses a
#     stale fork checkout and checks what the environment imports. Checked: test_mpm on the repaired environment; the
#     new test failing on stock Newton; the stock opt-in skipping; the setup script in a throwaway venv, and its refusal.
# Resumable (scratch/b15/STATUS.txt). ~1 h.
set -u
W=.
R=.
S=$W/scratch/b15
L=$S/logs
O=$W/runs/mpm_main/batch_15
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
WG=$W/scratch/wt-guard
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_15 start $(date)"
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
  tail -2 "$log" | sed 's/^/    /'
  health
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc"; exit 91; fi
  return $rc
}
health

# ---------------------------------------------------------------- P1 batch_11's sync, then its pin move
B11=$W/scratch/b11
if ! done_ P1; then
  . "$B11/pins.txt"
  ev=$("$PY" - "$NEW_HT" <<'EOF'
import glob, json, sys
f = glob.glob(".venv/lib/python3.12/site-packages/newton-*.dist-info/direct_url.json")
d = json.load(open(f[0]))
print("venv newton commit_id", d.get("vcs_info", {}).get("commit_id"), "expected", sys.argv[1], "ok" if d.get("vcs_info", {}).get("commit_id") == sys.argv[1] else "MISMATCH")
EOF
)
  hn=$(git -C "$R/upstream/newton" rev-parse HEAD)
  echo "$ev; upstream/newton $hn (expected $NEW_NF)" | tee "$L/sync11_evidence.log"
  if echo "$ev" | grep -q " ok$" && [ "$hn" = "$NEW_NF" ]; then
    grep -v "^SYNC11 FAIL" "$B11/STATUS.txt" > "$B11/STATUS.tmp" && mv "$B11/STATUS.tmp" "$B11/STATUS.txt"
    echo "SYNC11 PASS $(date +%H:%M:%S) verified by batch_15 (json + git): .venv newton ${NEW_HT:0:8}, upstream/newton ${NEW_NF:0:8}" >> "$B11/STATUS.txt"
    PIN_MOVE=1 bash "$B11/batch_11.sh" > "$L/batch_11_resume.log" 2>&1
    rc=$?; tail -12 "$L/batch_11_resume.log"
    [ $rc = 0 ] && mark P1 DONE "$(grep -E '^(MSFINAL11|PR11)' "$B11/STATUS.txt" | tr '\n' ';')" || mark P1 FAIL "batch_11 rc=$rc (scratch/b11/STATUS.txt)"
  else
    mark P1 FAIL "sync not verified: $ev; upstream/newton $hn"
  fi
fi

# ---------------------------------------------------------------- P2 G1 deterministic under capture: which capture feature
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
G1=$W/scripts/diagnostics/det_cost_g1.py
if ! done_ P2A; then
  res=""
  for v in "base:" "no_early_exit:MJW_METAL_ICB_EARLY_EXIT=0" "no_capture_reuse:WP_METAL_CAPTURE_REUSE=0" "no_icb:WP_METAL_ICB=0" \
           "no_early_exit_unfused:MJW_METAL_ICB_EARLY_EXIT=0 MJW_METAL_FUSE_UPDATE=0"; do
    name=${v%%:*}; envs=${v#*:}
    gpu mm15_$name 20 "$L/g1_64_$name.log" "$W" "$W" -- env $envs "$PY" -P "$G1" --worlds 64 --deterministic 1 --frames 40 --warmup 10 \
        --json "$O/g1_64_$name.json"
    rc=$?
    why=$(grep -hoE "phase mismatch|Array shapes must not exceed[^,]*|counter overflow" "$L/g1_64_$name.log" | head -1)
    res="$res $name:rc=$rc${why:+($why)}"
  done
  mark P2A DONE "64 worlds:$res"
fi
if ! done_ P2B; then
  res=""
  for v in "no_early_exit:MJW_METAL_ICB_EARLY_EXIT=0" "no_capture_reuse:WP_METAL_CAPTURE_REUSE=0" "no_icb:WP_METAL_ICB=0"; do
    name=${v%%:*}; envs=${v#*:}
    grep -q "$name:rc=0" "$STATUS" || continue        # only configurations that passed at 64 worlds
    gpu mm15_${name}_1024 25 "$L/g1_1024_$name.log" "$W" "$W" -- env $envs "$PY" -P "$G1" --worlds 1024 --deterministic 1 --frames 30 \
        --warmup 10 --json "$O/g1_1024_$name.json"
    rc=$?
    why=$(grep -hoE "phase mismatch|Array shapes must not exceed[^,]*|counter overflow" "$L/g1_1024_$name.log" | head -1)
    res="$res $name:rc=$rc${why:+($why)}"
  done
  mark P2B DONE "1,024 worlds:${res:- none passed at 64}"
fi
cp -R "$L" "$O/"; cp "$S/batch_15.sh" "$STATUS" "$O/" 2>/dev/null
git -C "$W" add runs/mpm_main/batch_15
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: G1 under capture, which capture feature (batch 15)

Captured G1 deterministic runs with MuJoCo Warp's indirect-command-
buffer early exit, Warp's capture memory reuse and indirect-command-
buffer replay each switched off.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2

# ---------------------------------------------------------------- P3 guard branch
if ! done_ P3; then
  git -C "$W" fetch -q origin
  [ -d "$WG" ] || git -C "$W" worktree add -q -B newton152-guard "$WG" origin/main || { mark P3 FAIL "worktree"; exit 1; }
  if git -C "$WG" diff --quiet HEAD -- tests/test_mpm.py scripts/setup_newton152.sh; then    # not applied yet
    git -C "$WG" apply "$S/guard.diff" || { mark P3 FAIL "guard.diff does not apply to origin/main $(git -C "$WG" rev-parse --short HEAD)"; exit 1; }
  fi
  for p in assets/isaac/G1 upstream/mujoco_menagerie; do mkdir -p "$(dirname "$WG/$p")"; [ -e "$WG/$p" ] || ln -s "$R/$p" "$WG/$p"; done
  gpu mm15_guard_fork 30 "$L/guard_test_mpm_fork.log" "$WG" "$WG" -- "$PY152" -P -m pytest -q -rfEs tests/test_mpm.py
  r1=$?
  gpu mm15_guard_stock 15 "$L/guard_test_mpm_stock.log" "$WG" "$R/upstream/newton-1.5.2:$WG" -- "$PY152" -P -m pytest -q -rfEs \
      tests/test_mpm.py -k "environment_has_the_forks"
  r2=$?
  gpu mm15_guard_stock_ok 15 "$L/guard_test_mpm_stock_optin.log" "$WG" "$R/upstream/newton-1.5.2:$WG" -- env METALSIM_NEWTON152_STOCK=1 \
      "$PY152" -P -m pytest -q -rfEs tests/test_mpm.py -k "environment_has_the_forks or color_blocks"
  r3=$?
  ( cd "$R" && rm -rf "$W/scratch/venv-guard" && zsh "$WG/scripts/setup_newton152.sh" "$W/scratch/venv-guard" ) > "$L/guard_setup_ok.log" 2>&1
  r4=$?
  ( cd "$R" && METALSIM_NEWTON152_DIR=upstream/newton-1.5.2 zsh "$WG/scripts/setup_newton152.sh" "$W/scratch/venv-guard" ) > "$L/guard_setup_stale.log" 2>&1
  r5=$?
  say "fork rc=$r1 stock rc=$r2 stock-optin rc=$r3 setup rc=$r4 setup-stale rc=$r5"
  if [ $r1 = 0 ] && [ $r2 != 0 ] && grep -q "is stock Newton 1.5.2" "$L/guard_test_mpm_stock.log" && [ $r3 = 0 ] \
     && grep -q "SKIPPED" "$L/guard_test_mpm_stock_optin.log" && [ $r4 = 0 ] && [ $r5 != 0 ] && grep -q "not the pinned" "$L/guard_setup_stale.log"; then
    mark P3 PASS "fork: $(tail -1 "$L/guard_test_mpm_fork.log"); stock fails as intended; stock opt-in skips; setup ok; stale checkout refused"
  else
    mark P3 FAIL "fork rc=$r1 stock rc=$r2 optin rc=$r3 setup rc=$r4 stale rc=$r5"
  fi
fi
if done_ P3 && ! done_ P3PR; then
  mkdir -p "$WG/runs/newton152_guard"; cp "$L"/guard_* "$WG/runs/newton152_guard/"
  git -C "$WG" add tests/test_mpm.py scripts/setup_newton152.sh runs/newton152_guard
  git -C "$WG" commit -q -s -F - <<'EOF'
.venv-newton152: fail, not skip, when the environment is stock

tests/test_mpm.py: test_environment_has_the_forks fails when the
environment imports stock Newton 1.5.2 or a Warp without the fork's
API; the fork-only tests skip only in a stock environment declared on
purpose (METALSIM_NEWTON152_STOCK=1). setup_newton152.sh: an existing
fork checkout off the pin is an error (unless METALSIM_NEWTON152_REF),
and the final check requires the environment to import the checkout it
installed, with the fork's API. On 2026-10-13 this Mac's environment
imported stock Newton for hours with the fork tests skipped.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  git -C "$WG" push -q -u origin newton152-guard 2>&1 | grep -v "^remote:" | tail -2
  ( cd "$WG" && gh pr create --base main --head newton152-guard --title ".venv-newton152: fail, not skip, when the environment is stock" \
      --body-file "$S/pr_body.md" ) > "$L/pr.log" 2>&1
  cat "$L/pr.log"; mark P3PR DONE "$(tail -1 "$L/pr.log")"
fi
say "=== batch_15 end $(date)"; cat "$STATUS"
