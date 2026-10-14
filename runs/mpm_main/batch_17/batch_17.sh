#!/bin/bash
# batch 17 (run outside the agent sandbox):  bash /Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741/scratch/b17/batch_17.sh
#
# #29, G1 deterministic under capture.
# A  name the tripping launch UNDER CAPTURE: scripts/diagnostics/det_capture_probe.py records, after every deterministic
#    launch captured into a graph, a tiny kernel that copies Warp's sticky deterministic flag into a per-launch slot and
#    clears it; after the replays the first non-zero slot names the launch. Runs: 1,024 worlds with the early exit and
#    the fused update off (the unexplained third cause), and 64 worlds with each of them on (causes 1 and 2), so each
#    cause gets a kernel name.
# B  which switch carries batch_16's +116 % (coupled box, deterministic, captured, 64 worlds): early exit off only,
#    unfused only, both off, both on, deterministic and default mode (so the cost of each switch is separated from the
#    cost of the deterministic passes), and early-exit-off / unfused at 1,024 worlds; timing class, one process per run,
#    3 interleaved repetitions. Also the deterministic launch count per coupled step (uncaptured probe) with the fused
#    update on and off, the likely carrier of a launch-bound cost.
# C  #29's body: draft, three causes, cost table (scratch/b17/pr29_status.py); a comment with A and B.
# Resumable (scratch/b17/STATUS.txt). ~1 h.
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
R=/Users/aditya/robosim
S=$W/scratch/b17
L=$S/logs
O=$W/runs/mpm_main/batch_17
PY=$R/.venv/bin/python
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L" "$O"
STATUS=$S/STATUS.txt
touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
cd "$W" || exit 1
echo "=== batch_17 start $(date)"
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
  say "gpu job $name ($kind, <= $min min) -> $log" >&2
  "$R/scripts/gpu_run.sh" "$name" "$kind" "$((min + 5))" -- python3 "$W/scripts/watchdog.py" --stall 300 \
      --timeout $((min * 60)) --cwd "$W" -- env PYTHONPATH="$W" PYTHONDONTWRITEBYTECODE=1 "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  tail -2 "$log" | sed 's/^/    /' >&2
  health >&2
  if [ $rc -ge 124 ] && [ $rc -le 126 ]; then mark "$name" FAIL "watchdog rc=$rc" >&2; exit 91; fi
  return $rc
}
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = batchsim-deterministic ] || { say "det worktree not on batchsim-deterministic"; exit 1; }
for p in assets/isaac/G1 upstream/mujoco_menagerie; do [ -e "$W/$p" ] || ln -s "$R/$p" "$W/$p"; done
cp "$S/det_capture_probe.py" "$W/scripts/diagnostics/det_capture_probe.py"
health

# ---------------------------------------------------------------- A captured probe
P=$W/scripts/diagnostics/det_capture_probe.py
if ! done_ PROBE17; then
  : > "$L/probe_summary.txt"
  for v in "both_off:1024:MJW_METAL_ICB_EARLY_EXIT=0 MJW_METAL_FUSE_UPDATE=0" "unfused:64:MJW_METAL_FUSE_UPDATE=0" \
           "no_early_exit:64:MJW_METAL_ICB_EARLY_EXIT=0" "both_on:64:X=1"; do
    tag=${v%%:*}; rest=${v#*:}; n=${rest%%:*}; envs=${rest#*:}
    gpu mm17_probe_${tag}_$n check 30 "$L/probe_${tag}_$n.log" -- env $envs "$PY" -P "$P" --worlds $n --frames 20 \
        --json "$O/probe_${tag}_$n.json"
    r=$(grep "^RESULT " "$L/probe_${tag}_$n.log" | tail -1 | cut -c8-)
    "$PY" - "$tag" "$n" "$r" >> "$L/probe_summary.txt" 2>/dev/null <<'EOF'
import json, sys
tag, n, r = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.loads(r)
except Exception:
    print(f"{tag} @ {n} worlds: no result"); sys.exit()
fh = d.get("first_hit")
e = d.get("eager", {}).get("tripped")
hit = f"first tripping launch {fh['kernel']} (module {fh['module']}, dim {fh['dim']}, flag {fh['flag']}); {d['hit_slots']} of {d['recorded_det_launches']} captured launches set it; kernels {d['kernels_hit']}" if fh else f"no captured launch set the flag ({d['recorded_det_launches']} recorded)"
print(f"{tag} @ {n} worlds: {hit}" + (f"; eager trip {e}" if e else ""))
EOF
  done
  cat "$L/probe_summary.txt"
  mark PROBE17 DONE "$(tr '\n' '|' < "$L/probe_summary.txt" | cut -c1-900)"
fi

# ---------------------------------------------------------------- B cost decomposition (timing class)
BOX=$W/metalsim/physics/mpm/probes/scale_coupled.py
if ! done_ COST17; then
  for rep in 1 2 3; do
    for d in 1 0; do
      for v in "on:X=1" "ee_off:MJW_METAL_ICB_EARLY_EXIT=0" "unfused:MJW_METAL_FUSE_UPDATE=0" "both_off:MJW_METAL_ICB_EARLY_EXIT=0 MJW_METAL_FUSE_UPDATE=0"; do
        tag=${v%%:*}; envs=${v#*:}
        f="$O/box_64_d${d}_${tag}_r$rep.json"
        [ -s "$f" ] || gpu mm17_box64_d${d}_${tag}_r$rep timing 15 "$L/box_64_d${d}_${tag}_r$rep.log" -- env $envs "$PY" -P "$BOX" \
            --worlds 64 --deterministic $d --time --steps 40 --warmup 10 --json "$f"
      done
    done
    for v in "ee_off:MJW_METAL_ICB_EARLY_EXIT=0" "unfused:MJW_METAL_FUSE_UPDATE=0"; do
      tag=${v%%:*}; envs=${v#*:}
      f="$O/box_1024_d1_${tag}_r$rep.json"
      [ -s "$f" ] || gpu mm17_box1024_${tag}_r$rep timing 20 "$L/box_1024_d1_${tag}_r$rep.log" -- env $envs "$PY" -P "$BOX" \
          --worlds 1024 --deterministic 1 --time --steps 40 --warmup 10 --json "$f"
    done
  done
  python3 - "$O" "$W/runs/mpm_main/batch_16" > "$L/cost_summary.txt" 2>&1 <<'EOF'
import glob, json, statistics, sys
o, b16 = sys.argv[1], sys.argv[2]
def med(pat):
    v = [json.load(open(p))["ms_per_step_median"] for p in sorted(glob.glob(pat))]
    return statistics.median(v) if v else None
print("| scene, worlds | mode | both on | early exit off | unfused | both off |")
print("|---|---|---|---|---|---|")
for d, name in ((1, "deterministic"), (0, "default")):
    row = [med(f"{o}/box_64_d{d}_{t}_r*.json") for t in ("on", "ee_off", "unfused", "both_off")]
    print(f"| coupled box, 64 | {name} | " + " | ".join("-" if x is None else f"{x:.2f}" for x in row) + " |")
on = med(f"{b16}/box_1024_on_r*.json"); off = med(f"{b16}/box_1024_off_r*.json")
ee = med(f"{o}/box_1024_d1_ee_off_r*.json"); uf = med(f"{o}/box_1024_d1_unfused_r*.json")
print(f"| coupled box, 1,024 | deterministic | " + " | ".join("-" if x is None else f"{x:.2f}" for x in (on, ee, uf, off)) + " |")
EOF
  cat "$L/cost_summary.txt"
  gpu mm17_launches_on check 15 "$L/launches_fused.log" -- "$PY" -P "$W/scripts/diagnostics/det_mismatch_probe.py" --worlds 64 --frames 20
  gpu mm17_launches_off check 15 "$L/launches_unfused.log" -- env MJW_METAL_FUSE_UPDATE=0 "$PY" -P "$W/scripts/diagnostics/det_mismatch_probe.py" --worlds 64 --frames 20
  echo "deterministic launches over 20 G1 env steps (uncaptured): fused $(grep -ohE '"launches": [0-9]+' "$L/launches_fused.log" | head -1), unfused $(grep -ohE '"launches": [0-9]+' "$L/launches_unfused.log" | head -1)" | tee -a "$L/cost_summary.txt"
  mark COST17 DONE "see $L/cost_summary.txt"
fi

# ---------------------------------------------------------------- C #29
cp -R "$L" "$O/"; cp "$S/batch_17.sh" "$STATUS" "$O/" 2>/dev/null
python3 "$S/pr29_status.py" "$W/scratch/b08b/pr29_body.md" "$L/probe_summary.txt" && cp "$W/scratch/b08b/pr29_body.md" "$O/pr29_body.md"
{
  echo "Batch 17 (16-inch). Captured per-launch flag probe (which deterministic launch sets the tripwire under capture):"
  echo; echo '```'; cat "$L/probe_summary.txt"; echo '```'
  echo; echo "Cost decomposition, coupled box, ms per captured coupled step (median of 3 interleaved repetitions):"; echo
  grep "^|" "$L/cost_summary.txt"; echo; grep "^deterministic launches" "$L/cost_summary.txt"
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
} > "$S/pr29_comment.md"
( cd "$W" && gh pr edit 29 --body-file "$W/scratch/b08b/pr29_body.md"; gh pr ready 29 --undo; gh pr comment 29 --body-file "$S/pr29_comment.md" ) > "$L/pr29.log" 2>&1
git -C "$W" add scripts/diagnostics/det_capture_probe.py runs/mpm_main/batch_17
git -C "$W" diff --cached --quiet || git -C "$W" commit -q -s -F - <<'EOF'
Deterministic BatchSim: captured per-launch probe, cost decomposition (batch 17)

A probe that names the deterministic launch setting Warp's counter flag
inside a captured graph (a per-launch flag snapshot kernel recorded
after each deterministic launch), run on the G1 task for each of the
three causes, and the coupled-box timing of the early exit and fused
update switched off one at a time, deterministic and default mode.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git -C "$W" push -q origin batchsim-deterministic 2>&1 | grep -v "^remote:" | tail -2
say "=== batch_17 end $(date)"; cat "$STATUS"
