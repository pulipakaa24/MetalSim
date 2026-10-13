#!/bin/bash
# Final timings (Warp 95f6f121): the render branch's commits and this branch, then this branch per feature.
#   base = 532577fe (PR #4 merged), p2b = 2ef6641f (before workstream C), p3 = 5a7cc077 (workstream C merged), cur = this branch
# Usage: gpu_run.sh rg2_final timing 60 -- bash runs/render_gaps2/perf_final.sh ENVS
set -u
WT=/Users/aditya/robosim/.claude/worktrees/agent-a5314e27bee02c142
PY=/Users/aditya/robosim/.venv/bin/python
N=${1:-1024}
OUT=$WT/runs/render_gaps2/perf_final.jsonl
T=/Users/aditya/robosim/scratch/rg2_trees
run() {  # scene renderer envs root tag features allOff
  echo "== $*"
  $PY $WT/scripts/diagnostics/render_perf_profile.py --scene $1 --renderer $2 --envs $3 --root $4 --tag "$5" \
      --features $6 --all-off $7 --out $OUT 2>&1 | grep '^{'
}
for R in tier1 tier2; do
  for tree in p2b p3 base; do run cloth $R $N $T/$tree "$tree" none 0; done
  run cloth $R $N $WT cur all 1
done
if [ "$N" -le 1024 ]; then
  for tree in p2b base; do run glass tier2 64 $T/$tree "$tree" none 0; done
  run glass tier2 64 $WT cur all 1
  run cloth_inf tier1 $N $WT cur none 1
  run cloth_inf tier2 $N $WT cur none 1
fi
