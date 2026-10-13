#!/bin/bash
# Frame time of tier 1 / tier 2 (cloth, defaults) at the render branch's commits, one process per tree and renderer.
# Trees: git archive of the commit's metalsim/ and scripts/diagnostics/ into scratch/rg2_trees/<tag>.
#   p1  2e58979d  sbvh for sensors / tier 2, per_env for tier 1 (the section-5 ramp)
#   p2  a40daac2  compact env grid
#   p2b 2ef6641f  merge of particle-atlas-guard
#   p3  5a7cc077  merge of workstream C
#   cur this branch
# Usage: gpu_run.sh rg2_bisect timing 30 -- bash runs/render_gaps2/bisect.sh ENVS
set -u
WT=/Users/aditya/robosim/.claude/worktrees/agent-a5314e27bee02c142
PY=/Users/aditya/robosim/.venv/bin/python
N=${1:-1024}
OUT=$WT/runs/render_gaps2/bisect.jsonl
for R in tier1 tier2; do
  for T in p1 p2 p2b p3 cur; do
    ROOT=$WT; [ "$T" != cur ] && ROOT=/Users/aditya/robosim/scratch/rg2_trees/$T
    AO=0; [ "$T" = cur ] && AO=1
    echo "== $R $T $N"
    $PY $WT/scripts/diagnostics/render_perf_profile.py --scene cloth --renderer $R --envs $N --features none --all-off $AO \
        --root $ROOT --tag $T --out $OUT 2>&1 | grep '^{'
  done
done
