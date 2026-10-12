#!/bin/bash
# verify.sh TREE... : repeat checks (eager + captured graph) with dumps, det off and on, 64 and 2048 worlds; one job each
J=/Users/aditya/robosim/.claude/worktrees/agent-a6ef6785c0d23ee50/runs/rigid_perf/tools/job.sh
O=/Users/aditya/robosim/.claude/worktrees/agent-a6ef6785c0d23ee50/runs/rigid_perf/out
for W in 64 2048; do
  for t in "$@"; do
    for det in off on; do
      extra=""; [ $det = on ] && extra="--det on --pipeline-det"
      id=rep_${t}_${det}_$W
      [ -f $O/$id.json ] && continue
      kind=check; [ $W -ge 2048 ] && kind=timing
      /Users/aditya/robosim/scripts/gpu_run.sh rig_$id $kind 20 -- bash $J $id $t --timeout 2400 -- --part rigid \
        --device metal:0 --worlds $W --max-triangle-pairs 8388608 --mode capture --steps 3 --warmup 2 --repeat-check \
        --dump $O/dump_$id $extra | tail -1
    done
  done
done
