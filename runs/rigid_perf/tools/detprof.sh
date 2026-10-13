#!/bin/bash
# det-on collide profiles at 2048 (base vs dev = bounded deterministic sort), one gpu_run job per config
J=/Users/aditya/robosim/.claude/worktrees/agent-a6ef6785c0d23ee50/runs/rigid_perf/tools/job.sh
for t in base dev; do
  for ph in collide substep; do
    /Users/aditya/robosim/scripts/gpu_run.sh rig_detprof_${t}_$ph timing 10 -- bash $J detprof_${t}_$ph $t --timeout 900 \
      --env WP_METAL_PROFILE=1 -- --part rigid --device metal:0 --worlds 2048 --max-triangle-pairs 8388608 \
      --det on --pipeline-det --warmup 3 --kprof $ph --kprof-ticks 4 | tail -1
  done
done
