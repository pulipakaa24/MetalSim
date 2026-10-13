#!/bin/bash
# launched mesh threadgroups sweep with the barrier fix (one gpu_run job per config)
J=/Users/aditya/robosim/.claude/worktrees/agent-a6ef6785c0d23ee50/runs/rigid_perf/tools/job.sh
for b in 160 320 640 2560 1280; do
  /Users/aditya/robosim/scripts/gpu_run.sh rig_sweep_$b timing 10 -- bash $J sweepfix_$b hangfix --timeout 900 \
    --env NP_EXP_MESH_BLOCKS=$b -- --part rigid --device metal:0 --worlds 2048 --max-triangle-pairs 8388608 \
    --warmup 3 --kprof collide --kprof-ticks 20 | tail -1
done
