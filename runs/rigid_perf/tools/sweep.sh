#!/bin/bash
# launched mesh threadgroups sweep with the barrier fix (one gpu_run job per config)
J=runs/rigid_perf/tools/job.sh
for b in 160 320 640 2560 1280; do
  scripts/gpu_run.sh rig_sweep_$b timing 10 -- bash $J sweepfix_$b hangfix --timeout 900 \
    --env NP_EXP_MESH_BLOCKS=$b -- --part rigid --device metal:0 --worlds 2048 --max-triangle-pairs 8388608 \
    --warmup 3 --kprof collide --kprof-ticks 20 | tail -1
done
