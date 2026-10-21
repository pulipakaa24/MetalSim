#!/bin/bash
# table.sh TREE... : collide / substep / tick / captured tick at 1..4096 worlds, det off and on; one gpu_run job each
J=runs/rigid_perf/tools/job.sh
O=runs/rigid_perf/out
for W in 1 64 512 2048 4096; do
  for det in off on; do
    for t in "$@"; do
      extra=""; [ $det = on ] && extra="--det on --pipeline-det"
      [ $det = on ] && [ $W -gt 2048 ] && continue   # guard: deterministic buffers above 2048 worlds not run
      id=tab_${t}_${det}_$W
      [ -f $O/$id.json ] && continue
      scripts/gpu_run.sh rig_$id timing 15 -- bash $J $id $t --timeout 1500 -- --part rigid \
        --device metal:0 --worlds $W --max-triangle-pairs 8388608 --mode capture --steps 10 --warmup 6 $extra | tail -1
    done
  done
done
