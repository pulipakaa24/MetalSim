#!/bin/bash
# eager tick vs captured tick at the same trajectory point (after 70 warm-up ticks), base and new; one job each
J=runs/rigid_perf/tools/job.sh
for t in base new; do
  for ph in tick captured; do
    scripts/gpu_run.sh rig_late_${t}_$ph timing 10 -- bash $J late_${t}_$ph $t --timeout 1200 -- \
      --part rigid --device metal:0 --worlds 2048 --max-triangle-pairs 8388608 --warmup 70 --kprof $ph --kprof-ticks 6 | tail -1
  done
done
scripts/gpu_run.sh rig_late_base_prof timing 10 -- bash $J late_base_prof base --timeout 1200 \
  --env WP_METAL_PROFILE=1 -- --part rigid --device metal:0 --worlds 2048 --max-triangle-pairs 8388608 --warmup 70 \
  --kprof tick --kprof-ticks 4 | tail -1
