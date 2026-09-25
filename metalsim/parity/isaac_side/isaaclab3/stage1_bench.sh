#!/bin/bash
# Stage 1: Isaac Lab 3.0's runtime benchmark (random actions, no policy; the 3.0 replacement of benchmark_non_rl.py)
# for G1 flat and rough at 4096 envs on PhysX (Kit) and Newton/MuJoCo Warp. debug_vis off.
# Three timing variants per case: (a) 3.0 default 1000 steps after 50 warm-up, host-return timing;
# (b) same with --measure_sync_step (serialized, synchronized: comparable to MetalSim's synchronized numbers);
# (c) 100 steps, no warm-up (the 2.3.2 benchmark_non_rl.py protocol used for the 5.1 reference).
source $HOME/parity3_scripts/common.sh
exec > $OUT/stage1_bench.log 2>&1
set -x
mkdir -p $OUT/bench
# retry of the Newton fidelity recording (first attempt died in the settings dump, fixed in record_g1.py)
if [ ! -f $OUT/fidelity/newton_mjwarp/DONE ]; then
  wait_for_gpu; rm -rf $OUT/fidelity/newton_mjwarp; t0=$(date +%s)
  timeout 3600 python $HOME/parity3_scripts/record_g1.py --out $OUT/fidelity/newton_mjwarp --frame_every 5 --viz none physics=newton_mjwarp > $OUT/fidelity_newton_mjwarp_retry.log 2>&1
  echo "fidelity newton_mjwarp retry exit $? in $(( $(date +%s) - t0 )) s"
fi
for TASK in Isaac-Velocity-Flat-G1 Isaac-Velocity-Rough-G1; do
  for B in newton_mjwarp isaacsim_physx; do
    for V in a b c; do
      case $V in a) X="--num_steps 1000 --warmup_steps 50";; b) X="--num_steps 1000 --warmup_steps 50 --measure_sync_step";; c) X="--num_steps 100 --warmup_steps 0";; esac
      d=$OUT/bench/${TASK}_${B}_$V; mkdir -p $d
      wait_for_gpu; t0=$(date +%s)
      timeout 5400 isaaclab benchmark runtime --task $TASK --num_envs 4096 --seed 0 $X --output_path $d --benchmark_formatter schema,summary,json \
        --viz none physics=$B $NOVIS > $d/run.log 2>&1
      echo "bench $TASK $B $V exit $? in $(( $(date +%s) - t0 )) s" | tee -a $OUT/bench/summary.txt
    done
  done
done
touch $OUT/BENCH_DONE
