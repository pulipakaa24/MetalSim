#!/bin/bash
# Franka-Pour on the L4 (Isaac Lab 3.0-EA, release/3.0.0 ae37b028): does Isaac's own IsaacContrib-Franka-Pour build and
# step with the published Isaac 6.1 assets? 4 envs, 1 rsl_rl iteration. Log: ~/parity3/pour/smoke.log
source $HOME/parity3_scripts/common.sh
mkdir -p $OUT/pour
exec > $OUT/pour/smoke.log 2>&1
set -x
wait_for_gpu
timeout 1800 isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour --num_envs 4 --max_iterations 1 \
  --seed 0 --run_name smoke --viz none
echo "exit $?"
touch $OUT/pour/SMOKE_DONE
