#!/bin/bash
# Smoke test: Isaac-Velocity-Flat-G1, 4096 envs, 3 rsl_rl iterations on each backend.
source $HOME/parity3_scripts/common.sh
exec > $OUT/stage0_smoke.log 2>&1
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv
for B in isaacsim_physx newton_mjwarp; do
  wait_for_gpu
  t0=$(date +%s)
  timeout 3600 isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 --num_envs 4096 --max_iterations 3 --seed 0 \
     --viz none physics=$B $NOVIS > $OUT/smoke_flat_$B.log 2>&1
  echo "smoke $B exit $? in $(( $(date +%s) - t0 )) s"
done
touch $OUT/SMOKE_DONE
