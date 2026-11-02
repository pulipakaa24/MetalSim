#!/bin/bash
# so101-servo MetalSim jobs, one GPU-queue job, sequential. From the MetalSim root:
#   scripts/gpu_run.sh so101-servo-metalsim render 20 -- bash runs/readme_media/scratch/so101-servo/run_metalsim.sh <so101Sim root>
set -u
S=runs/readme_media/scratch/so101-servo; SO=$1
echo "[ms] $(date '+%F %T') mujoco_warp $(git -C upstream/mujoco_warp log --oneline -1)"
.venv/bin/python $S/gpu_feeder_check.py 2>&1 | grep -v "^Module \|took .* ms"
for sv in ${SERVOS:-7v4_real 7v4}; do
  echo "[ms] $(date +%T) expert $sv"
  .venv/bin/python $S/eval_metalsim.py expert $sv $S/metalsim/expert_$sv $S/cpu/expert_${sv}_freshkin/reference.json $SO 2>&1 | grep -v "^Module \|took .* ms"
done
echo "[ms] $(date +%T) done"
