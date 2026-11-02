#!/bin/bash
# so101-expert MetalSim eval, one GPU-queue job. From the MetalSim root:
#   scripts/gpu_run.sh so101-expert-metalsim render 15 -- bash runs/readme_media/scratch/so101-expert/run_metalsim.sh
set -u
S=runs/readme_media/scratch/so101-expert
echo "[ms] $(date '+%F %T') mujoco_warp $(git -C upstream/mujoco_warp log --oneline -1) power: $(pmset -g batt | tail -1)"
.venv/bin/python $S/eval_metalsim.py 7v4_real $S/metalsim/deploy_7v4_real 2>&1 | grep -v "^Module \|took .* ms"
echo "[ms] $(date '+%F %T') done"
