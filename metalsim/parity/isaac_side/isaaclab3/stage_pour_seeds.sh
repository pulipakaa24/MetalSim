#!/bin/bash
# Franka-Pour seed study on the L4 (Isaac Lab 3.0-EA ae37b028, Isaac's own task + franka_pour_published_asset.patch):
# the matched configuration of PARITY.md §1b.5 (1,024 envs, 60 rsl_rl iterations, the task's runner config) for
# SEEDS (default "1 2 3 4"), one after another, then the VM powers itself off (POWEROFF=0 to keep it up).
# About 5.2 h per seed on the L4 (314 s per iteration), so ~21 h for four seeds.
# Logs: ~/parity3/pour/train_1024_curve_s<seed>.log
#
# From the Mac (after `gcloud auth login`):
#   gcloud compute instances start aditya-isaac --zone us-west1-a --project mcmc-193822
#   gcloud compute scp --zone us-west1-a --tunnel-through-iap --project mcmc-193822 \
#     metalsim/parity/isaac_side/isaaclab3/{stage_pour.sh,stage_pour_seeds.sh,franka_pour_published_asset.patch} \
#     aditya-isaac:~/parity3_scripts/
#   gcloud compute ssh aditya-isaac --zone us-west1-a --tunnel-through-iap --project mcmc-193822 \
#     --command 'chmod +x ~/parity3_scripts/stage_pour*.sh; nohup ~/parity3_scripts/stage_pour_seeds.sh > /dev/null 2>&1 &' < /dev/null
# Afterwards copy ~/parity3/pour/train_1024_curve_s*.log to runs/parity/isaaclab3_pour/ (replace the VM username
# with <vm-user>) and compare: python -m metalsim.learn.franka_pour_curve --isaac <log> ...
SEEDS=${SEEDS:-"1 2 3 4"}
for s in $SEEDS; do
  STAGES=train ITERS=60 NUM=1024 SEED=$s TAG=_curve_s$s $HOME/parity3_scripts/stage_pour.sh
done
touch $HOME/parity3/pour/SEEDS_DONE
[ "${POWEROFF:-1}" = 1 ] && { sleep 60; sudo poweroff; }
