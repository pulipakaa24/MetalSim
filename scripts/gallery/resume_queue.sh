#!/bin/bash
# GPU queue after the pause: stage videos -> fixed-PPO demonstration run -> replay re-render (+compare).
cd /Users/aditya/robosim
bash scripts/gallery/g1_stage_videos.sh > runs/g1_stage_videos.log 2>&1
while [ -e runs/.gpu_lock ]; do sleep 30; done
echo "g1_flat_ppowarp_fixed (main session) $(date)" > runs/.gpu_lock; trap "rm -f runs/.gpu_lock" EXIT
.venv/bin/python -m metalsim.learn.g1_velocity 4096 flat train 1000 runs/g1_flat_ppowarp_fixed.log runs/policies/g1_flat_ppowarp_fixed.pt 0.0025 > runs/g1_flat_ppowarp_fixed.stdout 2>&1
echo "PPO_FIXED_DONE exit $?" >> runs/g1_flat_ppowarp_fixed.stdout
rm -f runs/.gpu_lock
bash scripts/gallery/g1_replay_rerender.sh > runs/g1_replay_rerender.log 2>&1
echo QUEUE_DONE >> runs/g1_replay_rerender.log
