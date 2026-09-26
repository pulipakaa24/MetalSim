#!/bin/bash
# tp26 camera round: per-stage GPU profile of the camera-cartpole rollout and update (installed forks, MetalSim HEAD).
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
python scripts/diagnostics/camera_rollout_profile.py 1024 0 2>&1 | grep -v "^Module\|^Warp \|^   Dev\|^   CUDA\|^   Kern\|^     \"" | tee runs/tp26/camprof.log | cut -c1-170
echo "=== end $(date) power: $(pmset -g batt | head -1)"
