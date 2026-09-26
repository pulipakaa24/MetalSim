#!/bin/bash
# Go2 per-kernel attribution, both cones (the first generality job's grep dropped these lines); raw logs kept.
cd /Users/aditya/robosim && source .venv/bin/activate
export MENAGERIE=/Users/aditya/robosim/upstream/mujoco_menagerie
D=scripts/diagnostics/competitors
echo "=== start $(date) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for cone in pyramidal elliptic; do
  echo "== go2 $cone kernels"; CONE=$cone WP_METAL_PROFILE=1 python $D/metalsim_profile.py go2 4096 kernels > runs/mjw_tp/ab/go2_kernels_$cone.log 2>&1
  grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]" runs/mjw_tp/ab/go2_kernels_$cone.log | head -16
done
echo "=== end $(date)"
