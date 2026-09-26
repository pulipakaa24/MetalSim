#!/bin/bash
# Generality check (task D): the G1 fast paths on Go2 (competitor protocol, both cones), SO-101 lift and Panda (their own
# BatchSimOptions, both cones), 4096 worlds, with the eager per-kernel attribution. Installed fork (shared checkout).
cd /Users/aditya/robosim && source .venv/bin/activate
export MENAGERIE=/Users/aditya/robosim/upstream/mujoco_menagerie
D=scripts/diagnostics/competitors
echo "=== start $(date) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
python3 scripts/gpu_lock.py status
for cone in pyramidal elliptic; do
  echo "== go2 $cone (metalsim_step.py matched)"; CONE=$cone python $D/metalsim_step.py go2 4096 matched 2>&1 | grep "RESULT\|Error\|Traceback"
  echo "== go2 $cone kernels"; CONE=$cone WP_METAL_PROFILE=1 python $D/metalsim_profile.py go2 4096 kernels 2>&1 | grep "KERNELS\|ms \|Error\|Traceback" | head -14
done
for s in so101_lift panda; do
  for cone in pyramidal elliptic; do
    python scripts/diagnostics/g1_tp_generality.py $s 4096 --cone $cone 2>&1 | grep "^\[\|Error\|Traceback"
    WP_METAL_PROFILE=1 python scripts/diagnostics/g1_tp_generality.py $s 4096 --cone $cone --kernels 2>&1 | grep "^\[\|^   \|Error\|Traceback"
  done
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
