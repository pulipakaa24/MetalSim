#!/bin/bash
# hfield-epa validation V4 (timing class): repeat of V3's 2.3.2-config smokes, 93e5ce1 vs the hfield-epa fix only, three
# interleaved rounds of the seed-1 (boxes_local) and seed-0 (heightfield) configs, to separate a cost change from noise.
cd ~/MetalSim
R=runs/rough_rerun_2026-10-31; S=runs/readme_media/scratch/hfield-epa; O=$S/val4; PY=.venv/bin/python
mkdir -p $O; FIX=$(cat $S/mjw_fix/COMMIT)
C232="--reward_cfg rough_isaac --contact_cfg default --base_velocity origin --feet_slide_velocity origin --effort_limit actuator --initial_reset 1"
CS0="--seed 0 --scan_ordering ij --terrain_collision hfield --scan_surface grid $C232"
CS1="--seed 1 --scan_ordering xy --terrain_collision boxes_local --scan_surface exact $C232"
run() { local tag=$1 pp=$2 exp=$3; shift 3; rm -f $O/$tag.log $O/$tag.out; env RR_PYTHONPATH="$pp" RR_EXPECT_MJW=$exp $R/job.sh hfe_$tag $O/$tag.out -- $PY $R/train_wrap.py 4096 rough train 8 $O/$tag.log - 0.0025 "$@"; echo "$(date +%T) $tag rc $?"; }
echo "V4 start $(date +%T); power: $(pmset -g batt | head -2 | tr '\n' ' '); powermode $(pmset -g | awk '/powermode/ {print $2}')"
for r in 1 2 3; do
  run s1_base_r$r $PWD/$S/mjw_base 93e5ce18 $CS1
  run s1_fix_r$r $PWD/$S/mjw_fix ${FIX:0:8} $CS1
  run s0_base_r$r $PWD/$S/mjw_base 93e5ce18 $CS0
  run s0_fix_r$r $PWD/$S/mjw_fix ${FIX:0:8} $CS0
done
echo "V4 end $(date +%T); power: $(pmset -g batt | head -2 | tr '\n' ' ')"
