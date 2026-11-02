#!/bin/bash
# hfield-epa validation V3 (timing class): the 2.3.2 configs' 4096-env x 8-iteration smokes (rough-rerun commands,
# --initial_reset 1) interleaved across pinned 8eb51fe, 93e5ce1 and the hfield-epa fix: seed-0 config (heightfield
# everywhere, no gap) x 2 rounds, seed-1 config (boxes_local + heightfield, no gap) x 1 round. Ratios only.
cd ~/MetalSim
R=runs/rough_rerun_2026-10-31; S=runs/readme_media/scratch/hfield-epa; O=$S/val2; PY=.venv/bin/python
mkdir -p $O; FIX=$(cat $S/mjw_fix/COMMIT)
C232="--reward_cfg rough_isaac --contact_cfg default --base_velocity origin --feet_slide_velocity origin --effort_limit actuator --initial_reset 1"
CS0="--seed 0 --scan_ordering ij --terrain_collision hfield --scan_surface grid $C232"
CS1="--seed 1 --scan_ordering xy --terrain_collision boxes_local --scan_surface exact $C232"
run() { local tag=$1 pp=$2 exp=$3; shift 3; rm -f $O/$tag.log $O/$tag.out; env RR_PYTHONPATH="$pp" RR_EXPECT_MJW=$exp $R/job.sh hfe_$tag $O/$tag.out -- $PY $R/train_wrap.py 4096 rough train 8 $O/$tag.log - 0.0025 "$@"; echo "$(date +%T) $tag rc $?"; }
echo "V3 start $(date +%T); power: $(pmset -g batt | head -2 | tr '\n' ' '); powermode $(pmset -g | awk '/powermode/ {print $2}')"
for r in 1 2; do
  run s0_fix_r$r $PWD/$S/mjw_fix ${FIX:0:8} $CS0
  run s0_base_r$r $PWD/$S/mjw_base 93e5ce18 $CS0
  run s0_pin_r$r $PWD/$R/code/mjw_8eb51fe 8eb51fe6 $CS0
done
run s1_fix $PWD/$S/mjw_fix ${FIX:0:8} $CS1
run s1_base $PWD/$S/mjw_base 93e5ce18 $CS1
run s1_pin $PWD/$R/code/mjw_8eb51fe 8eb51fe6 $CS1
echo "V3 end $(date +%T); power: $(pmset -g batt | head -2 | tr '\n' ' ')"
