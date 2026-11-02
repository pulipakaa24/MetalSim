#!/bin/bash
# hfield-epa validation V1 (timing class): 4096-env x 8-iteration training smokes, the rough-rerun smoke's commands
# (train_wrap.py: EPA_HORIZON reported, not fatal), interleaved across MuJoCo Warp copies: pinned 8eb51fe (plane path),
# 93e5ce1 (heightfield fix, PR #86), the hfield-epa fix. rough_il3 x 2 rounds, then the 2.3.2 seed-0 config (hfield, no gap).
# Per-iteration time = from the cumulative sps lines (iterations 4-8). Not a timing machine: ratios only.
cd ~/MetalSim
R=runs/rough_rerun_2026-10-31; S=runs/readme_media/scratch/hfield-epa; O=$S/val; PY=.venv/bin/python
FIX=$(cat $S/mjw_fix/COMMIT)
CIL3="--seed 0 --reward_cfg rough_il3 --contact_cfg recommended_pyramidal --solver_cfg isaaclab3_every_substep_cap20 --base_velocity com --feet_slide_velocity com --effort_limit actuator --initial_reset 1"
CS0="--seed 0 --scan_ordering ij --terrain_collision hfield --scan_surface grid --reward_cfg rough_isaac --contact_cfg default --base_velocity origin --feet_slide_velocity origin --initial_reset 1 --effort_limit actuator"
run() { local tag=$1 pp=$2 exp=$3; shift 3; rm -f $O/$tag.log $O/$tag.out; env RR_PYTHONPATH="$pp" RR_EXPECT_MJW=$exp $R/job.sh hfe_$tag $O/$tag.out -- $PY $R/train_wrap.py 4096 rough train 8 $O/$tag.log - 0.0025 "$@"; echo "$(date +%T) $tag rc $?"; }
echo "V1 start $(date +%T); power: $(pmset -g batt | head -2 | tr '\n' ' '); powermode $(pmset -g | awk '/powermode/ {print $2}')"
for r in 1 2; do
  run il3_fix_r$r $PWD/$S/mjw_fix ${FIX:0:8} $CIL3
  run il3_base_r$r $PWD/$S/mjw_base 93e5ce18 $CIL3
  run il3_pin_r$r $PWD/$R/code/mjw_8eb51fe 8eb51fe6 $CIL3
done
run s0_fix $PWD/$S/mjw_fix ${FIX:0:8} $CS0
run s0_base $PWD/$S/mjw_base 93e5ce18 $CS0
echo "V1 end $(date +%T); power: $(pmset -g batt | head -2 | tr '\n' ' ')"
