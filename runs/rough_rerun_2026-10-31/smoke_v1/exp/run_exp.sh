#!/bin/bash
# Overflow hypothesis test (2026-10-31): are the smoke overflows (s1 config NEFC at it 1; rough_il3 EPA_HORIZON by it 2)
# caused by the qpos0 start (--initial_reset 0)? 8 iterations at 4096 envs each; one job.sh log per run.
cd ~/MetalSim
RR=runs/rough_rerun_2026-10-31; E=$RR/smoke/exp; PY=.venv/bin/python
C232="--reward_cfg rough_isaac --contact_cfg default --base_velocity origin --feet_slide_velocity origin --effort_limit actuator"
S12="--scan_ordering xy --terrain_collision boxes_local --scan_surface exact"
CIL3="--reward_cfg rough_il3 --contact_cfg recommended_pyramidal --solver_cfg isaaclab3_every_substep_cap20 --base_velocity com --feet_slide_velocity com --effort_limit actuator"
run() { local tag=$1 pp=$2 exp=$3; shift 3; rm -f $E/$tag.log; env RR_PYTHONPATH="$pp" RR_EXPECT_MJW=$exp $RR/job.sh exp_$tag $E/$tag.out -- $PY -m metalsim.learn.g1_velocity 4096 rough train 8 $E/$tag.log - 0.0025 "$@"; echo "$tag rc $?"; }
run s1_fix_reset1 "" 93e5ce18 --seed 1 $S12 $C232 --initial_reset 1
run il3_fix_reset1 "" 93e5ce18 --seed 0 $CIL3 --initial_reset 1
run s1_pin_reset0 $RR/code/mjw_8eb51fe 8eb51fe6 --seed 1 $S12 $C232 --initial_reset 0
run il3_pin_reset0 $RR/code/mjw_8eb51fe 8eb51fe6 --seed 0 $CIL3 --initial_reset 0
