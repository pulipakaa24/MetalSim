#!/bin/bash
# hfield-epa validation V2 (check class): the fix's tests on Metal, MetalSim's terrain tests, the rough-transfer protocol
# (full) and the EPA_HORIZON capture hook over 12 rough_il3 iterations, all with the hfield-epa fix (mjw_fix archive).
cd ~/MetalSim
R=runs/rough_rerun_2026-10-31; S=runs/readme_media/scratch/hfield-epa; O=$S/val; PY=$PWD/.venv/bin/python
FIX=$(cat $S/mjw_fix/COMMIT)
echo "V2 start $(date +%T); fix $FIX; power: $(pmset -g batt | head -2 | tr '\n' ' ')"
echo "== fork collision suites on Metal"
(cd $S/mjw_fix && PYTHONPATH=$PWD $PY -c "import mujoco_warp; print(mujoco_warp.__file__)" && PYTHONPATH=$PWD $PY -m pytest -q mujoco_warp/_src/collision_driver_test.py mujoco_warp/_src/collision_gjk_test.py mujoco_warp/_src/collision_primitive_core_test.py -p no:cacheprovider 2>&1 | tail -4)
echo "== MetalSim tests/test_terrain.py (Metal)"
PYTHONPATH=$PWD/$S/mjw_fix $PY -c "import mujoco_warp; print(mujoco_warp.__file__)"
PYTHONPATH=$PWD/$S/mjw_fix $PY -m pytest -q -s tests/test_terrain.py -p no:cacheprovider 2>&1 | grep -E 'passed|failed|error|min normal|pelvis|penetration' | tail -8
echo "== rough-transfer protocol (full) on the fix"
env RR_PYTHONPATH=$PWD/$S/mjw_fix RR_EXPECT_MJW=${FIX:0:8} $R/job.sh hfe_transfer $O/transfer_fix.out -- bash $R/jobs/transfer.sh $O/transfer fix full metalsim_ours_it1500=$R/ckpt_old/g1_rough_ppowarp_fixed_it1500.pt nvidia_pretrained_rough_it2999=assets/isaac/pretrained/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt
grep -h 'overflow flags at exit\|^--- rc' $O/transfer_fix.out
echo "== EPA_HORIZON capture hook, rough_il3 12 iterations x 4096, fix"
PYTHONPATH=$PWD/$S/mjw_fix $PY $S/probe/capture_metal.py $S/cap_fix 12 4096 > $O/capture_fix.log 2>&1; grep -h 'captured\|flags\|mujoco_warp ' $O/capture_fix.log
echo "V2 end $(date +%T)"
