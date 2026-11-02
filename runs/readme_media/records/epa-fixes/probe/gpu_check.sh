#!/bin/bash
# epa-fixes Metal check (check class): the fork's three collision suites and MetalSim's tests/test_terrain.py on Metal,
# both against the git archive of the new tip in epa-fixes/v_new (COMMIT holds the sha).
# Queue: cd ~/MetalSim && scripts/gpu_run.sh epa_fixes_check check 25 -- bash runs/readme_media/scratch/epa-fixes/probe/gpu_check.sh
cd ~/MetalSim
S=runs/readme_media/scratch/epa-fixes; O=$S/logs/gpu_check.log; PY=$PWD/.venv/bin/python; T=$PWD/$S/v_new
{
echo "epa_fixes_check start $(date +%T); tree $(cat $T/COMMIT)"
echo "power: $(pmset -g batt | head -2 | tr '\n' ' '); $(pmset -g | grep -i powermode | tr -s ' ')"
echo "== fork collision suites on Metal (default device)"
(cd $T && PYTHONPATH=$T $PY -c "import warp as wp, mujoco_warp; print(mujoco_warp.__file__, 'device', wp.get_device())" 2>&1 | tail -1 \
  && PYTHONPATH=$T $PY -m pytest -q mujoco_warp/_src/collision_driver_test.py mujoco_warp/_src/collision_gjk_test.py \
     mujoco_warp/_src/collision_primitive_core_test.py -p no:cacheprovider 2>&1 | grep -E 'passed|failed|error|^FAILED' | tail -8)
echo "== MetalSim tests/test_terrain.py (Metal)"
PYTHONPATH=$T $PY -c "import mujoco_warp; print(mujoco_warp.__file__)"
PYTHONPATH=$T $PY -m pytest -q -s tests/test_terrain.py -p no:cacheprovider 2>&1 | grep -E 'passed|failed|error|^FAILED|min normal|pelvis|penetration' | tail -12
echo "epa_fixes_check end $(date +%T)"
} > $O 2>&1
grep -q " failed\|^FAILED" $O && exit 1 || exit 0
