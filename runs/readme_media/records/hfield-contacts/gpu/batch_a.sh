#!/bin/bash
# GPU batch A (one queue job): fork collision tests on Metal (fix; -k hfield also on base), MetalSim tests/test_terrain.py
# (fix: whole file; base: the two contact tests), G1 falls (fix: seeds 0-4 x 3; base: seeds 0-4 x 2 more).
R=~/MetalSim; S=$R/runs/readme_media/scratch/hfield-contacts; O=$S/gpu; cd $R
echo "== power $(pmset -g batt | sed -n 2p | cut -c1-60) | $(pmset -g | grep powermode) | $(date +%H:%M:%S)"
which_mjw() { PYTHONPATH=$1 .venv/bin/python -c "import mujoco_warp; print('mujoco_warp', mujoco_warp.__file__)"; }
echo "== fork tests on Metal (fix)"; (cd $S/mjw && which_mjw $S/mjw && $R/.venv/bin/python -m pytest -q mujoco_warp/_src/collision_driver_test.py mujoco_warp/_src/collision_gjk_test.py mujoco_warp/_src/collision_primitive_core_test.py 2>&1 | tail -3)
echo "== fork -k hfield on Metal (base)"; (cd $S/mjw_base && which_mjw $S/mjw_base && $R/.venv/bin/python -m pytest -q mujoco_warp/_src/collision_driver_test.py -k hfield 2>&1 | tail -2)
echo "== test_terrain.py (fix)"; which_mjw $S/mjw; PYTHONPATH=$S/mjw .venv/bin/python -m pytest -q -s tests/test_terrain.py 2>&1 | grep -E "passed|failed|hfield contacts|step edge|Error|assert" | tail -12
echo "== test_terrain.py contact tests (base)"; which_mjw $S/mjw_base; PYTHONPATH=$S/mjw_base .venv/bin/python -m pytest -q -s tests/test_terrain.py -k "hfield_mesh_contacts or step_edge" 2>&1 | grep -E "passed|failed|hfield contacts|step edge|Error|assert" | tail -8
echo "== falls (fix) $(date +%H:%M:%S)"; $S/falls/repeat.sh $S/mjw fix 3 0 1 2 3 4 2>&1 | grep "^\[gpu"
echo "== falls (base) $(date +%H:%M:%S)"; $S/falls/repeat.sh $S/mjw_base baseall 2 0 1 2 3 4 2>&1 | grep "^\[gpu"
echo "== done $(date +%H:%M:%S) power $(pmset -g batt | sed -n 2p | cut -c1-60)"
