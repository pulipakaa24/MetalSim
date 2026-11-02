#!/bin/bash
# GPU batch B: fork collision tests on Metal (fix: three files; base: -k hfield), then G1 rough-task env step at 4096 envs
# (python -m metalsim.learn.g1_velocity 4096 rough: random actions, synchronized env-steps/s), base / fix interleaved x2.
R=~/MetalSim; S=$R/runs/readme_media/scratch/hfield-contacts; PY=$R/.venv/bin/python; cd $R
pw() { echo "power $(pmset -g batt | sed -n 2p | cut -c1-70) | $(pmset -g | grep powermode | tr -s ' ') | $(date +%H:%M:%S)"; }
which_mjw() { PYTHONPATH=$1 $PY -c "import mujoco_warp; print('mujoco_warp', mujoco_warp.__file__)"; }
echo "== $(pw)"
echo "== fork tests on Metal (fix)"; (cd $S/mjw && which_mjw $S/mjw && $PY -m pytest -q mujoco_warp/_src/collision_driver_test.py mujoco_warp/_src/collision_gjk_test.py mujoco_warp/_src/collision_primitive_core_test.py 2>&1 | tail -3)
echo "== fork -k hfield on Metal (base)"; (cd $S/mjw_base && which_mjw $S/mjw_base && $PY -m pytest -q mujoco_warp/_src/collision_driver_test.py -k hfield 2>&1 | tail -2)
for rep in 1 2; do for t in base fix; do
  W=$S/mjw; [ $t = base ] && W=$S/mjw_base
  echo "== timing $t rep $rep $(pw)"; which_mjw $W
  PYTHONPATH=$W $PY -m metalsim.learn.g1_velocity 4096 rough 2>&1 | grep -E "^G1|^N=|Error"
done; done
for t in base fix; do
  W=$S/mjw; [ $t = base ] && W=$S/mjw_base
  echo "== random actions $t $(pw)"; PYTHONPATH=$W $PY $S/gpu/random_actions.py 2>&1 | grep -E "^mujoco_warp|^400|Error|error"
done
echo "== done $(pw)"
