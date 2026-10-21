#!/bin/bash
# GPU job (render kind): MetalSim with the MuJoCo Warp fork at metalsim f1b35e2 (eq-block fix f52488b), equality
# workaround removed. Stages: capacity + overflow tests, the fork's own tests on Metal, rigid suite, render suite.
WT=.
PY=.venv/bin/python
L=$WT/runs/eqfix
export PYTHONPATH=$WT/scratch/mjw_fork:$WT
cd "$WT"
$PY -c "import warp as wp; wp.config.quiet=True; import mujoco_warp, mujoco, metalsim, subprocess, os
d=lambda m: subprocess.run(['git','-C',os.path.dirname(os.path.dirname(m.__file__)),'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()
print('warp', d(wp), wp.__version__, 'mjw', d(mujoco_warp), mujoco_warp.__file__, 'mujoco', mujoco.__version__, 'metalsim', metalsim.__file__)" > $L/versions.log 2>&1

echo "##### capacity + overflow tests (tests/test_capacity.py holds the overflow-guard tests)" > $L/cap_tests.log
$PY -m pytest -q -p no:cacheprovider tests/test_capacity.py -v >> $L/cap_tests.log 2>&1

echo "##### fork (scratch/mjw_fork @ $(git -C scratch/mjw_fork rev-parse --short HEAD)) on Metal: constraint_test.py" > $L/fork_tests.log
(cd scratch/mjw_fork && $PY -m pytest -q -p no:cacheprovider mujoco_warp/_src/constraint_test.py) >> $L/fork_tests.log 2>&1
echo "##### fork on Metal: forward_test.py + solver_test.py -k 'equality or weld or connect'" >> $L/fork_tests.log
(cd scratch/mjw_fork && $PY -m pytest -q -p no:cacheprovider mujoco_warp/_src/forward_test.py mujoco_warp/_src/solver_test.py \
  -k "equality or weld or connect") >> $L/fork_tests.log 2>&1

echo "##### RIGID SUITE (.venv): tests/ minus test_mpm.py" > $L/rigid_suite.log
$PY -m pytest -q -p no:cacheprovider tests --ignore=tests/test_mpm.py >> $L/rigid_suite.log 2>&1

echo "##### RENDER SUITE (.venv): render, sensor, scene-USD tests" > $L/render_suite.log
$PY -m pytest -q -p no:cacheprovider tests/test_render_*.py tests/test_sensor*.py tests/test_*usd*.py >> $L/render_suite.log 2>&1
echo done
