#!/bin/bash
# GPU job: the fork's forward_test.py and solver_test.py in full on Metal (they include the weld / connect models of
# SolverFastPathTest; a -k "equality or weld or connect" filter selects no test names), at f1b35e2.
WT=/Users/aditya/robosim/.claude/worktrees/agent-a4b4d2a53213dae91
PY=/Users/aditya/robosim/.venv/bin/python
cd "$WT/scratch/mjw_fork"
echo "##### fork @ $(git rev-parse --short HEAD) on Metal: forward_test.py + solver_test.py (full)"
$PY -c "import mujoco_warp; print(mujoco_warp.__file__)"
$PY -m pytest -q -p no:cacheprovider mujoco_warp/_src/forward_test.py mujoco_warp/_src/solver_test.py 2>&1 | grep -v "^Module .* load on device"
