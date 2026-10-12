#!/bin/bash
# parity.sh MODE -- parity_report.py of runs/mpm/fm_MODE against the L4 suite
W=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
cd $W
export PYTHONPATH=$W
exec /Users/aditya/robosim/.venv-newton152/bin/python -m metalsim.physics.mpm.parity_report \
    /Users/aditya/robosim/runs/parity/isaaclab3_mpm/mpm $W/runs/mpm/fm_$1 \
    --json $W/runs/mpm/approx/parity_$1.json --md $W/runs/mpm/approx/parity_$1.md
