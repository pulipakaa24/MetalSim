#!/bin/bash
# parity.sh MODE -- parity_report.py of runs/mpm/fm_MODE against the L4 suite
W=.
cd $W
export PYTHONPATH=$W
exec .venv-newton152/bin/python -m metalsim.physics.mpm.parity_report \
    runs/parity/isaaclab3_mpm/mpm $W/runs/mpm/fm_$1 \
    --json $W/runs/mpm/approx/parity_$1.json --md $W/runs/mpm/approx/parity_$1.md
