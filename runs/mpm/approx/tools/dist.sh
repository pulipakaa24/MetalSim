#!/bin/bash
# dist.sh OUT_PREFIX METAL_DIR...  -- floors_dist.py of Metal recordings against the L4 floor ensemble
W=.
L=runs/parity/isaaclab3_mpm_floors
OUT=$1; shift
cd $W
export PYTHONPATH=$W
exec .venv-newton152/bin/python -m metalsim.physics.mpm.floors_dist $L/mpm_floors $L/mpm_floors2 \
    --metal "$@" --md $OUT.md --json $OUT.json
