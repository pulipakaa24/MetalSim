#!/bin/bash
# ellip_ws: the fork's test suite on Metal from the frozen worktree mujoco_warp-ellip-test (default knobs: fused launches on, extrapolated warm start off).
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test
echo "=== start $(date) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test rev-parse --short HEAD)"
cd upstream/mujoco_warp-ellip-test && PYTHONPATH=$WT python -m pytest mujoco_warp/_src -q -p no:cacheprovider 2>&1 | grep -v '^linesearch\|^To disable\|^solver iterations' | tail -15
echo "=== end $(date)"
