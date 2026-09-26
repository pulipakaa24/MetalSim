#!/bin/bash
# ellip_ws: the fork's suite on Metal from the final frozen worktree mujoco_warp-ellip-test7 (fbf7da8 (433c305 + the test fix)) with the Warp fork 6bceb39d: default knobs, then MJW_LS_NOISE_FLOOR=8.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-ellip-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test7
echo "=== start $(date) frozen warp $(git -C upstream/warp-innate-ellip-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test7 rev-parse --short HEAD)"
cd upstream/mujoco_warp-ellip-test7
echo "--- default knobs $(date +%H:%M:%S)"; PYTHONPATH=$WT python -m pytest mujoco_warp/_src -q -p no:cacheprovider 2>&1 | grep -v '^linesearch\|^To disable\|^solver iterations\|^nvmax' | tail -4
echo "--- MJW_LS_NOISE_FLOOR=8 $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_LS_NOISE_FLOOR=8 python -m pytest mujoco_warp/_src -q -p no:cacheprovider 2>&1 | grep -v '^linesearch\|^To disable\|^solver iterations\|^nvmax' | tail -4
echo "=== end $(date)"
