#!/bin/bash
# tp26: MetalSim's whole test suite through the fork worktrees (= the merged heads), render class.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD)"
PYTHONPATH=$WT python -m pytest tests -q -x -p no:cacheprovider 2>&1 | tail -6
echo "=== end $(date)"
