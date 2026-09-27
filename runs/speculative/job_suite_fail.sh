#!/bin/bash
# list the MJW_SPECULATIVE_GAP=1 suite failures (frozen worktree 27f1fcd)
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate:/Users/aditya/robosim/upstream/mujoco_warp-spec-27f1fcd
cd upstream/mujoco_warp-spec-27f1fcd
PYTHONPATH=$WT MJW_SPECULATIVE_GAP=1 python -m pytest mujoco_warp/_src -q -p no:cacheprovider -rf 2>&1 | grep "^FAILED\|passed\|failed"
