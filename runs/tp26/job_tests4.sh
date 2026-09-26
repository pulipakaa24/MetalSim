#!/bin/bash
# tp26 tests 4: the full mujoco_warp suite with the log kept (runs/tp26/fork_suite.log) and Warp's test_metal, worktrees.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
echo "=== start $(date) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD)"
echo "--- warp test_metal $(date +%H:%M:%S)"; (cd upstream/warp-innate-tp && python -m pytest warp/tests/test_metal.py -q -p no:cacheprovider > /Users/aditya/robosim/runs/tp26/warp_test_metal.log 2>&1; tail -3 /Users/aditya/robosim/runs/tp26/warp_test_metal.log)
echo "--- mujoco_warp full suite $(date +%H:%M:%S)"; (cd upstream/mujoco_warp-tp && PYTHONPATH=$WT python -m pytest mujoco_warp/_src -q -p no:cacheprovider > /Users/aditya/robosim/runs/tp26/fork_suite.log 2>&1; grep "FAILED\|passed\|failed" /Users/aditya/robosim/runs/tp26/fork_suite.log | tail -12)
echo "=== end $(date)"
