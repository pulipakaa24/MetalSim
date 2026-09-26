#!/bin/bash
# tp26 suites on frozen worktrees (mujoco_warp 0a9de8e, warp 9df9acee; nothing edits them while jobs run): MetalSim's
# whole suite (no -x) and the full mujoco_warp suite with logs kept.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-tp-test
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp-test rev-parse --short HEAD)"
PYTHONPATH=$WT python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__)"
echo "--- metalsim suite $(date +%H:%M:%S)"; PYTHONPATH=$WT python -m pytest tests -q -p no:cacheprovider > runs/tp26/metalsim_suite.log 2>&1; grep "FAILED\|passed\|failed\|error" runs/tp26/metalsim_suite.log | tail -8
echo "--- mujoco_warp full suite $(date +%H:%M:%S)"; (cd upstream/mujoco_warp-tp-test && PYTHONPATH=$WT python -m pytest mujoco_warp/_src -q -p no:cacheprovider > /Users/aditya/robosim/runs/tp26/fork_suite2.log 2>&1; grep "FAILED\|passed\|failed" /Users/aditya/robosim/runs/tp26/fork_suite2.log | tail -10)
echo "--- warp test_metal $(date +%H:%M:%S)"; (cd upstream/warp-innate-tp-test && python -m pytest warp/tests/test_metal.py -q -p no:cacheprovider > /Users/aditya/robosim/runs/tp26/warp_test_metal2.log 2>&1; tail -2 /Users/aditya/robosim/runs/tp26/warp_test_metal2.log)
echo "=== end $(date)"
