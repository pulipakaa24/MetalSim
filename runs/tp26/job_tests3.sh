#!/bin/bash
# tp26 tests 3: the fork test suites through the worktrees (= the merged fork heads): the full mujoco_warp/_src suite on
# Metal (last known 1451 passed / 1 pre-existing failure) and Warp's test_metal (cholesky / tile / solve), then MetalSim's
# physics / capacity / policy / G1 tests.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
echo "=== start $(date) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD)"
echo "--- warp test_metal $(date +%H:%M:%S)"; (cd upstream/warp-innate-tp && python -m pytest warp/tests/test_metal.py -q -x --timeout=1200 2>&1 | tail -3)
echo "--- mujoco_warp full suite $(date +%H:%M:%S)"; (cd upstream/mujoco_warp-tp && PYTHONPATH=$WT python -m pytest mujoco_warp/_src -q -p no:cacheprovider 2>&1 | tail -8)
echo "--- metalsim tests $(date +%H:%M:%S)"; PYTHONPATH=$WT python -m pytest tests/test_physics.py tests/test_capacity.py tests/test_warp_policy.py tests/test_ppo_warp_rollout.py tests/test_g1_fast_factorization.py tests/test_g1_task_terms.py -q 2>&1 | tail -3
echo "=== end $(date)"
