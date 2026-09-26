#!/bin/bash
# tp26 tests 2: MetalSim tests with the installed forks after the G1 flat task moved to the auto capacity bound.
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date) metalsim $(git rev-parse --short HEAD)"
python -m pytest tests/test_warp_policy.py tests/test_ppo_warp_rollout.py tests/test_capacity.py tests/test_g1_task_terms.py tests/test_g1_parity.py tests/test_physics.py tests/test_g1_fast_factorization.py -q 2>&1 | tail -5
echo "=== end $(date)"
