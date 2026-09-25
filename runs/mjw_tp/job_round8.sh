#!/bin/bash
# round 8: profile after step 4, dispatch-cost calibration, MetalSim test suite, 20-iteration probe of steps 3 and 4
cd /Users/aditya/robosim && source .venv/bin/activate
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
echo "=== dispatch cost"
python scripts/diagnostics/metal_dispatch_cost.py 2>&1 | grep "dim"
python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
WP_METAL_PROFILE=1 python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --kernels 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
echo "=== end $(date)"
} > runs/mjw_tp/profile8.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
python -m pytest tests/test_g1_parity.py tests/test_g1_task_terms.py tests/test_physics.py tests/test_g1_fast_factorization.py -q 2>&1 | tail -6
for F in 0 1; do
  MJW_METAL_FUSE_H_CHOLESKY=$F MJW_TP_VARIANT="{\"label\":\"fuse $F\"}" python scripts/diagnostics/g1_tp_train_probe.py 4096 20 2>&1 | grep "^variant\|^it \|Error\|Traceback"
done
echo "=== end $(date)"
} > runs/mjw_tp/tests8.log 2>&1
