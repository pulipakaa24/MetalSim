#!/bin/bash
# round 7: step 4 candidate = MuJoCo Warp fork: incremental Hessian update fused into the register Cholesky on Metal
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
MJW_METAL_FUSE_H_CHOLESKY=0 MJW_TP_VARIANT='{"label":"default, fusion off (steps 1-3)"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_TP_VARIANT='{"label":"step4 fused H update + Cholesky"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_METAL_FUSE_H_CHOLESKY=0 MJW_TP_VARIANT='{"label":"fusion off repeat"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_TP_VARIANT='{"label":"step4 repeat"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
echo "=== end $(date)"
} > runs/mjw_tp/costsplit7.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
MJW_METAL_FUSE_H_CHOLESKY=0 MJW_TP_VARIANT='{"label":"fusion off"}' python scripts/diagnostics/g1_tp_check.py 512 50 --out runs/mjw_tp/traj_step3.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
MJW_TP_VARIANT='{"label":"fused"}' python scripts/diagnostics/g1_tp_check.py 512 50 --ref runs/mjw_tp/traj_step3.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
echo "=== pytest"
python -m pytest tests/test_g1_parity.py tests/test_g1_task_terms.py tests/test_physics.py tests/test_g1_fast_factorization.py -q 2>&1 | tail -3
echo "=== MuJoCo Warp fork tests (smooth / passive / derivative / solver / forward; Metal default device)"
(cd upstream/mujoco_warp && python -m pytest mujoco_warp/_src/smooth_test.py mujoco_warp/_src/passive_test.py mujoco_warp/_src/derivative_test.py mujoco_warp/_src/solver_test.py mujoco_warp/_src/forward_test.py -q 2>&1 | tail -25)
echo "=== end $(date)"
} > runs/mjw_tp/checks7.log 2>&1
