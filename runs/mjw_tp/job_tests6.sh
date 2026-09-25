#!/bin/bash
# tests after adopting steps 1+2 (and the MuJoCo Warp fork's serial sparse L'DL + no-op launch skips)
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
echo "=== G1 parity / task terms / physics / fast factorization"
python -m pytest tests/test_g1_parity.py tests/test_g1_task_terms.py tests/test_physics.py tests/test_g1_fast_factorization.py -q 2>&1 | tail -15
echo "=== Warp fork: Metal tile Cholesky tests"
(cd upstream/warp-innate && python -m pytest warp/tests/test_metal.py -k cholesky -q 2>&1 | tail -5)
echo "=== MuJoCo Warp fork: smooth / passive / derivative / forward tests (Metal default device)"
(cd upstream/mujoco_warp && timeout 1500 python -m pytest mujoco_warp/_src/smooth_test.py mujoco_warp/_src/passive_test.py mujoco_warp/_src/derivative_test.py mujoco_warp/_src/forward_test.py -q -x 2>&1 | tail -15)
echo "=== end $(date)"
