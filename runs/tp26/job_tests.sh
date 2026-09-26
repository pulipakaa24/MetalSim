#!/bin/bash
# tp26 tests: the fork worktrees' own test modules for the kernels changed (Warp: tile Cholesky / solve on Metal;
# MuJoCo Warp: smooth (L'DL), forward, solver), plus MetalSim's physics / G1 / capacity tests, all through the worktrees.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
echo "=== start $(date) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD)"
echo "--- warp test_metal (cholesky) $(date +%H:%M:%S)"; (cd upstream/warp-innate-tp && python -m pytest warp/tests/test_metal.py -k "cholesky or tile" -q 2>&1 | tail -4)
echo "--- mujoco_warp smooth / forward / solver tests $(date +%H:%M:%S)"; (cd upstream/mujoco_warp-tp && PYTHONPATH=$WT python -m pytest mujoco_warp/_src/smooth_test.py mujoco_warp/_src/forward_test.py mujoco_warp/_src/solver_test.py -q 2>&1 | tail -6)
echo "--- metalsim tests $(date +%H:%M:%S)"; PYTHONPATH=$WT python -m pytest tests/test_physics.py tests/test_capacity.py tests/test_g1_fast_factorization.py tests/test_g1_parity.py -q 2>&1 | tail -4
echo "=== end $(date)"
