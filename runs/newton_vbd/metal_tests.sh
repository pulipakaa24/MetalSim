#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
R=".venv-newton152/bin/python scripts/diagnostics/newton_vbd/run_newton_tests.py"
echo "=== Newton 1.5.2 VBD/cloth/softbody tests, metal_0 variants"
$R --device-tag metal_0 newton.tests.test_solver_vbd newton.tests.test_softbody newton.tests.test_cloth newton.tests.test_collision_cloth 2>&1 | grep -E "selected|RESULT|FAILED|\.\.\. (ok|FAIL|ERROR|skipped)"
echo "=== Newton 1.5.2 coupled-solver tests on metal:0 (default device)"
$R --device-tag metal_0 --untagged-on metal:0 newton.tests.test_coupled_solver newton.tests.test_admm_coupled_solver 2>&1 | grep -E "selected|RESULT|FAILED|\.\.\. (FAIL|ERROR)"
echo "=== coupled examples on metal:0 (Newton 1.5.2 + mujoco-warp 3.11.0)"
for E in multiphysics.example_mujoco_vbd_coupled_solver multiphysics.example_mujoco_vbd_admm_solver; do
  .venv-newton152/bin/python scripts/diagnostics/newton_vbd/coupled_probe.py --example $E --device metal:0 --frames 60 --out runs/newton_vbd/coupled_metal_${E##*.}.npz 2>&1 | grep "^{" | cut -c1-3000
done
