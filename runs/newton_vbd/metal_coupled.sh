#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
echo "=== coupled examples on metal:0 (Newton 1.5.2 + mujoco-warp 3.11.0; one eager frame before capture)"
for E in multiphysics.example_mujoco_vbd_coupled_solver multiphysics.example_mujoco_vbd_admm_solver; do
  .venv-newton152/bin/python scripts/diagnostics/newton_vbd/coupled_probe.py --example $E --device metal:0 --frames 60 --out runs/newton_vbd/coupled_metal_${E##*.}.npz 2>&1 | grep "^{" | cut -c1-3000
done
echo "=== collision_cloth tests needing usd-core, metal_0"
.venv-newton152/bin/python scripts/diagnostics/newton_vbd/run_newton_tests.py --device-tag metal_0 newton.tests.test_collision_cloth 2>&1 | grep -E "selected|RESULT|FAILED"
echo "=== single coupled test on metal"
.venv-newton152/bin/python scripts/diagnostics/newton_vbd/run_newton_tests.py --device-tag metal_0 --untagged-on metal:0 -k test_compacted_joint_targets_use_local_layout newton.tests.test_coupled_solver 2>&1 | grep -E "selected|RESULT|FAILED|Error" | head
echo "=== soft cube throughput, tile off, 256/1024 worlds"
PYTHONPATH=upstream/newton-1.5.2 .venv-newtonfork/bin/python scripts/diagnostics/newton_vbd/vbd_probe.py --device metal:0 --scenes cube --steps 30 --capture --worlds 256 1024 --no-tile 2>&1 | grep "^{" | cut -c1-700
