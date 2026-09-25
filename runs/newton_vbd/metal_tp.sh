#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
export PYTHONPATH=upstream/newton-1.5.2
P=".venv-newtonfork/bin/python scripts/diagnostics/newton_vbd/vbd_probe.py --device metal:0"
echo "=== graph replay, 1 world, 200 steps (vs eager final)"
$P --steps 200 --capture --out runs/newton_vbd/metal_graph_152.npz 2>&1 | grep "^{" | cut -c1-2000
echo "=== throughput, graph replay, 60 steps"
$P --steps 60 --capture --worlds 64 256 1024 4096 2>&1 | grep "^{" | cut -c1-2000
