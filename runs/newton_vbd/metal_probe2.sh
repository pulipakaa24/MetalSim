#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
echo "=== warp test_metal nextafterf"
.venv-newtonfork/bin/python -m pytest -q upstream/warp-innate/warp/tests/test_metal.py -k "nextafter" 2>&1 | tail -5
for S in newton-1.5.2 newton; do
  echo "=== $S metal 200 steps eager"
  PYTHONPATH=upstream/$S .venv-newtonfork/bin/python scripts/diagnostics/newton_vbd/vbd_probe.py --device metal:0 --steps 200 --out runs/newton_vbd/metal_${S}.npz 2>&1 | grep "^{" | cut -c1-3000
done
