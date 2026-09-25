#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
for SRC in upstream/newton-1.5.2 upstream/newton; do
  echo "=== $SRC tile on"
  PYTHONPATH=$SRC .venv-newtonfork/bin/python scripts/diagnostics/newton_vbd/vbd_probe.py --device metal:0 --steps 40 2>&1 | grep -v "^Module\|took" | tail -60
done
