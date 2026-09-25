#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
.venv/bin/python scripts/diagnostics/camera_update_profile.py --parts mlx2 --mlx-bound 2>&1 | grep -v gpu_lock
