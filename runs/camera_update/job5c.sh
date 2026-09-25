#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
P="scripts/diagnostics/camera_update_profile.py"
.venv/bin/python $P --parts fastk --update-reps 6 2>&1 | grep -E "median"
.venv/bin/python $P --parts mlx2 2>&1 | grep -v gpu_lock
.venv/bin/python $P --parts mlx2 --mlx-bound 2>&1 | grep -v gpu_lock
.venv/bin/python $P --parts fastk --update-reps 6 2>&1 | grep -E "median"
