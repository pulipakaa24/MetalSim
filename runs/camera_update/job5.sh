#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
.venv/bin/python -m pytest -q -p no:logging --tb=line tests/test_metal_conv.py tests/test_camera_update_fast.py 2>&1 | grep -v "^$\|Not enough SMs" | tail -6
P="scripts/diagnostics/camera_update_profile.py"
.venv/bin/python $P --parts fastk --update-reps 6 2>&1 | grep -E "ENABLED|gather_in|median"
.venv/bin/python $P --parts mlx2,mlxlayers 2>&1 | grep -v "gpu_lock"
.venv/bin/python $P --parts fastk --update-reps 6 2>&1 | grep -E "ENABLED|gather_in|median"
