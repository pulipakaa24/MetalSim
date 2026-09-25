#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
.venv/bin/python -m pytest -q -p no:logging --tb=line tests/test_metal_conv.py tests/test_camera_update_fast.py 2>&1 | grep -v "^$\|Not enough SMs" | tail -12
P="scripts/diagnostics/camera_update_profile.py"
.venv/bin/python $P --parts kernels
.venv/bin/python $P --parts fastk --update-reps 5 --no-gather-in-graph --kernels c1_wgrad,c2_dgrad
.venv/bin/python $P --parts fastk --update-reps 5 --no-gather-in-graph --kernels c1_wgrad,c2_dgrad,c2_wgrad
.venv/bin/python $P --parts fastk --update-reps 5 --no-gather-in-graph --kernels c1_wgrad,c2_dgrad,c2_wgrad,c3_wgrad
.venv/bin/python $P --parts fastk --update-reps 5 --no-gather-in-graph --kernels c1_wgrad,c2_dgrad,c2_wgrad,c3_wgrad,c1_fwd
.venv/bin/python $P --parts fastk --update-reps 5 --kernels c1_wgrad,c2_dgrad,c2_wgrad,c3_wgrad,c1_fwd
