#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
P="scripts/diagnostics/camera_update_profile.py"
for round in 1 2; do
.venv/bin/python $P --parts fastk --update-reps 6 --no-gather-in-graph --kernels c1_wgrad,c2_dgrad 2>&1 | grep -E "ENABLED|gather_in|median"
.venv/bin/python $P --parts fastk --update-reps 6 --no-gather-in-graph --kernels c1_wgrad,c2_dgrad,c1_fwd 2>&1 | grep -E "ENABLED|gather_in|median"
.venv/bin/python $P --parts fastk --update-reps 6 --kernels c1_wgrad,c2_dgrad,c1_fwd 2>&1 | grep -E "ENABLED|gather_in|median"
.venv/bin/python $P --parts fastk --update-reps 6 --kernels c1_wgrad,c2_dgrad,c1_fwd,c3_wgrad 2>&1 | grep -E "ENABLED|gather_in|median"
done
