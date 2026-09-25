#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
ps aux | grep -E "python .*(metalsim|diagnostics)" | grep -v grep | grep -v camera_update_profile | awk '{print "other python:", $2, $11, $12, $13, $14}'
.venv/bin/python scripts/diagnostics/camera_update_profile.py --parts update,fast,fastk,update --update-reps 5
