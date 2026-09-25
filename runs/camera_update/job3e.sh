#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
.venv/bin/python -m pytest -q -s -p no:logging --tb=line tests/test_metal_conv.py tests/test_camera_update_fast.py 2>&1 | grep -v "^$\|Not enough SMs" | tail -24
.venv/bin/python scripts/diagnostics/camera_update_profile.py --parts c1var,fastsplit,fast,fastk --update-reps 5
