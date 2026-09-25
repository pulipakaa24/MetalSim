#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
.venv/bin/python -m pytest -q -x tests/test_metal_conv.py tests/test_camera_update_fast.py 2>&1 | tail -25
.venv/bin/python scripts/diagnostics/camera_update_profile.py --parts kernels,update,fast,fastk
