#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
.venv/bin/python scripts/diagnostics/camera_update_profile.py --parts c2parts,interleave --update-reps 6
