#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status | head -1
.venv/bin/python /private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/equiv.py 2>&1 | grep -v Warn
.venv/bin/python scripts/diagnostics/camera_update_profile.py --parts c1var,optvar
