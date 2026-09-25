#!/bin/bash
cd /Users/aditya/robosim
.venv/bin/python scripts/diagnostics/camera_update_profile.py --parts prep,updvar
TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0 .venv/bin/python scripts/diagnostics/camera_update_profile.py --parts updvar
