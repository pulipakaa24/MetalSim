#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python -m pytest tests/test_render_terrain_slots.py tests/test_render_tier0.py -q -s -x 2>&1 | grep -v "Module \|Warning"
