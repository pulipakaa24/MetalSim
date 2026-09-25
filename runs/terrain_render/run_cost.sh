#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
python3 scripts/gpu_top.py --interval 1
echo "busy processes (cpu > 40 %):"; ps -eo pid,%cpu,command | awk 'NR > 1 && $2 > 40' | cut -c1-160
.venv/bin/python scripts/diagnostics/terrain_render_cost.py --n 1024 --size 128 --frames 10 --tiers 0,2 2>&1 | grep -v "Module \|Warn"
python3 scripts/gpu_top.py --interval 1
