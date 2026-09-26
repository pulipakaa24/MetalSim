#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
echo "busy processes (cpu > 40 %):"; ps -eo pid,%cpu,command | awk 'NR > 1 && $2 > 40' | cut -c1-160
P=".venv/bin/python scripts/diagnostics/terrain_render_cost.py --n 1024 --size 128 --frames 10 --out runs/terrain_render/cull"
$P --tiers 0 --configs before,slots,after,after_full 2>&1 | grep -v "Module \|Warn\|warn"
$P --tiers 2 --configs hfield,after 2>&1 | grep -v "Module \|Warn\|warn"
$P --tiers 2 --configs hfield,after --coarse 4 --tag _coarse4 2>&1 | grep -v "Module \|Warn\|warn"
