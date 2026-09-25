#!/bin/bash
# threshold defaults (48, 32) vs the previous (40, 64): correctness on every scene (throughput from the tp runs below),
# correctness part
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
python3 scripts/gpu_top.py --interval 1
echo "busy processes (cpu > 40 %):"; ps -eo pid,%cpu,command | awk 'NR > 1 && $2 > 40' | cut -c1-160
.venv/bin/python scripts/diagnostics/fast_factorization_scenes.py old --out runs/fastfact/old3.npz --n 64 --ntp 256 2>&1 | grep -v "Module \|Warn\|warn"
.venv/bin/python scripts/diagnostics/fast_factorization_scenes.py thr --out runs/fastfact/thr3.npz --ref runs/fastfact/old3.npz --n 64 --ntp 256 2>&1 | grep -v "Module \|Warn\|warn"
