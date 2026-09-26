#!/bin/bash
# tp26 round 2b: (a) residency of the n = 43 register Cholesky: the tile path vs a native snippet with 1 / 2 / 4 / 8 worlds
# per threadgroup (bitwise check against the tile path); (b) the unrolled L'DL solve redesign (distributed x, level-
# interleaved): bitwise check vs the serial kernels and the isolated bench.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
echo "=== start $(date) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- residency n=43 $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/metal_cholesky_residency.py 43 2>&1 | grep "^n=\|Error\|Traceback" | cut -c1-160
echo "--- residency n=32 $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/metal_cholesky_residency.py 32 2>&1 | grep "^n=\|Error\|Traceback" | cut -c1-160
echo "--- L'DL check $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/ldl_lanes_check.py 64 2>&1 | grep "unrolled\|ALL OK\|FAIL\|Error\|Traceback"
echo "--- L'DL bench $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/ldl_lanes_bench.py 4096 2>&1 | grep "serial\|unrolled\|Error\|Traceback"
echo "=== end $(date) power: $(pmset -g batt | head -1)"
