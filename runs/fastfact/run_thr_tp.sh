#!/bin/bash
# threshold defaults (48, 32) vs the previous (40, 64): throughput alternating old, thr, old, thr (go1 in the first pair only)
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
python3 scripts/gpu_top.py --interval 1
echo "busy processes (cpu > 40 %):"; ps -eo pid,%cpu,command | awk 'NR > 1 && $2 > 40' | cut -c1-160
for c in old thr; do
  .venv/bin/python scripts/diagnostics/fast_factorization_scenes.py $c --tp_only --out runs/fastfact/tp_thr.jsonl --ntp 4096 2>&1 | grep "^tp "
done
for c in old thr; do
  .venv/bin/python scripts/diagnostics/fast_factorization_scenes.py $c --tp_only --out runs/fastfact/tp_thr.jsonl --ntp 4096 \
    --scenes cartpole_rgb,cartpole_state,so101_lift,lidar_nav,tron1_wf,tron1_sim,panda 2>&1 | grep "^tp "
done
python3 scripts/gpu_top.py --interval 1
