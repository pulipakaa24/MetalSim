#!/bin/bash
# throughput only, all scenes, alternating configurations (old, new, old, new); busy-process check first
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
echo "--- other busy processes (cpu > 20 %):"; ps -Ao pid,pcpu,etime,command | awk 'NR==1 || $2 > 20' | cut -c1-160
for c in old new; do
  .venv/bin/python scripts/diagnostics/fast_factorization_scenes.py $c --tp_only --out runs/fastfact/tp.jsonl --ntp 4096 2>&1 | grep "^tp "
done
# second pair without go1 (3.7 K env-steps/s: 2.8 min per measurement)
for c in old new; do
  .venv/bin/python scripts/diagnostics/fast_factorization_scenes.py $c --tp_only --out runs/fastfact/tp.jsonl --ntp 4096 \
    --scenes cartpole_rgb,cartpole_state,so101_lift,lidar_nav,tron1_wf,tron1_sim,panda 2>&1 | grep "^tp "
done
echo "--- other busy processes at the end:"; ps -Ao pid,pcpu,etime,command | awk 'NR==1 || $2 > 20' | cut -c1-160
