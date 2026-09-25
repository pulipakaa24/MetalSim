#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
for P in legacy brdf lights tonemap tonemap_cal; do
  runs/render_parity/run_preset.sh $P --spp 16 --passes 8
done
