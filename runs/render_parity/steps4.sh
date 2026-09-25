#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
for P in tonemap_fv tonemap_fvg tonemap_cal_fv tonemap_cal_fvg; do
  runs/render_parity/run_preset.sh $P --spp 16 --passes 8
done
