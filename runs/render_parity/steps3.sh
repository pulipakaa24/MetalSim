#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
for P in brdf_v1 brdf lights tonemap tonemap_cal_v1 tonemap_cal oidn_cal atrous_cal; do
  runs/render_parity/run_preset.sh $P --spp 16 --passes 8
done
.venv/bin/python -m metalsim.render.bench_cost --out runs/render_parity/bench_cost.json --presets legacy,tonemap_cal,atrous_cal,oidn_cal 2>&1 | grep -v Warning | grep -v Module
