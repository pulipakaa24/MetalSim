#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
for P in tonemap_fvg tonemap_cal_fvg atrous_cal_fvg oidn_cal_fvg; do
  for K in rt pt; do
    .venv/bin/python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/$K --out runs/render_parity/${K}_${P}_clip --preset $P --spp 16 --passes 8 2>&1 | grep -v Warning | tail -3
  done
done
# denoisers at 16 spp x 4 vs 256 spp ground truth, final material/light/tone model
for K in rt pt; do
  EV=1; [ $K = rt ] && EV=2
  .venv/bin/python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/$K --out runs/render_parity/dn2_${K}_gt256 --preset tonemap_cal_fvg --spp 16 --passes 16 --every $EV --no_metrics 2>&1 | tail -1
  for P in tonemap_cal_fvg atrous_cal_fvg oidn_cal_fvg; do
    .venv/bin/python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/$K --out runs/render_parity/dn2_${K}_${P}_16x4 --preset $P --spp 16 --passes 4 --every $EV 2>&1 | grep -v Warning | tail -1
  done
done
