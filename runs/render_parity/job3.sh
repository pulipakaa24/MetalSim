#!/bin/bash
# denoiser evaluation: 16 spp x 4 passes (none / atrous / oidn) vs 256 spp converged, and vs RTX; then cost bench
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
BEST=${BEST:-tonemap_cal}
.venv/bin/python -m pytest tests/test_render_tier2.py -q -s -k "rtx_tonemap" 2>&1 | grep -E "mean|passed|failed|Error"
for K in rt pt; do
  EV=1; [ $K = rt ] && EV=2
  .venv/bin/python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/$K --out runs/render_parity/dn_${K}_gt256 --preset $BEST --spp 16 --passes 16 --every $EV 2>&1 | grep -v Warning | tail -40
  for P in $BEST ${BEST/tonemap/atrous} ${BEST/tonemap/oidn}; do
    .venv/bin/python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/$K --out runs/render_parity/dn_${K}_${P}_16x4 --preset $P --spp 16 --passes 4 --every $EV 2>&1 | grep -v Warning | tail -40
  done
done
.venv/bin/python -m metalsim.render.bench_cost --out runs/render_parity/bench_cost.json 2>&1 | grep -v Warning
