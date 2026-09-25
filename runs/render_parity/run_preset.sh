#!/bin/bash
# usage: run_preset.sh PRESET [extra args]  -> renders rt and pt kinematic replays and metrics
set -u
P=$1; shift
cd /Users/aditya/robosim

for K in rt pt; do
  .venv/bin/python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/$K --out runs/render_parity/${K}_$P --preset $P "$@" 2>&1 | grep -v Warning
done
