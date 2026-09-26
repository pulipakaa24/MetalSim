#!/bin/bash
# g1_falls_batch.sh N: falls of Isaac's PhysX-trained checkpoints per contact preset and seed (g1_falls.py).
#   scripts/gpu_run.sh g1_falls1 render 25 -- scripts/diagnostics/competitors/g1_falls_batch.sh 1
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python; S=scripts/diagnostics/competitors/g1_falls.py; O=runs/competitors/g1_falls; mkdir -p $O
C=runs/contact_research
run() { name=$1; shift; t0=$(date +%s); $PY $S "$@" --out $O/$name.json 2>&1 | grep -v "^Module\|^Warp\|^   " | tail -1; echo "$name rc=$? $(( $(date +%s)-t0 )) s"; }
if [ "$1" = 1 ]; then
  for seed in 1 2 3; do for p in tau10_impact_hardlimits tau10_impact_hardlimits_ellip1 tau10_impact_hardlimits_ellip10 tau10_impact_hardlimits_ellip100; do
    run isaac1000_${p}_s$seed $C/isaac_model_1000_metalsim.pt --contact_cfg $p --seed $seed
  done; done
else
  for ck in 500 1499; do for p in tau10_impact_hardlimits tau10_impact_hardlimits_ellip10; do
    run isaac${ck}_${p}_s1 $C/isaac_model_${ck}_metalsim.pt --contact_cfg $p --seed 1
  done; done
  for p in default tau5_imp99_hardlimits; do run isaac1000_${p}_s1 $C/isaac_model_1000_metalsim.pt --contact_cfg $p --seed 1; done
fi
