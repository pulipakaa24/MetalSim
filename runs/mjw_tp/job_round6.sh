#!/bin/bash
# round 6: profile of the current default (steps 1-3) and the next candidates on top of it
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
WP_METAL_PROFILE=1 python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --kernels 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
echo "=== end $(date)"
} > runs/mjw_tp/profile6.log 2>&1
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
for V in '{"label":"default (steps 1-3)"}' '{"jacobian":"sparse","label":"+ sparse Jacobian"}' '{"njmax":128,"nconmax":16,"label":"+ capacity 128/16"}' \
         '{"block_dim":{"sparse_ldl_serial":16},"label":"+ ldl block 16"}' '{"label":"default repeat"}'; do
  MJW_TP_VARIANT="$V" python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
done
echo "=== end $(date)"
} > runs/mjw_tp/costsplit6.log 2>&1
