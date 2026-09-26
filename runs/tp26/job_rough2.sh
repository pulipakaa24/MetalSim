#!/bin/bash
# tp26 round 2: rough loop with everything landed (installed forks at the merged heads, task defaults), full g1_tp_variants,
# R1 R2; and the flat loop once more for the record.
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__)"
for r in 1 2; do
  echo "--- rough R$r $(date +%H:%M:%S)"; MJW_TP_TERRAIN=rough MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"rough${r}_landed\"}" python scripts/diagnostics/g1_tp_variants.py 4096 > runs/tp26/rough2_R$r.log 2>&1; grep "$F" runs/tp26/rough2_R$r.log
done
echo "--- flat F1 $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":"recommended","label":"flat_landed"}' python scripts/diagnostics/g1_tp_variants.py 4096 > runs/tp26/rough2_flat.log 2>&1; grep "$F" runs/tp26/rough2_flat.log
echo "=== end $(date) power: $(pmset -g batt | head -1)"
