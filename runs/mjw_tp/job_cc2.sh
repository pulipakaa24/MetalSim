#!/bin/bash
# Decompose the contact preset's cost: hard limits alone ("hardlimits") vs the full preset vs tau10_imp99_hardlimits vs
# default; g1_tp_variants.py --quick (physics only + full env step) interleaved, then solver.solve for hardlimits.
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
G='^  \(physics\|whole\|solver\|mjw.step\)\|^variant\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for lab in hardlimits recommended tau10_imp99_hardlimits hardlimits default; do
  echo "--- $lab $(date +%H:%M:%S)"; MJW_TP_VARIANT="{\"contact_cfg\":\"$lab\",\"label\":\"$lab\"}" python scripts/diagnostics/g1_tp_variants.py 4096 --quick 2>&1 | grep "$F"
done
echo "--- profile hardlimits $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":"hardlimits"}' python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 2>&1 | grep "$G"
echo "=== end $(date)"
