#!/bin/bash
# Contact preset A/B at MetalSim HEAD, installed forks: task default contact_cfg="recommended" (tau10_impact_hardlimits,
# e3ba79f) vs contact_cfg=null (the model's own solref/solimp: the 2026-09-25 headline setting). Interleaved R D R D,
# g1_tp_variants.py 4096, then the step profile once each.
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
G='^  \(physics\|whole\|outside\|solver\|mjw.step\|fwd_\)\|^task\|^variant\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
python3 scripts/gpu_lock.py status | head -3
for r in 1 2; do
  echo "--- R$r $(date +%H:%M:%S)"; MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"R${r}_recommended\"}" python scripts/diagnostics/g1_tp_variants.py 4096 > runs/mjw_tp/ab/tp_R$r.log 2>&1; grep "$F" runs/mjw_tp/ab/tp_R$r.log
  echo "--- D$r $(date +%H:%M:%S)"; MJW_TP_VARIANT="{\"contact_cfg\":null,\"label\":\"D${r}_no_preset\"}" python scripts/diagnostics/g1_tp_variants.py 4096 > runs/mjw_tp/ab/tp_D$r.log 2>&1; grep "$F" runs/mjw_tp/ab/tp_D$r.log
done
echo "--- profile R $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 > runs/mjw_tp/ab/prof_R.log 2>&1; grep "$G" runs/mjw_tp/ab/prof_R.log
echo "--- profile D $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":null}' python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 > runs/mjw_tp/ab/prof_D.log 2>&1; grep "$G" runs/mjw_tp/ab/prof_D.log
echo "=== end $(date) power: $(pmset -g batt | head -1)"
