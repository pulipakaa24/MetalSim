#!/bin/bash
# tp26 job 4: rough terrain (the task's default rough config) under the recommended preset vs contact_cfg null, never
# measured before (README rough 41.9 K is the default-contact number); installed forks; g1_tp_variants --quick, R N R N.
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback\|^task'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for r in 1 2; do
  echo "--- R$r rough recommended $(date +%H:%M:%S)"; MJW_TP_TERRAIN=rough MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"R${r}_rough_recommended\"}" python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/rough_R$r.log 2>&1; grep "$F" runs/tp26/rough_R$r.log
  echo "--- N$r rough null $(date +%H:%M:%S)"; MJW_TP_TERRAIN=rough MJW_TP_VARIANT="{\"contact_cfg\":null,\"label\":\"N${r}_rough_null\"}" python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/rough_N$r.log 2>&1; grep "$F" runs/tp26/rough_N$r.log
done
echo "--- rough step profile, recommended $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 10 --terrain rough > runs/tp26/rough_prof.log 2>&1; grep '^  \(physics\|whole\|outside\|solver\|mjw.step\|fwd_\|sensor\|sum of\|  of which\)\|^task\|Error\|Traceback' runs/tp26/rough_prof.log
echo "=== end $(date) power: $(pmset -g batt | head -1)"
