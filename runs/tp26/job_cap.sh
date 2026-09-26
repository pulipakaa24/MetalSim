#!/bin/bash
# tp26 job 3: provable capacity bounds on the G1 flat task (installed forks, recommended preset): task defaults
# njmax 512 / nconmax 128 vs the bound njmax 144 / nconmax 24 (metalsim.physics.capacity), g1_tp_variants --quick,
# interleaved D A D A.
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for r in 1 2; do
  echo "--- D$r defaults $(date +%H:%M:%S)"; MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"D${r}_njmax512_ncon128\"}" python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/cap_D$r.log 2>&1; grep "$F" runs/tp26/cap_D$r.log
  echo "--- A$r bound $(date +%H:%M:%S)"; MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"njmax\":144,\"nconmax\":24,\"label\":\"A${r}_njmax144_ncon24\"}" python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/cap_A$r.log 2>&1; grep "$F" runs/tp26/cap_A$r.log
done
# the Isaac Lab 3.0 like-for-like preset (cap 20: 20 Newton iterations launched every substep, most of them idle) with
# the task's capacities vs the bound: the idle iterations' capacity-sized launches shrink with njmax
echo "--- I1 il3 defaults $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":"recommended","solver_cfg":"isaaclab3_every_substep_cap20","label":"I1_il3_njmax512"}' python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/cap_I1.log 2>&1; grep "$F" runs/tp26/cap_I1.log
echo "--- J1 il3 bound $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":"recommended","solver_cfg":"isaaclab3_every_substep_cap20","njmax":144,"nconmax":24,"label":"J1_il3_njmax144"}' python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/cap_J1.log 2>&1; grep "$F" runs/tp26/cap_J1.log
echo "=== end $(date) power: $(pmset -g batt | head -1)"
