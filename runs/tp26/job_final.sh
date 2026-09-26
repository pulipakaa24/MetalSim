#!/bin/bash
# tp26 final table: everything landed today vs the morning's starting point, full g1_tp_variants (physics, step, rollout +
# inference, PPO update, full loop), 4096 envs, recommended preset, interleaved B F B F.
#   B = installed forks (fe6fe71 / 4127c48), task capacities njmax 512 / nconmax 128, policy mapping A
#   F = worktrees metalsim-tp (register solve, fused launches; L'DL chains off pending their A/B), capacity bound, mapping D
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) installed warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD); wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for r in 1 2; do
  echo "--- B$r baseline $(date +%H:%M:%S)"; METALSIM_MLP_MAPPING=A MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"njmax\":512,\"nconmax\":128,\"label\":\"B${r}_baseline\"}" python scripts/diagnostics/g1_tp_variants.py 4096 > runs/tp26/final_B$r.log 2>&1; grep "$F" runs/tp26/final_B$r.log
  echo "--- F$r final $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_METAL_LDL_CHAINS=0 METALSIM_MLP_MAPPING=D MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"F${r}_final\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/tp26/final_F$r.log 2>&1; grep "$F" runs/tp26/final_F$r.log
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
