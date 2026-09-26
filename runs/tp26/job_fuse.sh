#!/bin/bash
# tp26 job 7: fused per-iteration update (worktree, MJW_METAL_FUSE_UPDATE 1 vs 0; the register solve on in both) and
# the policy layer mapping (installed forks, METALSIM_MLP_MAPPING D vs A) on the full g1_tp_variants (physics, step,
# rollout + inference, PPO loop), interleaved F U F U then D A D A. Raw logs runs/tp26/fuse_*.log.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
run_wt() { PYTHONPATH=$WT MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"$1\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096']+sys.argv[1:]; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" $2 > runs/tp26/fuse_$1.log 2>&1; grep "$F" runs/tp26/fuse_$1.log; }
for r in 1 2; do
  echo "--- F$r fused $(date +%H:%M:%S)"; MJW_METAL_FUSE_UPDATE=1 run_wt F${r}_fused --quick
  echo "--- U$r unfused $(date +%H:%M:%S)"; MJW_METAL_FUSE_UPDATE=0 run_wt U${r}_unfused --quick
done
echo "--- itercost fused $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['x','4096','10']; runpy.run_path('scripts/diagnostics/g1_solve_iteration_cost.py', run_name='__main__')" > runs/tp26/itercost_recommended_fused.log 2>&1; grep '^  *[0-9]* |\|^  k\|^state\|Error\|Traceback' runs/tp26/itercost_recommended_fused.log
for r in 1 2; do
  echo "--- D$r mapping D (installed forks, full variants) $(date +%H:%M:%S)"; METALSIM_MLP_MAPPING=D MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"D${r}_map4\"}" python scripts/diagnostics/g1_tp_variants.py 4096 > runs/tp26/fuse_D$r.log 2>&1; grep "$F" runs/tp26/fuse_D$r.log
  echo "--- A$r mapping A $(date +%H:%M:%S)"; METALSIM_MLP_MAPPING=A MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"A${r}_map1\"}" python scripts/diagnostics/g1_tp_variants.py 4096 > runs/tp26/fuse_A$r.log 2>&1; grep "$F" runs/tp26/fuse_A$r.log
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
