#!/bin/bash
# ellip_ws: lanes per world of the elliptic fused launch (htot fusion on), frozen worktree mujoco_warp-ellip-test3: itercost (4096, cap 20)
# at 64 and 128 lanes (32 is runs/ellip_ws/iter_H_fused_htot.log), then g1_tp_variants --quick interleaved 32 64 128 32 64 128.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test3
R='^  *[0-9]* |\|^  k\|^state\|Error\|Traceback'; F='^\[\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test3 rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for L in 64 128; do
  echo "--- itercost lanes $L $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_METAL_FUSE_LANES=$L MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python -c "import runpy, sys; sys.argv=['x','4096','20']; runpy.run_path('scripts/diagnostics/g1_solve_iteration_cost.py', run_name='__main__')" > runs/ellip_ws/iter_lanes$L.log 2>&1; grep "$R" runs/ellip_ws/iter_lanes$L.log
done
for r in 1 2; do for L in 32 64 128; do
  echo "--- L$L r$r $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_METAL_FUSE_LANES=$L MJW_TP_VARIANT="{\"label\":\"lanes${L}_r$r\"}" python -c "import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/ellip_ws/time_lanes${L}_r$r.log 2>&1; grep "$F" runs/ellip_ws/time_lanes${L}_r$r.log
done; done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
