#!/bin/bash
# ellip_ws: g1_tp_variants.py 4096 --quick (physics only, full env step) on the frozen worktree, elliptic ellip10 cap 20 (the task
# default), interleaved U F H E U F H E: U = unfused per-iteration launches (MJW_METAL_FUSE_UPDATE=0), F = fused (default),
# H = fused + htot fusion (MJW_METAL_FUSE_HTOT=1), E = fused + extrapolated warm start (MJW_WARMSTART_EXTRAP=1).
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
run() { PYTHONPATH=$WT env $2 MJW_TP_VARIANT="{\"label\":\"$1\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/ellip_ws/time_$1.log 2>&1; grep "$F" runs/ellip_ws/time_$1.log; }
for r in 1 2; do
  echo "--- U$r unfused $(date +%H:%M:%S)"; run U${r}_unfused "MJW_METAL_FUSE_UPDATE=0"
  echo "--- F$r fused $(date +%H:%M:%S)"; run F${r}_fused "MJW_METAL_FUSE_UPDATE=1"
  echo "--- H$r fused+htot $(date +%H:%M:%S)"; run H${r}_fused_htot "MJW_METAL_FUSE_HTOT=1"
  echo "--- E$r fused+extrap $(date +%H:%M:%S)"; run E${r}_fused_extrap "MJW_WARMSTART_EXTRAP=1"
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
