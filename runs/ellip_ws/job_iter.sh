#!/bin/bash
# ellip_ws: marginal graph-mode cost per Newton iteration (g1_solve_iteration_cost.py 4096 20), elliptic ellip10 (the task default),
# frozen worktree: U unfused, F fused, H fused + htot fusion, E fused + extrapolated warm start.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test
R='^  *[0-9]* |\|^  k\|^state\|^variant\|imports\|Error\|Traceback\|total'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
run() { PYTHONPATH=$WT env $2 MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['x','4096','20']; runpy.run_path('scripts/diagnostics/g1_solve_iteration_cost.py', run_name='__main__')" > runs/ellip_ws/iter_$1.log 2>&1; grep "$R" runs/ellip_ws/iter_$1.log; }
echo "--- U unfused $(date +%H:%M:%S)"; run U_unfused "MJW_METAL_FUSE_UPDATE=0"
echo "--- F fused $(date +%H:%M:%S)"; run F_fused "MJW_METAL_FUSE_UPDATE=1"
echo "--- H fused+htot $(date +%H:%M:%S)"; run H_fused_htot "MJW_METAL_FUSE_HTOT=1"
echo "--- E fused+extrap $(date +%H:%M:%S)"; run E_fused_extrap "MJW_WARMSTART_EXTRAP=1"
echo "=== end $(date) power: $(pmset -g batt | head -1)"
