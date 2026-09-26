#!/bin/bash
# ellip_ws: where the rollout's elliptic cost goes. Frozen worktree mujoco_warp-ellip-test3 (ca431fc: fused + htot, lanes knob).
# (a) itercost 4096/20 on the state after one PPO rollout (untrained policy) for U (unfused), F (fused, per-entry htot),
# H32, H128 (fused + htot at 32 / 128 lanes); (b) g1_tp_variants full (rollout + inference, loop) for U, F, H128.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test3
R='^  *[0-9]* |\|^  k\|^state\|Error\|Traceback'; F='^\[\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test3 rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
it() { echo "--- (a) itercost rollout state $1 $(date +%H:%M:%S)"; PYTHONPATH=$WT env $2 MJW_TP_VARIANT='{"contact_cfg":"recommended","state":"rollout"}' python -c "import runpy, sys; sys.argv=['x','4096','20']; runpy.run_path('scripts/diagnostics/g1_solve_iteration_cost.py', run_name='__main__')" > runs/ellip_ws/iterroll_$1.log 2>&1; grep "$R" runs/ellip_ws/iterroll_$1.log; }
it U "MJW_METAL_FUSE_UPDATE=0"; it F "MJW_METAL_FUSE_HTOT=0"; it H32 "MJW_METAL_FUSE_LANES=32"; it H128 "MJW_METAL_FUSE_LANES=128"
full() { echo "--- (b) full variants $1 $(date +%H:%M:%S)"; PYTHONPATH=$WT env $2 MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"$1\"}" python -c "import runpy, sys; sys.argv=['g1_tp_variants.py','4096']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/ellip_ws/full_$1.log 2>&1; grep "$F" runs/ellip_ws/full_$1.log; }
full U "MJW_METAL_FUSE_UPDATE=0"; full F "MJW_METAL_FUSE_HTOT=0"; full H128 "MJW_METAL_FUSE_LANES=128"
echo "=== end $(date) power: $(pmset -g batt | head -1)"
