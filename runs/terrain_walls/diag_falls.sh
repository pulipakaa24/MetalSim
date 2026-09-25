#!/bin/bash
# per-step trajectories of the Isaac checkpoints that still fall on the box surface
cd "$(dirname "$0")/../.."; source .venv/bin/activate
python3 scripts/gpu_lock.py status
I=runs/parity/isaac/rough/rsl_rl_g1_rough/2026-09-25_09-54-11
for MODE in boxes_local boxes hfield; do
python scripts/diagnostics/g1_rough_transfer.py --isaac_play runs/parity/isaac/rough/play_rough --level 3 6 \
  --ckpt isaac_it500=$I/model_500.pt --ckpt isaac_it1000=$I/model_1000.pt --terrain_collision $MODE \
  --traj_dir runs/terrain_walls/traj 2>&1 | grep -E "^\||falls|total|Error|Trace"
done
