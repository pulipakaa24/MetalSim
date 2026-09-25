#!/bin/bash
# Fall rates with 8 starts per cell type (rep 0 = protocol, 7 jittered by up to +-5 cm), Isaac's and our checkpoints,
# levels 3 and 6, per terrain collision surface; MuJoCo Warp fork a8e6485 (clean export).
cd "$(dirname "$0")/../.."
export PYTHONPATH=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/mjw_a8e6485
.venv/bin/python scripts/gpu_lock.py status
I=runs/parity/isaac/rough/rsl_rl_g1_rough/2026-09-25_09-54-11
O=runs/policies/g1_rough_ppowarp_fixed
for TOK in "$@"; do
  MODE=${TOK%%/*}; SCAN=grid; [ "$TOK" != "$MODE" ] && SCAN=${TOK#*/}
  echo "=== terrain_collision $MODE scan_surface $SCAN reps 8"
  .venv/bin/python scripts/diagnostics/g1_rough_transfer.py --isaac_play runs/parity/isaac/rough/play_rough --level 3 6 --reps 8 \
    --ckpt isaac_it500=$I/model_500.pt --ckpt isaac_it1000=$I/model_1000.pt --ckpt isaac_it1499=$I/model_1499.pt \
    --ckpt metalsim_ours_it500=${O}_it500.pt --ckpt metalsim_ours_it1000=${O}_it1000.pt --ckpt metalsim_ours_it1500=${O}_it1500.pt \
    --terrain_collision "$MODE" --scan_surface "$SCAN" --out runs/terrain_walls/transfer_reps.jsonl 2>&1 | grep -E "falls|stairs:|boxes:|rough:|total|Error|Trace"
done
