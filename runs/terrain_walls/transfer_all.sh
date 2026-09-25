#!/bin/bash
# Rough transfer (scripts/diagnostics/g1_rough_transfer.py) on each terrain collision surface; Isaac's checkpoints
# 500/1000/1499 and ours 500/1000/1500, levels 3 and 6. Run through scripts/gpu_run.sh.
cd "$(dirname "$0")/../.."
source .venv/bin/activate  # (keeps an exported PYTHONPATH)
python3 scripts/gpu_lock.py status
I=runs/parity/isaac/rough/rsl_rl_g1_rough/2026-09-25_09-54-11
O=runs/policies/g1_rough_ppowarp_fixed
for TOK in "$@"; do             # MODE or MODE/SCAN (SCAN: grid | exact)
  MODE=${TOK%%/*}; SCAN=grid; [ "$TOK" != "$MODE" ] && SCAN=${TOK#*/}
  echo "=== terrain_collision $MODE scan_surface $SCAN"
  python scripts/diagnostics/g1_rough_transfer.py --isaac_play runs/parity/isaac/rough/play_rough --level 3 6 \
    --ckpt isaac_it500=$I/model_500.pt --ckpt isaac_it1000=$I/model_1000.pt --ckpt isaac_it1499=$I/model_1499.pt \
    --ckpt metalsim_ours_it500=${O}_it500.pt --ckpt metalsim_ours_it1000=${O}_it1000.pt --ckpt metalsim_ours_it1500=${O}_it1500.pt \
    --terrain_collision "$MODE" --scan_surface "$SCAN" --out runs/terrain_walls/transfer.jsonl 2>&1 | grep -v "^Module\|^   \|^Warp\|warnings.warn\|UserWarning"
done
