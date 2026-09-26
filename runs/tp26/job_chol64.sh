#!/bin/bash
# tp26 job 12: 64-lane register Cholesky (Warp worktree, block_dim 64) vs the 32-lane form: bitwise A/B of factor and solve,
# cost split at n = 32, 43, 48, then the G1 step with MJW_METAL_CHOL_LANES 64 vs 32 (worktrees, interleaved).
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
S=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- bitwise 64 vs 32 lanes $(date +%H:%M:%S)"; PYTHONPATH=$WT CHOL_BLOCK_DIM=32 python scripts/diagnostics/metal_register_cholesky_ab.py --out $S/chol_bd32.npz 2>&1 | grep "saved\|Error\|Traceback"; PYTHONPATH=$WT CHOL_BLOCK_DIM=64 python scripts/diagnostics/metal_register_cholesky_ab.py --ref $S/chol_bd32.npz 2>&1 | grep "n=\|bitwise\|DIFF\|BITWISE\|DIFFERENCES\|Error\|Traceback" | cut -c1-120
echo "--- cost split, 32 lanes $(date +%H:%M:%S)"; PYTHONPATH=$WT CHOL_BLOCK_DIM=32 python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|block_dim"
echo "--- cost split, 64 lanes $(date +%H:%M:%S)"; PYTHONPATH=$WT CHOL_BLOCK_DIM=64 python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|block_dim\|Error\|Traceback"
run_wt() { PYTHONPATH=$WT MJW_METAL_CHOL_LANES=$2 MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"$1\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/tp26/chol64_$1.log 2>&1; grep "$F" runs/tp26/chol64_$1.log; }
for r in 1 2; do
  echo "--- L64 $r $(date +%H:%M:%S)"; run_wt L64_$r 64
  echo "--- L32 $r $(date +%H:%M:%S)"; run_wt L32_$r 32
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
