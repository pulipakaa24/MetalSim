#!/bin/bash
# tp26 round 2c: split-loop register Cholesky step on the tile path (Warp worktree, WP_METAL_CHOL_SPLIT 1 vs 0): bitwise
# A/B of factor and solve, cost split at n = 32, 43, 48, then the G1 step (worktrees, interleaved P G P G).
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
S=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- bitwise split vs generic $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_CHOL_SPLIT=0 python scripts/diagnostics/metal_register_cholesky_ab.py --out $S/chol_generic.npz 2>&1 | grep "saved\|Error\|Traceback"; PYTHONPATH=$WT WP_METAL_CHOL_SPLIT=1 python scripts/diagnostics/metal_register_cholesky_ab.py --ref $S/chol_generic.npz 2>&1 | grep "bitwise\|DIFF\|BITWISE\|DIFFERENCES\|Error\|Traceback" | cut -c1-100
echo "--- cost split, generic step $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_CHOL_SPLIT=0 python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|chol split"
echo "--- cost split, split step $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_CHOL_SPLIT=1 python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|chol split"
run_wt() { PYTHONPATH=$WT WP_METAL_CHOL_SPLIT=$2 MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"$1\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/tp26/split_$1.log 2>&1; grep "$F" runs/tp26/split_$1.log; }
for r in 1 2; do
  echo "--- P$r split $(date +%H:%M:%S)"; run_wt P${r}_split 1
  echo "--- G$r generic $(date +%H:%M:%S)"; run_wt G${r}_generic 0
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
