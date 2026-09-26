#!/bin/bash
# tp26 job 8: rolled register Cholesky (thread-memory columns, runtime loops) vs the unrolled register form, cost split at
# n = 32, 40, 43, 48 (worktree Warp), plus bitwise check of the factor against the unrolled form.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
S=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad
echo "=== start $(date) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- unrolled (worktree default) $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/metal_cholesky_parts.py 32 40 43 48 2>&1 | grep "^n=\|rolled"
echo "--- rolled above 32 $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_ROLLED_CHOLESKY=32 python scripts/diagnostics/metal_cholesky_parts.py 32 40 43 48 2>&1 | grep "^n=\|rolled"
echo "--- rolled above 0 (also n = 32) $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_ROLLED_CHOLESKY=1 python scripts/diagnostics/metal_cholesky_parts.py 32 43 2>&1 | grep "^n=\|rolled"
echo "--- bitwise: rolled vs unrolled $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/metal_register_cholesky_ab.py --out $S/chol_unrolled.npz 2>&1 | grep saved; PYTHONPATH=$WT WP_METAL_ROLLED_CHOLESKY=1 python -c "
import os, warp as wp; wp.config.metal_rolled_cholesky = 1
import runpy, sys; sys.argv=['x','--ref','$S/chol_unrolled.npz']; runpy.run_path('scripts/diagnostics/metal_register_cholesky_ab.py', run_name='__main__')" 2>&1 | grep "bitwise\|DIFF\|BITWISE\|DIFFERENCES"
echo "=== end $(date) power: $(pmset -g batt | head -1)"
