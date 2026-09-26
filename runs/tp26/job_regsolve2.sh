#!/bin/bash
# tp26 job 2b: worktrees after the A/B of job 2 (compact register layout off by default; L'DL lanes rewritten on device
# memory). Bitwise check of the L'DL kernels first (must print ALL OK), then the L'DL bench and the interleaved step A/B:
# I = installed, S = worktree register solve only (MJW_METAL_LDL_LANES=0), B = worktree both.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) / wt $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) / wt $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- L'DL bitwise check (worktree) $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/ldl_lanes_check.py 64 2>&1 | grep -v "^Module\|^Warp\|^   \|^$"
echo "--- L'DL bench (worktree) $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/ldl_lanes_bench.py 4096 2>&1 | grep "per launch\|Error\|Traceback"
echo "--- chol parts (worktree defaults: register solve on, compact off) $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|register solve"
run_wt() { PYTHONPATH=$WT MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"$1\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/tp26/regsolve2_$1.log 2>&1; grep "$F" runs/tp26/regsolve2_$1.log; }
for r in 1 2; do
  echo "--- I$r installed $(date +%H:%M:%S)"; MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"I${r}_installed\"}" python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/regsolve2_I$r.log 2>&1; grep "$F" runs/tp26/regsolve2_I$r.log
  echo "--- S$r solve only $(date +%H:%M:%S)"; MJW_METAL_LDL_LANES=0 run_wt S${r}_regsolve
  echo "--- B$r both $(date +%H:%M:%S)"; run_wt B${r}_both
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
