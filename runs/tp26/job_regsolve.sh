#!/bin/bash
# tp26 job 2: fork worktrees metalsim-tp (Warp: register triangular solve in tile_cholesky_solve; MuJoCo Warp: lane-parallel
# sparse L'DL) vs the installed forks on the G1 step (recommended preset), g1_tp_variants --quick, interleaved:
# I = installed, S = worktree with the register solve only (MJW_METAL_LDL_LANES=0), L = worktree with the L'DL lanes only
# (WP_METAL_REGISTER_SOLVE=0), B = worktree with both. The worktree Warp also carries the compact register Cholesky layout
# (bitwise the generic one): its cost split is measured separately (compact off / on). Plus the isolated kernel costs (chol parts, L'DL bench).
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) / wt $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) / wt $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- chol parts, register solve off (worktree, knob 0) $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_REGISTER_SOLVE=0 python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|register solve"
echo "--- chol parts, register solve on (worktree) $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_REGISTER_SOLVE=1 python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|register solve"
echo "--- chol parts, register solve on, compact register layout off (worktree) $(date +%H:%M:%S)"; PYTHONPATH=$WT WP_METAL_COMPACT_REGISTER_CHOLESKY=0 python scripts/diagnostics/metal_cholesky_parts.py 32 43 48 2>&1 | grep "^n=\|register solve"
echo "--- L'DL bench (worktree) $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/ldl_lanes_bench.py 4096 2>&1 | grep "per launch\|Error\|Traceback"
run_wt() { PYTHONPATH=$WT MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"$1\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/tp26/regsolve_$1.log 2>&1; grep "$F" runs/tp26/regsolve_$1.log; }
for r in 1 2; do
  echo "--- I$r installed $(date +%H:%M:%S)"; MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"I${r}_installed\"}" python scripts/diagnostics/g1_tp_variants.py 4096 --quick > runs/tp26/regsolve_I$r.log 2>&1; grep "$F" runs/tp26/regsolve_I$r.log
  echo "--- S$r solve only $(date +%H:%M:%S)"; MJW_METAL_LDL_LANES=0 run_wt S${r}_regsolve
  echo "--- L$r ldl lanes only $(date +%H:%M:%S)"; WP_METAL_REGISTER_SOLVE=0 run_wt L${r}_ldl
  echo "--- B$r both $(date +%H:%M:%S)"; run_wt B${r}_both
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
