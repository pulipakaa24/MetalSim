#!/bin/bash
# tp26 checks: g1_tp_check.py state-difference protocol (512 envs, 50 steps; recommended preset = the task default):
# installed forks (reference file) vs the worktrees (register solve + L'DL lanes), each with its own run-to-run floor,
# replay-vs-eager, capacity/overflow and the MuJoCo C protocol.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD)"
echo "--- baseline (installed) $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"label":"installed"}' python scripts/diagnostics/g1_tp_check.py 512 50 --out runs/tp26/check_base.npz 2>&1 | grep -v "^Module\|^Warp\|^   [^st]\|^$"
echo "--- worktree (both) $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_TP_VARIANT='{"label":"worktree regsolve+ldl"}' python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_check.py','512','50','--ref','runs/tp26/check_base.npz','--out','runs/tp26/check_wt.npz']; runpy.run_path('scripts/diagnostics/g1_tp_check.py', run_name='__main__')" 2>&1 | grep -v "^Module\|^Warp\|^   [^st]\|^$"
echo "=== end $(date)"
