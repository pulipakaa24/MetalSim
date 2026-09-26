#!/bin/bash
# tp26 job 5: marginal graph-mode cost per Newton iteration (g1_solve_iteration_cost.py), recommended vs null, installed
# forks vs the worktrees (register solve + L'DL lanes). Raw output in runs/tp26/itercost_<cfg>_<inst|wt>.log.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
R='^  *[0-9]* |\|^  k\|^state\|^variant\|imports\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for cfg in recommended null; do
  V="{\"contact_cfg\":$([ $cfg = null ] && echo null || echo \"$cfg\")}"
  echo "--- installed, contact_cfg $cfg $(date +%H:%M:%S)"; MJW_TP_VARIANT="$V" python scripts/diagnostics/g1_solve_iteration_cost.py 4096 10 > runs/tp26/itercost_${cfg}_inst.log 2>&1; grep "$R" runs/tp26/itercost_${cfg}_inst.log
  echo "--- worktree, contact_cfg $cfg $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_TP_VARIANT="$V" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['x','4096','10']; runpy.run_path('scripts/diagnostics/g1_solve_iteration_cost.py', run_name='__main__')" > runs/tp26/itercost_${cfg}_wt.log 2>&1; grep "$R" runs/tp26/itercost_${cfg}_wt.log
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
