#!/bin/bash
# ellip_ws: ICB early exit of the Newton loop (Warp fork 6bceb39d indirect execution ranges, mujoco_warp 12cc1e2 (the warp.context fix)), frozen worktrees
# warp-innate-ellip-test / mujoco_warp-ellip-test5, elliptic ellip10 cap 20 (task default, fused + htot): (a) fork graph-capture tests
# on Metal, (b) g1_tp_check 512/50 knob off (reference) vs on, (c) itercost 4096/20 off and on, (d) g1_tp_variants --quick off/on x2.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-ellip-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test5
R='^  *[0-9]* |\|^  k\|^state\|Error\|Traceback'; F='^\[\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-ellip-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test5 rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
chk() { PYTHONPATH=$WT env $1 MJW_TP_VARIANT="{\"label\":\"$2\"}" python -c "import runpy, sys; sys.argv=['g1_tp_check.py','512','50']+sys.argv[1:]; runpy.run_path('scripts/diagnostics/g1_tp_check.py', run_name='__main__')" $3 2>&1 | grep -v "^Module\|^Warp\|^   [^s]\|^$\|^linesearch\|^To disable\|^solver iterations"; }
echo "--- (b) check: early exit off (reference) $(date +%H:%M:%S)"; chk "MJW_METAL_ICB_EARLY_EXIT=0" "icb off" "--out runs/ellip_ws/check_icb2_off.npz"
echo "--- (b) check: early exit on vs off $(date +%H:%M:%S)"; chk "MJW_METAL_ICB_EARLY_EXIT=1" "icb on" "--ref runs/ellip_ws/check_icb2_off.npz --out runs/ellip_ws/check_icb2_on.npz"
for k in 0 1; do
  echo "--- (c) itercost early exit $k $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_METAL_ICB_EARLY_EXIT=$k MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python -c "import runpy, sys; sys.argv=['x','4096','20']; runpy.run_path('scripts/diagnostics/g1_solve_iteration_cost.py', run_name='__main__')" > runs/ellip_ws/iter_icb2_$k.log 2>&1; grep "$R" runs/ellip_ws/iter_icb2_$k.log
done
for r in 1 2; do for k in 0 1; do
  echo "--- (d) quick early exit $k r$r $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_METAL_ICB_EARLY_EXIT=$k MJW_TP_VARIANT="{\"label\":\"icb${k}_r$r\"}" python -c "import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/ellip_ws/time_icb2_${k}_r$r.log 2>&1; grep "$F" runs/ellip_ws/time_icb2_${k}_r$r.log
done; done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
