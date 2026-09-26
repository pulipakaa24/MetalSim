#!/bin/bash
# ellip_ws: line-search derivative noise floor (MJW_LS_NOISE_FLOOR, eps multiples) on Metal, frozen worktrees warp-innate-ellip-test (6bceb39d)
# / mujoco_warp-ellip-test5 (12cc1e2: fused + htot lanes 32, ICB early exit on, floor off by default), elliptic ellip10 cap 20:
# (a) g1_tp_check 512/50 floor off (reference) vs 8 eps vs 32 eps, (b) itercost 4096/20 off / 8, (c) g1_tp_variants --quick off / 8 / 32 x2.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-ellip-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test5
R='^  *[0-9]* |\|^  k\|^state\|Error\|Traceback'; F='^\[\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-ellip-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test5 rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
chk() { PYTHONPATH=$WT MJW_LS_NOISE_FLOOR=$1 MJW_TP_VARIANT="{\"label\":\"ls floor $1\"}" python -c "import runpy, sys; sys.argv=['g1_tp_check.py','512','50']+sys.argv[1:]; runpy.run_path('scripts/diagnostics/g1_tp_check.py', run_name='__main__')" $2 2>&1 | grep -v "^Module\|^Warp\|^   [^s]\|^$\|^linesearch\|^To disable\|^solver iterations"; }
echo "--- (a) check floor off (reference) $(date +%H:%M:%S)"; chk 0 "--out runs/ellip_ws/check_ls0.npz"
echo "--- (a) check floor 8 eps vs off $(date +%H:%M:%S)"; chk 8 "--ref runs/ellip_ws/check_ls0.npz --out runs/ellip_ws/check_ls8.npz"
echo "--- (a) check floor 32 eps vs off $(date +%H:%M:%S)"; chk 32 "--ref runs/ellip_ws/check_ls0.npz --out runs/ellip_ws/check_ls32.npz"
for e in 0 8; do
  echo "--- (b) itercost floor $e $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_LS_NOISE_FLOOR=$e MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python -c "import runpy, sys; sys.argv=['x','4096','20']; runpy.run_path('scripts/diagnostics/g1_solve_iteration_cost.py', run_name='__main__')" > runs/ellip_ws/iter_ls$e.log 2>&1; grep "$R" runs/ellip_ws/iter_ls$e.log
done
for r in 1 2; do for e in 0 8 32; do
  echo "--- (c) quick floor $e r$r $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_LS_NOISE_FLOOR=$e MJW_TP_VARIANT="{\"label\":\"ls${e}_r$r\"}" python -c "import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/ellip_ws/time_ls${e}_r$r.log 2>&1; grep "$F" runs/ellip_ws/time_ls${e}_r$r.log
done; done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
