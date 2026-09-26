#!/bin/bash
# tp26 job 9: chain-parallel sparse L'DL (worktree, MJW_METAL_LDL_CHAINS 1 vs 0): correctness check (solve bitwise, factor
# float noise), isolated bench, then the interleaved step A/B (register solve + fused launches on in both).
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- check $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/ldl_lanes_check.py 64 2>&1 | grep -v "^Module\|^Warp\|^   [^0-9]\|^$" | tail -12
echo "--- bench $(date +%H:%M:%S)"; PYTHONPATH=$WT python scripts/diagnostics/ldl_lanes_bench.py 4096 2>&1 | grep "per launch\|Error\|Traceback"
run_wt() { PYTHONPATH=$WT MJW_METAL_LDL_CHAINS=$2 MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"$1\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096','--quick']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/tp26/chains_$1.log 2>&1; grep "$F" runs/tp26/chains_$1.log; }
for r in 1 2; do
  echo "--- C$r chains $(date +%H:%M:%S)"; run_wt C${r}_chains 1
  echo "--- S$r serial $(date +%H:%M:%S)"; run_wt S${r}_serial 0
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
