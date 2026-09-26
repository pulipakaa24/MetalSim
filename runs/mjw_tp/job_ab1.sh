#!/bin/bash
# A/B: MuJoCo Warp fork 1791414 (shared checkout, editable install) vs 07a51a6 (worktree via PYTHONPATH), same Warp
# (upstream/warp-innate 9050cb54), same MetalSim commit; interleaved A B A B; g1_tp_variants.py 4096.
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback\|mjw:'
B=/Users/aditya/robosim/upstream/mujoco_warp-07a51a6
run() {  # run LABEL PYTHONPATH_OR_EMPTY
  local lab=$1 pp=$2
  echo "--- $lab $(date +%H:%M:%S) power: $(pmset -g batt | head -1 | tr -d '\n') therm: $(pmset -g therm | tr '\n' ' ')"
  echo "mjw: $(PYTHONPATH=$pp python -c 'import mujoco_warp;print(mujoco_warp.__file__)' 2>/dev/null | tail -1)"
  PYTHONPATH=$pp MJW_TP_VARIANT="{\"label\":\"$lab\"}" python scripts/diagnostics/g1_tp_variants.py 4096 > runs/mjw_tp/ab/tp_$lab.log 2>&1
  grep "$F" runs/mjw_tp/ab/tp_$lab.log
}
echo "=== start $(date) metalsim $(git rev-parse --short HEAD 2>/dev/null) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw-A $(git -C upstream/mujoco_warp rev-parse --short HEAD) mjw-B $(git -C $B rev-parse --short HEAD)"
python3 scripts/gpu_lock.py status
run A1_1791414 ""
run B1_07a51a6 $B
run A2_1791414 ""
run B2_07a51a6 $B
echo "=== end $(date) power: $(pmset -g batt | head -1)"
