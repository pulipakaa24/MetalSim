#!/bin/bash
# A/B step profile (segment graphs, --reps 20): fork 1791414 vs 07a51a6, interleaved A B A B.
cd /Users/aditya/robosim && source .venv/bin/activate
B=/Users/aditya/robosim/upstream/mujoco_warp-07a51a6
run() {
  local lab=$1 pp=$2
  echo "--- $lab $(date +%H:%M:%S) power: $(pmset -g batt | head -1 | tr -d '\n')"
  echo "mjw: $(PYTHONPATH=$pp python -c 'import mujoco_warp;print(mujoco_warp.__file__)' 2>/dev/null | tail -1)"
  PYTHONPATH=$pp python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 > runs/mjw_tp/ab/prof_$lab.log 2>&1
  grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]" runs/mjw_tp/ab/prof_$lab.log
}
echo "=== start $(date)"
python3 scripts/gpu_lock.py status
run A1_1791414 ""
run B1_07a51a6 $B
run A2_1791414 ""
run B2_07a51a6 $B
echo "=== end $(date) power: $(pmset -g batt | head -1)"
