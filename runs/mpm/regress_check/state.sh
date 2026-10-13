#!/bin/bash
# Item-4 machine-state probe: cycles of (whole-GPU FMA throughput, granular B, column a=2 B), to correlate the MPM
# ms/frame with the GPU's available throughput over time.   state.sh CYCLES TAG
H=/Users/aditya/robosim/.claude/worktrees/agent-a7a2cb997dc44ebcc
D=$H/runs/mpm/regress_check; T=$H/upstream/t4
PY=/Users/aditya/robosim/.venv-newton152/bin/python
for c in $(seq 1 $1); do
  f=$(WARP_CACHE_PATH=$D/kc_B PYTHONPATH=$T/warp-B $PY $D/fma_peak.py 2>&1 | /usr/bin/grep FMA_TFLOPS)
  echo "c$c $f $(date +%H:%M:%S)"
  OUT=$D/out_state_$2/c$c $D/ab.sh run 1 2>/dev/null | sed "s/^/c$c /" &
  wait
done
