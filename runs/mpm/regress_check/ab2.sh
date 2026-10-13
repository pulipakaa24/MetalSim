#!/bin/bash
# Item-4 machine-state check on the pinned trees (B): shared .venv-newton152 (S), the reproducibility agent's fresh
# .venv-newton152 (F), and S with 12 busy CPU processes running alongside (L, `yes > /dev/null`). Interleaved.
#   ab2.sh REPS
H=/Users/aditya/robosim/.claude/worktrees/agent-a7a2cb997dc44ebcc
D=$H/runs/mpm/regress_check; T=$H/upstream/t4; O=$D/out2; mkdir -p $O
PYS=/Users/aditya/robosim/.venv-newton152/bin/python
PYF=/private/tmp/claude-501/-Users-aditya-robosim/3a553738-d88f-4d8d-ac36-4ecff3ac4bc1/scratchpad/dry/MetalSim/.venv-newton152/bin/python
export PYTHONDONTWRITEBYTECODE=1
args() { case $1 in granular) echo "example:mpm_granular" ;; column_a2) echo "column --set aspect=2" ;; esac; }
one() { # arm scene rep
  local arm=$1 sc=$2 rep=$3 py=$PYS pids=""
  [ $arm = F ] && py=$PYF
  if [ $arm = L ]; then for i in $(seq 12); do yes > /dev/null & pids="$pids $!"; done; fi
  local j=$O/${sc}_${arm}_r$rep.json
  (cd $H && MPM_DA=keep WARP_CACHE_PATH=$D/kc_B PYTHONPATH=$T/warp-B:$T/newton-B:$H $py $D/speed_da.py $(args $sc) \
     --frames 60 --warmup 5 --metal-fast-math relaxed --json $j > ${j%.json}.log 2>&1)
  [ -n "$pids" ] && { kill $pids; wait $pids 2>/dev/null; }
  echo "$sc $arm r$rep: $($PYS -c "import json;print(round(json.load(open('$j'))['ms_median'],2))" 2>&1 | tail -1) util=$(/Users/aditya/robosim/scripts/gpu_util.sh) load=$(sysctl -n vm.loadavg) $(date +%H:%M:%S)"; }
for rep in $(seq 1 $1); do for sc in granular column_a2; do for arm in S F L; do one $arm $sc $rep; done; done; done
