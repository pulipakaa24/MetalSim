#!/bin/bash
# kamino152.sh ID TREE   Newton 1.5.2-line Kamino metrics tests (unittest, as Newton runs them) on metal:0 and cpu
WT=/Users/aditya/robosim/.claude/worktrees/agent-a7a2cb997dc44ebcc
D=$WT/runs/rigid17
PY=/Users/aditya/robosim/.venv-newton152/bin/python
export PYTHONPATH=$WT/upstream/t4/warp-B:$WT/upstream/$2
export WARP_CACHE_PATH=$D/kcache/kcT_$2
for dev in metal:0 cpu; do
  tag=$1_kamino_${dev%%:*}
  /Users/aditya/robosim/scripts/gpu_run.sh "ntest_$tag" check 30 -- python3 $D/tools/watch.py --log $D/out/$tag.log \
    --timeout 3000 --stall 150 -- $PY $D/tools/ut_dev.py $dev \
    newton._src.solvers.kamino.tests.test_solvers_metrics > $D/out/$tag.watch 2>&1
  tail -n 3 $D/out/$tag.log
done
