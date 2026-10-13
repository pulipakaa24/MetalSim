#!/bin/bash
# suite152.sh ID TREE   Newton 1.5.2-line Kamino metrics tests (default device metal:0, then cpu) and the collision /
# contact tests of upstream/TREE (cpu + metal:0), .venv-newton152 python, pinned Warp build
WT=/Users/aditya/robosim/.claude/worktrees/agent-a7a2cb997dc44ebcc
D=$WT/runs/rigid17
T=$WT/upstream/$2/newton
PY=/Users/aditya/robosim/.venv-newton152/bin/python
export PYTHONPATH=$WT/upstream/t4/warp-B:$WT/upstream/$2
export WARP_CACHE_PATH=$D/kcache/kcT_$2
for dev in metal:0 cpu; do
  tag=$1_kamino_${dev%%:*}
  /Users/aditya/robosim/scripts/gpu_run.sh "ntest_$tag" check 30 -- python3 $D/tools/watch.py --log $D/out/$tag.log \
    --timeout 3000 --stall 150 -- $PY $D/tools/pt_dev.py $dev -q -p no:cacheprovider -rfE --junitxml=$D/out/$tag.xml \
    $T/_src/solvers/kamino/tests/test_solvers_metrics.py > $D/out/$tag.watch 2>&1
  tail -n 3 $D/out/$tag.log
done
/Users/aditya/robosim/scripts/gpu_run.sh "ntest_$1" check 30 -- python3 $D/tools/watch.py --log $D/out/$1.log \
  --timeout 3000 --stall 150 -- $PY $D/tools/pt_all.py -q -p no:cacheprovider -rfE --junitxml=$D/out/$1.xml \
  $T/tests/test_contact_reduction_global.py $T/tests/test_hashtable.py $T/tests/test_collision_pipeline.py \
  $T/tests/test_sdf_contact.py > $D/out/$1.watch 2>&1
tail -n 3 $D/out/$1.log
