#!/bin/bash
# suite152b.sh ID TREE   1.5.2 line: Kamino metrics (unittest, metal:0 and cpu) and the collision / hashtable / SDF tests
# (cpu + metal:0), without test_contact_reduction_global.py (that one runs alone, exclusive)
WT=/Users/aditya/robosim/.claude/worktrees/agent-a7a2cb997dc44ebcc
D=$WT/runs/rigid17
T=$WT/upstream/$2/newton
PY=/Users/aditya/robosim/.venv-newton152/bin/python
$D/tools/kamino152.sh "$1_k" "$2"
export PYTHONPATH=$WT/upstream/t4/warp-B:$WT/upstream/$2
export WARP_CACHE_PATH=$D/kcache/kcT_$2
/Users/aditya/robosim/scripts/gpu_run.sh "ntest_$1" check 30 -- python3 $D/tools/watch.py --log $D/out/$1.log \
  --timeout 3000 --stall 150 -- $PY $D/tools/pt_all.py -q -p no:cacheprovider -rfE --junitxml=$D/out/$1.xml \
  $T/tests/test_hashtable.py $T/tests/test_collision_pipeline.py $T/tests/test_sdf_contact.py > $D/out/$1.watch 2>&1
tail -n 3 $D/out/$1.log
