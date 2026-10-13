#!/bin/bash
# ntest.sh ID TREE PY -- pytest args   Newton's own tests of a Newton tree (upstream/TREE in this worktree) with the
# pinned Warp fork build (upstream/t4/warp-B, 152808b7), devices cpu + metal:0 (Newton test mode "all"), through the
# GPU queue (check) and under the watchdog. PY: python of the environment (.venv-newtonfork for the 1.7 line,
# .venv-newton152 for the 1.5.2 line). Log: out/ID.log
WT=/Users/aditya/robosim/.claude/worktrees/agent-a7a2cb997dc44ebcc
D=$WT/runs/rigid17
ID=$1; TREE=$2; PY=$3; shift 3; [ "$1" = "--" ] && shift
export PYTHONPATH=$WT/upstream/t4/warp-B:$WT/upstream/$TREE
export WARP_CACHE_PATH=$D/kcache/kcT_$TREE
cd $WT/upstream/$TREE
/Users/aditya/robosim/scripts/gpu_run.sh "ntest_$ID" check 30 -- python3 $D/tools/watch.py --log $D/out/$ID.log \
  --timeout 3000 --stall 150 -- $PY $D/tools/pt_all.py -q -p no:cacheprovider "$@" > $D/out/$ID.watch 2>&1
rc=$?; cat $D/out/$ID.watch; tail -n 15 $D/out/$ID.log; exit $rc
