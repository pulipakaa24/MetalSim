#!/bin/bash
# job.sh ID TREE [guard options...] -- scale_pour args   (one guarded config; log out/ID.log, json out/ID.json)
D=/Users/aditya/robosim/.claude/worktrees/agent-a6ef6785c0d23ee50/runs/rigid_perf
ID=$1; TREE=$2; shift 2
G=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do G+=("$1"); shift; done; shift
python3 $D/tools/guarded.py --tree $TREE --log $D/out/$ID.log "${G[@]}" -- "$@" --json $D/out/$ID.json > $D/out/$ID.guard 2>&1
rc=$?; cat $D/out/$ID.guard; exit $rc
