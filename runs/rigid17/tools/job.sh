#!/bin/bash
# job.sh ID TREE KIND [guard options...] -- collide17 args   (one guarded run through the GPU queue; log out/ID.log)
D=runs/rigid17
ID=$1; TREE=$2; KIND=$3; shift 3
G=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do G+=("$1"); shift; done; shift
scripts/gpu_run.sh "r17_$ID" "$KIND" 20 -- python3 $D/tools/guarded.py --tree $TREE \
  --log $D/out/$ID.log "${G[@]}" -- "$@" --json $D/out/$ID.json > $D/out/$ID.guard 2>&1
rc=$?; cat $D/out/$ID.guard; tail -n 30 $D/out/$ID.log | /usr/bin/grep -v '^Module'; exit $rc
