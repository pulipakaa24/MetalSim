#!/bin/bash
# scripts/gpu_run.sh NAME KIND MINUTES -- command args...   (KIND: timing | render | train)
# Acquires the GPU through the priority queue, runs the command, releases on any exit. The lock records the
# job's own pid (not this wrapper's), and the job is killed if this wrapper is terminated, so a cut-off tool
# call cannot leave a running job with an abandoned-looking lock.
set -u
NAME=$1; KIND=$2; MIN=$3; shift 3; [ "$1" = "--" ] && shift
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py acquire "$NAME" --kind "$KIND" --minutes "$MIN" --pid $$
"$@" &
CHILD=$!
python3 scripts/gpu_lock.py setpid "$NAME" --pid $CHILD
trap 'kill $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; python3 scripts/gpu_lock.py release "$NAME" --pid $CHILD' EXIT INT TERM HUP
wait $CHILD
