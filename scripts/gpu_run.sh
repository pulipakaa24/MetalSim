#!/bin/bash
# scripts/gpu_run.sh NAME KIND MINUTES -- command args...   (KIND: timing | render | train)
# Acquires the GPU through the priority queue, runs the command, releases on any exit.
set -u
NAME=$1; KIND=$2; MIN=$3; shift 3; [ "$1" = "--" ] && shift
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py acquire "$NAME" --kind "$KIND" --minutes "$MIN" --pid $$
trap 'python3 scripts/gpu_lock.py release "$NAME"' EXIT
"$@"
