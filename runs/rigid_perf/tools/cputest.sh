#!/bin/bash
# CPU-only Newton unit tests on the final fork (Newton's test device list has no Metal device)
N=/Users/aditya/robosim/.claude/worktrees/agent-a6ef6785c0d23ee50/_forks/newton
cd $N
export PYTHONPATH=/Users/aditya/robosim/upstream/warp-innate-integrated:$N
export WARP_CACHE_PATH=/Users/aditya/robosim/.claude/worktrees/agent-a6ef6785c0d23ee50/runs/rigid_perf/kcache/kc_cputest
exec /Users/aditya/robosim/.venv-newton152/bin/python "$@"
