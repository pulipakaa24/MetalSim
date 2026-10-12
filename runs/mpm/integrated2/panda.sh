#!/bin/bash
# Franka panda (tendons) one step on the final Warp tree per WP_METAL_DEVICE_ARGS mode, fresh kernel cache each
A=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/runs/a2
for m in auto 1 0; do
  rm -rf $A/kc/panda_$m
  WP_METAL_DEVICE_ARGS=$m PYTHONPATH=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/upstream/warp-i2 \
    WARP_CACHE_PATH=$A/kc/panda_$m /Users/aditya/robosim/.venv-newton152/bin/python $A/py/tendon_repro.py > $A/out/panda_$m.log 2>&1
  echo "mode $m: $(/usr/bin/grep RESULT $A/out/panda_$m.log | cut -c1-80) | fallback warnings: $(/usr/bin/grep -c 'device-memory argument form' $A/out/panda_$m.log)"
done
