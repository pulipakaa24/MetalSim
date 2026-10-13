#!/bin/bash
# Re-run of the overflow harness's trained-policy rows with the PPOWarp load fix (2026-10-13), 16-inch, check jobs.
# Checkpoint: the 2026-09-26 elliptic-default rough run (its settings = today's defaults).
W=/Users/aditya/robosim/.claude/worktrees/agent-a3214b300d0b6e295
PY=/Users/aditya/robosim/.venv/bin/python
CK=/Users/aditya/robosim/runs/ellip_default/g1_rough_ellip_default_s0.pt
H=$W/scripts/diagnostics/overflow_training_check.py
O=$W/runs/g1regress
for spec in "g1_rough_policy:" "g1_rough_policy_hfield:" "g1_rough_policy:--freeze_levels" "g1_rough_policy_hfield:--freeze_levels"; do
  t=${spec%%:*}; f=${spec#*:}; tag=$t${f:+_frozen}
  /Users/aditya/robosim/scripts/gpu_run.sh g1r_$tag check 15 -- $PY $H $t --n 4096 --iters 150 --ckpt $CK $f --log $O/fixed_$tag.log > /dev/null 2>&1
  echo "$tag rc=$? $(tail -1 $O/fixed_$tag.log)"
  ioreg -r -c AGXAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'
  sleep 5; ioreg -r -c AGXAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'
done
