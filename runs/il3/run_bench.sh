#!/bin/bash
# throughput of the G1 flat task variants (env step incl. rewards/resets/obs, 4096 envs, synchronized; module CLI)
cd "$(dirname "$0")/../.."
LOG=runs/il3/bench.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
for V in ":" "--reward_cfg flat_il3:" "--reward_cfg flat_il3:--solver_cfg isaaclab3" "--reward_cfg flat_il3:--solver_cfg isaaclab3_cap20" \
         "--reward_cfg flat_il3:--solver_cfg isaaclab3_collide_every_substep" "--reward_cfg flat_il3:--solver_cfg hardlimits"; do
  A=${V%%:*}; B=${V#*:}
  echo "-- variant [$A $B] $(date +%T)" >> $LOG
  python -m metalsim.learn.g1_velocity 4096 flat 0.0025 $A $B >> $LOG 2>&1
done
echo "exit $? $(date)" >> $LOG
