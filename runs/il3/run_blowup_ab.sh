#!/bin/bash
# early-phase (initial policy, std 1) rough_il3 blow-ups per single-item variant, 300 control steps x 4096 envs each
cd "$(dirname "$0")/../.."
LOG=runs/il3/blowup_ab.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
for P in isaaclab3_every_substep_cap20 isaaclab3_every_substep_cap20_hardlimits isaaclab3_every_substep_cap20_nogap \
         isaaclab3_every_substep_cap20_mjcontact isaaclab3_collide_every_substep contact:default contact:recommended; do
  echo "-- rough init $P $(date +%T)" >> $LOG
  python runs/il3/blowup_probe.py rough init $P 300 runs/il3/blowup_ab_rough_${P/:/_}.json >> $LOG 2>&1
done
echo "exit $? $(date)" >> $LOG
