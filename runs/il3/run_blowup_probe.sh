#!/bin/bash
cd "$(dirname "$0")/../.."
LOG=runs/il3/blowup_probe.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
for P in isaaclab3_mixed_recommended_limits isaaclab3_every_substep_cap20 isaaclab3_collide_every_substep; do
  echo "-- rough $P $(date +%T)" >> $LOG
  python runs/il3/blowup_probe.py rough runs/il3/ckpt/train_rough_il3_isaaclab3_every_substep_cap20_s0.pt $P 500 runs/il3/blowup_rough_$P.json >> $LOG 2>&1
done
for P in isaaclab3_mixed_recommended_limits isaaclab3_every_substep_cap20; do
  echo "-- flat $P $(date +%T)" >> $LOG
  python runs/il3/blowup_probe.py flat runs/il3/ckpt/train_flat_il3_isaaclab3_every_substep_cap20_s0.pt $P 500 runs/il3/blowup_flat_$P.json >> $LOG 2>&1
done
echo "exit $? $(date)" >> $LOG
