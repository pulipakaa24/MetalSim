#!/bin/bash
# late rough blow-ups: final corrected-rough policy, 1500 control steps, per variant
cd "$(dirname "$0")/../.."
LOG=runs/il3/late_ab.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
for P in isaaclab3_every_substep_cap20 isaaclab3_every_substep_cap20_hardfingers isaaclab3_every_substep_cap20_hardlimits \
         isaaclab3_every_substep_cap20+mjwfactor isaaclab3_every_substep_cap20_jointeffort contact:default; do
  echo "-- $P $(date +%T)" >> $LOG
  python runs/il3/blowup_probe.py rough runs/il3/ckpt/il3fix_rough_s0.pt $P 1500 runs/il3/blowup_late_rough_${P//[:+]/_}.json >> $LOG 2>&1
done
echo "exit $? $(date)" >> $LOG
