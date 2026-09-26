#!/bin/bash
# late rough blow-ups: same final policy, 1500 control steps, per limit model (finger limits are the suspects)
cd "$(dirname "$0")/../.."
LOG=runs/il3/late_ab.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
for P in isaaclab3_every_substep_cap20_hardlimits contact:default isaaclab3_mixed_recommended_limits; do
  echo "-- $P $(date +%T)" >> $LOG
  python runs/il3/blowup_probe.py rough runs/il3/ckpt/il3fix_rough_s0.pt $P 1500 runs/il3/blowup_late_rough_${P/:/_}.json >> $LOG 2>&1
done
echo "exit $? $(date)" >> $LOG
