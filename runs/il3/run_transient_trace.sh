#!/bin/bash
cd "$(dirname "$0")/../.."
LOG=runs/il3/transient_trace.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
for A in "rough isaaclab3_every_substep_cap20 noreset" "rough isaaclab3_every_substep_cap20 reset" "rough contact:default noreset" "flat isaaclab3_every_substep_cap20 noreset"; do
  set -- $A; echo "-- $A $(date +%T)" >> $LOG
  python runs/il3/transient_trace.py $1 $2 $3 320 runs/il3/transient_$1_${2/:/_}_$3.npz >> $LOG 2>&1
done
echo "exit $? $(date)" >> $LOG
