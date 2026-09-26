#!/bin/bash
cd "$(dirname "$0")/../.."
LOG=runs/il3/cap_probe2.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
for T in flat rough; do python runs/il3/cap_probe.py $T runs/il3/cap_probe2_$T.json >> $LOG 2>&1; done
echo "exit $? $(date)" >> $LOG
