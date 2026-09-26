#!/bin/bash
cd "$(dirname "$0")/../.."
LOG=runs/il3/fullsuite.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } > $LOG
source .venv/bin/activate
python -m pytest tests -q -x -p no:cacheprovider >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
