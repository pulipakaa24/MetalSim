#!/bin/bash
# step-1/2 unit tests (Metal) for the Isaac Lab 3.0 port and the isaaclab3 solver preset
cd "$(dirname "$0")/../.."
LOG=${1:-runs/il3/tests.log}
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
python -m pytest -q tests/test_solver_presets.py tests/test_g1_task_terms.py -p no:cacheprovider >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
