#!/bin/bash
# repeat_check.py variants: W R OUTLOG
W=$1; R=$2
cd "$W" || exit 1
P="$W/metalsim/physics/mpm/probes/repeat_check.py"
for args in "--capture 1" "--capture 0" "--capture 0 --mass kinematic" "--capture 0 --worlds 1"; do
  echo "== $args"
  PYTHONPATH=$W python3 $W/scripts/watchdog.py --stall 150 --timeout 900 -- "$R/.venv/bin/python" "$P" --runs 3 $args 2>&1 | grep -E "^run|Error|Traceback"
done
