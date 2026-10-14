#!/bin/bash
# multiworld_repeat.py in both environments: W R [args]
W=$1; R=$2; shift 2
cd "$W" || exit 1
P="$W/metalsim/physics/mpm/probes/multiworld_repeat.py"
for v in .venv-newton152 .venv; do
  echo "== $v"
  PYTHONPATH=$W python3 $W/scripts/watchdog.py --stall 150 --timeout 1500 -- "$R/$v/bin/python" "$P" "$@" 2>&1 | grep -E "^newton|^solver|Error|Traceback"
done
