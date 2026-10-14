#!/bin/bash
# GS multi-world repeatability: offsets and world counts (.venv): W R
W=$1; R=$2
cd "$W" || exit 1
P="$W/metalsim/physics/mpm/probes/multiworld_repeat.py"
for args in "--worlds 2 --offset 0" "--worlds 2 --offset 10" "--worlds 4 --offset 0"; do
  echo "== $args"
  PYTHONPATH=$W python3 $W/scripts/watchdog.py --stall 150 --timeout 900 -- "$R/.venv/bin/python" "$P" --solvers gauss-seidel --bases S2 --cells auto --steps 3 $args 2>&1 | grep -E "^solver|worlds differing|Error|Traceback"
done
