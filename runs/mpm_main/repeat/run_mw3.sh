#!/bin/bash
# GS multi-world repeatability under the fork's opt-out switches (.venv): W R
W=$1; R=$2
cd "$W" || exit 1
P="$W/metalsim/physics/mpm/probes/multiworld_repeat.py"
run() { echo "== $*"; env "$@" PYTHONPATH=$W python3 $W/scripts/watchdog.py --stall 150 --timeout 900 -- "$R/.venv/bin/python" "$P" --solvers gauss-seidel --bases S2 --cells auto --steps 2 --worlds 2 2>&1 | grep -E "^solver|worlds differing|Error|Traceback"; }
run WP_METAL_BOUNDED_LAUNCH=0
run NEWTON_METAL_LISTED_ROWS_MV=0 NEWTON_METAL_RIGIDITY_COMPACT=0
run WP_METAL_ICB=0
run WP_METAL_EAGER_CONDITIONALS=0 WP_METAL_ICB_BATCH=1
