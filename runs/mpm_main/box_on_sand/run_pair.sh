#!/bin/bash
# box_on_sand: MetalSim main simulator (.venv) vs Isaac's coupling path (Newton 1.5.2 SolverCoupledProxy, .venv-newton152).
# Usage: run_pair.sh WORKTREE METALSIM_CLONE OUTDIR TAG [box_on_sand.py args...]
W=$1; R=$2; O=$3; TAG=$4; shift 4
mkdir -p "$O"
P="$W/metalsim/physics/mpm/probes/box_on_sand.py"
WD="python3 $W/scripts/watchdog.py --stall 120 --timeout 1500 --"
cd "$W" || exit 1
PYTHONPATH=$W $WD "$R/.venv/bin/python" "$P" --engine metalsim --out "$O/${TAG}_metalsim.npz" "$@" || exit 1
PYTHONPATH=$W $WD "$R/.venv-newton152/bin/python" "$P" --engine newton --out "$O/${TAG}_newton.npz" "$@" || exit 1
"$R/.venv/bin/python" "$P" --compare "$O/${TAG}_metalsim.npz" "$O/${TAG}_newton.npz"
