#!/bin/bash
# Vendored-solver bitwise check: column a=1 0.25 s in .venv-newton152 (Newton's solver, vendored solver) and .venv (vendored).
W=.
P="$W/research_notes/MPM in main simulator/probe_vendored_bitwise.py"
O=$W/runs/mpm_main/vendored_bitwise
SCENE=${1:-column}
cd "$W" || exit 1
PYTHONPATH=$W METALSIM_MPM_SOLVER=newton   .venv-newton152/bin/python "$P" "$O/n152_newton_$SCENE.json" "$SCENE" || exit 1
PYTHONPATH=$W METALSIM_MPM_SOLVER=vendored .venv-newton152/bin/python "$P" "$O/n152_vendored_$SCENE.json" "$SCENE" || exit 1
PYTHONPATH=$W                              .venv/bin/python "$P" "$O/venv_vendored_$SCENE.json" "$SCENE" || exit 1
