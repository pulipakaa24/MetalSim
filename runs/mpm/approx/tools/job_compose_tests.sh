#!/bin/bash
W=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
P=$W/metalsim/physics/mpm/probes
O=$W/runs/mpm/approx
PY=/Users/aditya/robosim/.venv-newton152/bin/python
export PYTHONPATH=$W/_forks/warp
export WARP_CACHE_PATH=$O/kcache
$PY $P/metal_compose_check.py --json $O/compose_metal.json
cd $W/_forks/warp && $PY -m unittest warp.tests.test_metal.TestMetal.test_fast_math_approximates_division_and_sqrt_only warp.tests.test_metal.TestMetal.test_fast_math_cuda_mode_matches_l4 2>&1 | tail -15
