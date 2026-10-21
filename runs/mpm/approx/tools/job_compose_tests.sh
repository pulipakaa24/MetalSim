#!/bin/bash
W=.
P=$W/metalsim/physics/mpm/probes
O=$W/runs/mpm/approx
PY=.venv-newton152/bin/python
export PYTHONPATH=$W/_forks/warp
export WARP_CACHE_PATH=$O/kcache
$PY $P/metal_compose_check.py --json $O/compose_metal.json
cd $W/_forks/warp && $PY -m unittest warp.tests.test_metal.TestMetal.test_fast_math_approximates_division_and_sqrt_only warp.tests.test_metal.TestMetal.test_fast_math_cuda_mode_matches_l4 2>&1 | tail -15
