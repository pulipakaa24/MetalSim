#!/bin/bash
W=.
export PYTHONPATH=$W/_forks/warp
export WARP_CACHE_PATH=$W/runs/mpm/approx/kcache
exec .venv-newton152/bin/python $W/metalsim/physics/mpm/probes/metal_approx_timing.py --json $W/runs/mpm/approx/op_timing.json
