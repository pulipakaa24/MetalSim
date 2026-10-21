#!/bin/bash
# probe + compose + Warp tests, final header
W=.
$W/runs/mpm/approx/tools/job_probe.sh --only "cuda emulation"
cd $W/_forks/warp && PYTHONPATH=$W/_forks/warp WARP_CACHE_PATH=$W/runs/mpm/approx/kcache .venv-newton152/bin/python -m unittest warp.tests.test_metal.TestMetal.test_fast_math_approximates_division_and_sqrt_only warp.tests.test_metal.TestMetal.test_fast_math_cuda_mode_matches_l4 2>&1 | /usr/bin/grep -E "^(OK|FAILED|Ran|ERROR|FAIL)"
