#!/bin/bash
W=.
cd $W/_forks/warp
export PYTHONPATH=$W/_forks/warp WARP_CACHE_PATH=$W/runs/mpm/approx/kcache_tests
.venv-newton152/bin/python -m unittest warp.tests.test_metal 2>&1 | /usr/bin/grep -E "^(OK|FAILED|Ran|ERROR:|FAIL:)"
