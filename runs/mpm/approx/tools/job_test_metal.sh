#!/bin/bash
W=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
cd $W/_forks/warp
export PYTHONPATH=$W/_forks/warp WARP_CACHE_PATH=$W/runs/mpm/approx/kcache_tests
/Users/aditya/robosim/.venv-newton152/bin/python -m unittest warp.tests.test_metal 2>&1 | /usr/bin/grep -E "^(OK|FAILED|Ran|ERROR:|FAIL:)"
