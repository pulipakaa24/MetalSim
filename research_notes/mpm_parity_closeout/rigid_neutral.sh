#!/bin/bash
# rigid_neutral.py with the pinned Warp (152808b7, upstream/warp-innate) and with the closeout commits (warp_push),
# each with its own kernel cache; then the closeout tree with WP_METAL_CUDA_LIBM=0.
W=/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c
PY=/Users/aditya/robosim/.venv/bin/python
S=$W/research_notes/mpm_parity_closeout/rigid_neutral.py
cd /private/tmp
echo "== pinned 152808b7"
PYTHONPATH=/Users/aditya/robosim/upstream/warp-innate:$W WARP_CACHE_PATH=$W/runs/mpm/closeout/kcache_rigid_pin $PY $S 2>&1 | /usr/bin/grep -v "^Module\|Warp \|CUDA\|Devices\|\"\|Kernel cache\|^ *$\|/Users"
echo "== closeout (metal_cuda_libm on)"
PYTHONPATH=$W/_forks/warp_push:$W WARP_CACHE_PATH=$W/runs/mpm/closeout/kcache_rigid_new $PY $S 2>&1 | /usr/bin/grep -v "^Module\|Warp \|CUDA\|Devices\|\"\|Kernel cache\|^ *$\|/Users"
echo "== closeout (metal_cuda_libm off)"
WP_METAL_CUDA_LIBM=0 PYTHONPATH=$W/_forks/warp_push:$W WARP_CACHE_PATH=$W/runs/mpm/closeout/kcache_rigid_new $PY $S 2>&1 | /usr/bin/grep -v "^Module\|Warp \|CUDA\|Devices\|\"\|Kernel cache\|^ *$\|/Users"
