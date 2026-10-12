#!/bin/bash
# rigid_neutral.py with the closeout Warp tree (warp_push, 43ffc6dd, built) with metal_cuda_libm off (the kernel code
# of 152808b7 for everything outside warp.fem) and on, twice each (run-to-run repeatability first), own kernel caches.
W=/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c
PY=/Users/aditya/robosim/.venv/bin/python
S=$W/research_notes/mpm_parity_closeout/rigid_neutral.py
F='^Module\|Warp \|CUDA\|Devices\|"\|Kernel cache\|^ *$\|/Users\|warn'
cd /private/tmp
for rep in 1 2; do
  echo "== metal_cuda_libm off, run $rep"
  WP_METAL_CUDA_LIBM=0 PYTHONPATH=$W/_forks/warp_push:$W WARP_CACHE_PATH=$W/runs/mpm/closeout/kcache_rigid_off $PY $S 2>&1 | /usr/bin/grep -v "$F"
  echo "== metal_cuda_libm on, run $rep"
  PYTHONPATH=$W/_forks/warp_push:$W WARP_CACHE_PATH=$W/runs/mpm/closeout/kcache_rigid_on $PY $S 2>&1 | /usr/bin/grep -v "$F"
done
