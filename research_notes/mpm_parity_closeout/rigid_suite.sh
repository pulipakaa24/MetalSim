#!/bin/bash
# rigid_suite.sh WARP_TREE TAG [extra env assignments are taken from the environment]
# runs/fma_rigid_parity/harness/run_suite.sh for an arbitrary Warp tree path: full tests/ minus test_mpm.py with the
# locals-capture plugin, into runs/fma_rigid_parity/<TAG>.jsonl of this worktree (metrics.py compares the files).
W=/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c
S=$W/research_notes/mpm_parity_closeout/rigid_cl
TREE=$1; TAG=$2
cd /Users/aditya/robosim   # the main checkout's tests and fetched assets, as run_suite.sh (read only)
export PYTHONPATH=$TREE:.:$S WARP_CACHE_PATH=$W/runs/mpm/closeout/kcache_rigid_$(basename $TREE) CAPTURE_OUT=$W/runs/fma_rigid_parity/$TAG.jsonl
rm -f $CAPTURE_OUT
/Users/aditya/robosim/.venv/bin/python -c "import warp as wp; wp.init(); print('WARP', wp.__file__, wp.config.kernel_cache_dir, wp.config.metal_cuda_libm if hasattr(wp.config, 'metal_cuda_libm') else '-')"
/Users/aditya/robosim/.venv/bin/python -m pytest -p capture_locals -q -rA --ignore=tests/test_mpm.py -p no:cacheprovider tests
