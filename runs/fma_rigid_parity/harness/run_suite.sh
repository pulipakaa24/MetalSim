#!/bin/bash
# run_suite.sh TREE TAG [VENV] [pytest args...]: full tests/ minus test_mpm.py with locals capture
S=/private/tmp/claude-501/-Users-aditya-robosim/3a553738-d88f-4d8d-ac36-4ecff3ac4bc1/scratchpad/rigid_fma
TREE=$1; TAG=$2; VENV=${3:-.venv}; shift 3
cd .
export PYTHONPATH=upstream/$TREE:.:$S WARP_CACHE_PATH=$S/cache_$TREE CAPTURE_OUT=runs/fma_rigid_parity/$TAG.jsonl
rm -f $CAPTURE_OUT
$VENV/bin/python -c "import warp as wp; wp.init(); print('WARP', wp.__file__, wp.config.kernel_cache_dir)"
$VENV/bin/python -m pytest -p capture_locals -q -rA --ignore=tests/test_mpm.py -p no:cacheprovider "${@:-tests}"
