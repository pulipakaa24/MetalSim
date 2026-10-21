#!/bin/bash
# GPU jobs for the tilted-dome change (run through scripts/gpu_run.sh; logs next to this file).
#   job.sh tiny    : only the new FLAG_ENVROT lookup test (tiny-first, under a watchdog)
#   job.sh suite   : byte comparison against origin/main c3a25d9d, then tests/test_render_tier2.py and the render tests
WT=.
PY=.venv/bin/python
SNAP=${SNAP:-/private/tmp/claude-501/-Users-aditya-robosim/3a553738-d88f-4d8d-ac36-4ecff3ac4bc1/scratchpad/main_snap}
cd $WT
WD="python3 $WT/scripts/watchdog.py --stall 180 --rss-gb 32 --cwd $WT"
case "$1" in
  tiny)
    PYTHONPATH=$WT $WD --timeout 600 -- $PY -m pytest -x -s -q tests/test_render_tier2.py -k tilted_dome_lookup -p no:cacheprovider
    ;;
  suite)
    rc=0
    PYTHONPATH=$SNAP $WD --timeout 900 -- $PY scripts/diagnostics/dome_untilted_bytes.py --out $SNAP/../untilted_main.npz || rc=1
    PYTHONPATH=$WT $WD --timeout 900 -- $PY scripts/diagnostics/dome_untilted_bytes.py --out $SNAP/../untilted_branch.npz --compare $SNAP/../untilted_main.npz || rc=1
    PYTHONPATH=$WT $WD --timeout 3000 -- $PY -m pytest -s -q -rA tests/test_render_tier2.py -p no:cacheprovider || rc=1
    shift
    if [ $# -gt 0 ]; then PYTHONPATH=$WT $WD --timeout 5400 -- $PY -m pytest -q -rfE "$@" -p no:cacheprovider || rc=1; fi
    exit $rc
    ;;
esac
