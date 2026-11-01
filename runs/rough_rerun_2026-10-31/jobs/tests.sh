#!/bin/bash
# tests.sh MODE   (MODE full | smoke). Research note §9 item 1: tests/test_terrain.py on the new pin (-s prints the
# heightfield-contact figures), plus the gates the original training scripts ran before training:
#   runs/g1_rough_ppowarp_fixed.sh: pytest tests/test_g1_task_terms.py -q -k rough_isaac
#   runs/il3/run_train3.sh:         pytest -q tests/test_solver_presets.py tests/test_g1_task_terms.py -p no:cacheprovider (a superset)
# Run once here instead of before every training job (the code does not change between jobs; status.jsonl records it).
set -u
trap 'kill -TERM $P 2>/dev/null; wait $P 2>/dev/null; exit 143' TERM INT HUP
PY=.venv/bin/python; RC=0
run() { echo "--- $(date +%T) $*"; "$@" & P=$!; wait $P; local rc=$?; echo "--- rc $rc"; [ $rc -ne 0 ] && RC=$rc; }
if [ "${1:-full}" = smoke ]; then
  run $PY -m pytest -q -s tests/test_terrain.py -k hfield_mesh_contacts -p no:cacheprovider
else
  run $PY -m pytest -q -s tests/test_terrain.py -p no:cacheprovider
  run $PY -m pytest -q tests/test_solver_presets.py tests/test_g1_task_terms.py -p no:cacheprovider
fi
exit $RC
