#!/bin/bash
# J3: the fix branch (worktree wt-fix). Tests, then the G1 flat probe alone, next to one compute load (the j2_c1
# condition that reproduced the corruption), and guard-only (old GPU-side hand-over, new error check) next to the load.
S=runs/gpu_watchdog_2026-11-01
PY=<repo>/.venv/bin/python
export PYTHONPATH=$S/wt-fix TORCH_EXTENSIONS_DIR=$S/torch_ext
cd $S/wt-fix
echo "== tests $(date +%H:%M:%S)"
$PY -m pytest -v -p no:cacheprovider tests/test_interop.py tests/test_ppo_warp_rollout.py > $S/j3_tests.out 2>&1; echo "== tests rc $? $(date +%H:%M:%S)"
probe() {  # name iterations
  rm -f $S/$1.log*
  echo "== $1 start $(date +%H:%M:%S)"
  $PY $S/g1_probe.py 4096 flat train $2 $S/$1.log - 0.0025 > $S/$1.out 2>&1; echo "== $1 rc $? $(date +%H:%M:%S)"
}
probe j3_fix_alone 2
$PY $S/load_compute.py 170 load1 > $S/j3_fix_c1_load.out 2>&1 & L=$!
sleep 3; probe j3_fix_c1 2; kill $L 2>/dev/null; wait $L 2>/dev/null
sleep 3
$PY $S/load_compute.py 100 load1 > $S/j3_guard_c1_load.out 2>&1 & L=$!
sleep 3; export PROBE_OLD_HANDOVER=1; probe j3_guard_c1 1; unset PROBE_OLD_HANDOVER; kill $L 2>/dev/null; wait $L 2>/dev/null
echo "== done $(date +%H:%M:%S)"
