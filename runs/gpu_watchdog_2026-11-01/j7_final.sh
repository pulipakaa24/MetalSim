#!/bin/bash
# J7: final fix branch: tests touching the bridge / PPOWarp, then the rough probe with the fix next to one compute load.
S=runs/gpu_watchdog_2026-11-01
PY=<repo>/.venv/bin/python
export PYTHONPATH=$S/wt-fix TORCH_EXTENSIONS_DIR=$S/torch_ext
cd $S/wt-fix
echo "== tests $(date +%H:%M:%S)"
$PY -m pytest -v -p no:cacheprovider tests/test_interop.py tests/test_ppo_warp_rollout.py tests/test_warp_policy.py tests/test_lift_ctrl_ordering.py > $S/j7_tests.out 2>&1; echo "== tests rc $? $(date +%H:%M:%S)"
$PY $S/load_compute.py 240 load1 > $S/j7_rough_c1_load.out 2>&1 & L=$!
sleep 3
rm -f $S/j7_rough_c1.log*
echo "== j7_rough_c1 start $(date +%H:%M:%S)"
$PY $S/g1_probe.py 4096 rough train 2 $S/j7_rough_c1.log - 0.0025 --reward_cfg rough_isaac > $S/j7_rough_c1.out 2>&1; echo "== j7_rough_c1 rc $? $(date +%H:%M:%S)"
kill $L 2>/dev/null; wait $L 2>/dev/null
echo "== done $(date +%H:%M:%S)"
