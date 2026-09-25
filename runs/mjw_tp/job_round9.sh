#!/bin/bash
# round 9: step 5 candidate = Warp fork: compact per-lane storage in the Metal register Cholesky (n in (32, 64])
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
echo "=== register Cholesky cost (after; before: runs/mjw_tp/cholcost9.log)"
python scripts/diagnostics/metal_cholesky_cost.py 2>&1 | grep "n=\|rror"
echo "=== Warp fork Metal Cholesky tests"
(cd upstream/warp-innate && python -m pytest warp/tests/test_metal.py -k cholesky -q 2>&1 | tail -3)
MJW_TP_VARIANT='{"label":"step5 compact register Cholesky"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_TP_VARIANT='{"label":"step5 repeat"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
echo "=== end $(date)"
} > runs/mjw_tp/costsplit9.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
MJW_TP_VARIANT='{"label":"step5"}' python scripts/diagnostics/g1_tp_check.py 512 50 --ref runs/mjw_tp/traj_step3.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
python -m pytest tests/test_g1_parity.py tests/test_g1_task_terms.py tests/test_physics.py tests/test_g1_fast_factorization.py -q 2>&1 | tail -2
MJW_TP_VARIANT='{"label":"step5"}' python scripts/diagnostics/g1_tp_train_probe.py 4096 20 2>&1 | grep "^variant\|^it \|Error\|Traceback"
echo "=== end $(date)"
} > runs/mjw_tp/checks9.log 2>&1
