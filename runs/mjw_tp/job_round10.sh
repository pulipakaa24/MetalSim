#!/bin/bash
# round 10: blocked (16-wide tiles) vs single register Cholesky for the 43-dof Newton Hessian; baseline re-measure after the revert
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
MJW_TP_VARIANT='{"label":"default (steps 1-4)"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_METAL_DENSE_CHOL_MAX=32 MJW_TP_VARIANT='{"label":"blocked Hessian Cholesky"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_TP_VARIANT='{"label":"default repeat"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
echo "=== end $(date)"
} > runs/mjw_tp/costsplit10.log 2>&1
