#!/bin/bash
# round 2: factorization levers (register Cholesky bound, sparse L'DL of M, sparse Jacobian): scan + checks
cd /Users/aditya/robosim && source .venv/bin/activate
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"
python3 scripts/gpu_lock.py status
S=scripts/diagnostics/g1_tp_variants.py
for V in '{}' '{"metal_register_cholesky_max":48}' '{"m_dense_max":0}' '{"m_dense_max":0,"metal_register_cholesky_max":48}' \
         '{"jacobian":"sparse"}' '{"jacobian":"sparse","m_dense_max":0,"metal_register_cholesky_max":48}' '{"label":"baseline repeat"}'; do
  MJW_TP_VARIANT="$V" python $S 4096 --quick 2>&1 | grep "^\[\|Error\|Traceback"
done
echo "=== end $(date)"
} > runs/mjw_tp/variants2.log 2>&1
{
echo "=== start $(date)"
python3 scripts/gpu_lock.py status
C=scripts/diagnostics/g1_tp_check.py
MJW_TP_VARIANT='{}' python $C 512 50 --out runs/mjw_tp/traj_base.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]"
for V in '{"metal_register_cholesky_max":48}' '{"m_dense_max":0}' '{"jacobian":"sparse"}'; do
  MJW_TP_VARIANT="$V" python $C 512 50 --ref runs/mjw_tp/traj_base.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]"
done
echo "=== end $(date)"
} > runs/mjw_tp/checks2.log 2>&1
