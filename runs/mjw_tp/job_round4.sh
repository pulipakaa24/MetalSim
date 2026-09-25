#!/bin/bash
# round 4: profile of the accepted configuration (step 2) to rank the next levers; rough-terrain cost split and checks
cd /Users/aditya/robosim && source .venv/bin/activate
S2='{"metal_register_cholesky_max":48,"m_dense_max":0,"label":"step2"}'
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
MJW_TP_VARIANT="$S2" python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
MJW_TP_VARIANT="$S2" WP_METAL_PROFILE=1 python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --kernels 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
echo "=== end $(date)"
} > runs/mjw_tp/profile4.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
for V in '{"label":"rough base"}' "$S2"; do
  MJW_TP_TERRAIN=rough MJW_TP_VARIANT="$V" python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep '^\[\|Error\|Traceback'
done
C=scripts/diagnostics/g1_tp_check.py
MJW_TP_TERRAIN=rough MJW_TP_VARIANT='{}' python $C 512 50 --out runs/mjw_tp/traj_rough_base.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
MJW_TP_TERRAIN=rough MJW_TP_VARIANT="$S2" python $C 512 50 --ref runs/mjw_tp/traj_rough_base.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
echo "=== end $(date)"
} > runs/mjw_tp/rough4.log 2>&1
