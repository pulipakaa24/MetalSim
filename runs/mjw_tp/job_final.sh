#!/bin/bash
# final: committed state (MetalSim d867143 + Warp b9557cb + MuJoCo Warp ccfaffb/1791414) - cost split flat and rough,
# the original configuration re-measured in the same session (register bound 40, dense M, fork fusion off), rough checks, profile
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
OLD='{"metal_register_cholesky_max":40,"m_dense_max":64,"label":"original settings (fork fusion off)"}'
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
MJW_TP_VARIANT='{"label":"final flat"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_METAL_FUSE_H_CHOLESKY=0 MJW_TP_VARIANT="$OLD" python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_TP_TERRAIN=rough MJW_TP_VARIANT='{"label":"final rough"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_METAL_FUSE_H_CHOLESKY=0 MJW_TP_TERRAIN=rough MJW_TP_VARIANT="$OLD" python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
MJW_TP_VARIANT='{"label":"final flat repeat"}' python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
echo "=== end $(date)"
} > runs/mjw_tp/costsplit_final.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
WP_METAL_PROFILE=1 python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --kernels 2>&1 | sed -n '/top kernels/,$p'
MJW_METAL_FUSE_H_CHOLESKY=0 MJW_TP_TERRAIN=rough MJW_TP_VARIANT="$OLD" python scripts/diagnostics/g1_tp_check.py 512 50 --out runs/mjw_tp/traj_rough_orig.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]\|warn"
MJW_TP_TERRAIN=rough MJW_TP_VARIANT='{"label":"final rough"}' python scripts/diagnostics/g1_tp_check.py 512 50 --ref runs/mjw_tp/traj_rough_orig.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]\|warn"
echo "=== end $(date)"
} > runs/mjw_tp/final_checks.log 2>&1
