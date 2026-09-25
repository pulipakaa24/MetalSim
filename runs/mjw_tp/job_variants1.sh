#!/bin/bash
# variant scan 1: capacity, Jacobian layout, post-reset kinematics, Warp Metal knobs (physics + full step, --quick)
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date) power: $(pmset -g batt | head -1)"
python3 scripts/gpu_lock.py status
S=scripts/diagnostics/g1_tp_variants.py
MJW_TP_VARIANT='{}' python $S 4096 --quick
MJW_TP_VARIANT='{"njmax":128}' python $S 4096 --quick
MJW_TP_VARIANT='{"njmax":128,"nconmax":16}' python $S 4096 --quick
MJW_TP_VARIANT='{"jacobian":"sparse"}' python $S 4096 --quick
MJW_TP_VARIANT='{"no_kin":true}' python $S 4096 --quick
WP_METAL_INFLIGHT=16 MJW_TP_VARIANT='{}' python $S 4096 --quick
WP_METAL_INFLIGHT=256 MJW_TP_VARIANT='{}' python $S 4096 --quick
WP_METAL_ICB_BATCH=256 MJW_TP_VARIANT='{}' python $S 4096 --quick
MJW_TP_VARIANT='{"label":"baseline repeat"}' python $S 4096 --quick
echo "=== end $(date)"
