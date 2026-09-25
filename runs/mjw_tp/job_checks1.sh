#!/bin/bash
# correctness checks for the scan-1 candidates (replay==eager, variant vs baseline vs floor, MuJoCo C protocol, capacity)
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date)"
python3 scripts/gpu_lock.py status
C=scripts/diagnostics/g1_tp_check.py
MJW_TP_VARIANT='{}' python $C 512 50
MJW_TP_VARIANT='{"njmax":128,"nconmax":16}' python $C 512 50
MJW_TP_VARIANT='{"jacobian":"sparse"}' python $C 512 50
MJW_TP_VARIANT='{"no_kin":true}' python $C 512 50
echo "=== end $(date)"
