#!/bin/bash
cd "$(dirname "$0")/../.."
export SKIP_TESTS=1
cp runs/il3/run_train2.sh runs/il3/run_train2_ps.sh
runs/il3/run_train2_ps.sh rough rough_isaac recommended isaaclab3_every_substep_cap20 12 0 early_rough232task_it12
LOG=runs/il3/policy_side.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
python runs/il3/policy_stats.py init rough_il3 isaaclab3_every_substep_cap20 runs/il3/policy_init.json >> $LOG 2>&1
python runs/il3/policy_stats.py runs/il3/ckpt/early_fixed_it12.pt rough_il3 isaaclab3_every_substep_cap20 runs/il3/policy_il3_it12.json >> $LOG 2>&1
python runs/il3/policy_stats.py runs/il3/ckpt/early_rough232task_it12.pt rough_isaac isaaclab3_every_substep_cap20 runs/il3/policy_232_it12.json >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
