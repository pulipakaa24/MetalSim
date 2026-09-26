#!/bin/bash
cd "$(dirname "$0")/../.."
LOG=runs/il3/late_classify.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
python -m pytest -q tests/test_g1_task_terms.py tests/test_ppo_warp_rollout.py tests/test_solver_presets.py -p no:cacheprovider >> $LOG 2>&1; echo "tests exit $?" >> $LOG
# late rough blow-ups: final corrected-rough policy, 1500 control steps (30 s: pushes at 10-15 s included)
python runs/il3/blowup_probe.py rough runs/il3/ckpt/il3fix_rough_s0.pt isaaclab3_every_substep_cap20 1500 runs/il3/blowup_late_rough.json >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
