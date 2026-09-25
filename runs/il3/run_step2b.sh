#!/bin/bash
cd "$(dirname "$0")/../.."
LOG=runs/il3/step2b.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
python -m pytest -q tests/test_solver_presets.py tests/test_g1_task_terms.py -p no:cacheprovider >> $LOG 2>&1
echo "tests exit $?" >> $LOG
for B in newton_mjwarp isaacsim_physx; do
python scripts/diagnostics/il3_transfer.py --backend $B --its 500,1000,1499 --presets default,hardlimits,isaaclab3,isaaclab3_collide_every_substep,isaaclab3_hardlimits \
   --json runs/il3/transfer_$B.json >> $LOG 2>&1
done
python runs/il3/niter_probe.py >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
