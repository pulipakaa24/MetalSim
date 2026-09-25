#!/bin/bash
# Step 2: fidelity protocol (hold / random / drop) against Isaac Lab 3.0's Newton/MuJoCo-Warp recording, per preset,
# then the transfer of Isaac's 3.0 Newton checkpoints, then throughput of the task variants.
cd "$(dirname "$0")/../.."
LOG=runs/il3/fidelity.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } >> $LOG
source .venv/bin/activate
REF=runs/parity3/isaac/fidelity/newton_mjwarp
for P in "default:--contact_tuning default" "hardlimits:--contact_tuning hardlimits" "tau10_impact_hardlimits:--contact_tuning tau10_impact_hardlimits" \
         "isaaclab3:--solver_cfg isaaclab3" "isaaclab3_collide_every_substep:--solver_cfg isaaclab3_collide_every_substep" \
         "isaaclab3_hardlimits:--solver_cfg isaaclab3_hardlimits"; do
  NAME=${P%%:*}; ARGS=${P#*:}
  echo "-- record $NAME ($ARGS) $(date +%T)" >> $LOG
  python -m metalsim.parity.record_g1 --isaac $REF --out runs/il3/fidelity/$NAME --no_render $ARGS >> $LOG 2>&1
  python -m metalsim.parity.compare --isaac $REF --metalsim runs/il3/fidelity/$NAME --out runs/il3/fidelity/report_$NAME --no_render > /dev/null 2>> $LOG
  echo "   compare exit $?" >> $LOG
done
echo "-- transfer $(date +%T)" >> $LOG
python scripts/diagnostics/il3_transfer.py --its 500,1000,1499 --presets default,hardlimits,tau10_impact_hardlimits,isaaclab3,isaaclab3_hardlimits \
   --json runs/il3/transfer_newton.json >> $LOG 2>&1
echo "exit $? $(date)" >> $LOG
