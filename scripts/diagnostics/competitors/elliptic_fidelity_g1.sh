#!/bin/bash
# elliptic_fidelity_g1.sh MJW_PATH STAGE: the G1 fidelity protocol (PARITY §1.7 / DECISIONS 2026-09-25) for the elliptic
# variants of the adopted contact preset (registered by elliptic_presets.py), MuJoCo Warp from MJW_PATH.
#   rec      : A_hold / B_random / C_drop vs Isaac's parity_out2/rt recording + compare + momentum impulses  (render class)
#   transfer : Isaac 5.1 PhysX checkpoints 500/1000/1499 played on the task (x travelled vs 3.19/3.02/3.11 m)  (render)
#   air      : feet_air_time / feet_slide of Isaac's checkpoint 1000 per preset                               (render)
#   cost     : physics-only step time at 4096 envs per preset (bench_contact_tuning.py)                        (timing)
set -u
MJW=$1; STAGE=$2; shift 2
ROOT=/Users/aditya/robosim; cd $ROOT
export PYTHONPATH=$MJW${PYTHONPATH:+:$PYTHONPATH}
PY=$ROOT/.venv/bin/python; W=$ROOT/scripts/diagnostics/competitors/elliptic_presets.py
PRESETS=${PRESETS:-"tau10_impact_elliptic_hardlimits tau10_impact_elliptic_imp10_hardlimits tau10_impact_elliptic_imp100_hardlimits"}
echo "== stage $STAGE mjw=$MJW head=$(git -C $MJW rev-parse --short HEAD) presets=$PRESETS $(date)"; python3 scripts/gpu_lock.py status | head -1
case $STAGE in
  rec)
    for p in $PRESETS; do
      $PY $W -m metalsim.parity.record_g1 --isaac runs/parity/isaac/parity_out2/rt --out runs/parity/tuning/$p --contact_tuning $p --no_render 2>&1 | grep -v "^Warning\|^Module\|^Warp\|^   "
      $PY $W -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/rt --metalsim runs/parity/tuning/$p --out runs/parity/tuning/report_$p --no_render 2>&1 | grep -v "^Warning\|^Module" | tail -3
      $PY scripts/diagnostics/contact_research/momentum_impulse.py runs/parity/tuning/$p 2>&1 | grep -v "^Warning\|^Module" | tail -12
    done ;;
  transfer)
    S=$(for p in $PRESETS; do printf "mjwarp:%s," $p; done); S=${S%,}
    $PY $W scripts/diagnostics/newton_transfer.py --its 500,1000,1499 --settings mjwarp:tau10_impact_hardlimits,$S 2>&1 | grep -v "^Warning\|^Module\|^Warp\|^   " ;;
  air)
    for p in $PRESETS; do
      $PY $W scripts/diagnostics/contact_research/air_time_rollout.py runs/contact_research/isaac_model_1000_metalsim.pt --contact_tuning $p --out runs/contact_research/air_time/isaac1000_$p.json 2>&1 | grep -v "^Warning\|^Module\|^Warp\|^   " | tail -25
    done ;;
  cost)
    $PY $W scripts/diagnostics/bench_contact_tuning.py $PRESETS 2>&1 | grep -v "^Warning\|^Module\|^Warp\|^   " ;;
esac
echo "== stage $STAGE done $(date)"
