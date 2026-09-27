#!/bin/bash
# sweep.sh STAGE [PRESET...]: the G1 fidelity protocol (PARITY §1.7) for the 2026-09-26 penetration sweep
# (docs/research/penetration_2026-09-26.md). MuJoCo Warp from the shared fork checkout (upstream/mujoco_warp, f824af1).
#   rec      : A_hold / B_random / C_drop replay vs Isaac 5.1's parity_out2/rt (record_g1 with per-substep per-body
#              penetration, forces, Newton iterations) + compare.py                                   (render class)
#              DT=0.00125 records at 1.25 ms (16 substeps) into <preset>_dt1p25; TAG=_r2 writes a repeat run
#   transfer : Isaac 5.1 PhysX checkpoints 500/1000/1499 played on the task (x travelled vs 3.19/3.02/3.11 m) (render)
#   air      : feet_air_time / feet_slide of Isaac's checkpoint 1000                                    (render)
#   cost     : physics-only step time at 4096 envs, each preset at its own Newton cap                   (timing)
#   cap      : Newton cap probe (4096 envs, random actions and the elliptic-trained policy)             (render)
set -u
STAGE=$1; shift
ROOT=/Users/aditya/robosim; cd $ROOT
PY=$ROOT/.venv/bin/python; OUT=$ROOT/runs/penetration
# MJW=<fork worktree> runs another MuJoCo Warp (e.g. the speculative-contact prototype upstream/mujoco_warp-spec with
# MJW_SPECULATIVE_GAP=1); outputs then go to <name>_spec when MJW_SPECULATIVE_GAP=1
MJW=${MJW:-$ROOT/upstream/mujoco_warp}; export PYTHONPATH=$MJW${PYTHONPATH:+:$PYTHONPATH}
DT=${DT:-0.0025}; SUF=""; [ "$DT" != 0.0025 ] && SUF="_dt$DT"
SUF="$SUF${TAG:-}"; [ "${MJW_SPECULATIVE_GAP:-0}" = 1 ] && SUF="${SUF}_spec"
echo "== stage $STAGE dt=$DT mjw=$MJW@$(git -C $MJW rev-parse --short HEAD) spec=${MJW_SPECULATIVE_GAP:-0} presets=$* $(date)"; python3 scripts/gpu_lock.py status | head -1
F='^Warning\|^Module\|^Warp\|^   '
case $STAGE in
  rec)
    for p in "$@"; do
      d=$OUT/rec/$p$SUF
      $PY -m metalsim.parity.record_g1 --isaac runs/parity/isaac/parity_out2/rt --out $d --contact_tuning $p --physics_dt $DT --no_render 2>&1 | grep -v "$F"
      $PY -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/rt --metalsim $d --out $d/report --no_render 2>&1 | grep -v "$F" | tail -1
    done ;;
  transfer)
    S=$(for p in "$@"; do printf "mjwarp:%s," $p; done); S=${S%,}
    $PY scripts/diagnostics/newton_transfer.py --its 500,1000,1499 --settings $S 2>&1 | grep -v "$F" ;;
  air)
    for p in "$@"; do
      $PY scripts/diagnostics/contact_research/air_time_rollout.py runs/contact_research/isaac_model_1000_metalsim.pt --contact_tuning $p --out $OUT/air/isaac1000_$p$SUF.json 2>&1 | grep -v "$F" | tail -25
    done ;;
  cost)
    $PY scripts/diagnostics/bench_contact_tuning.py "$@" 2>&1 | grep -v "$F" ;;
  cap)
    for p in "$@"; do
      $PY scripts/diagnostics/competitors/cap_probe_ellip.py flat $p random $OUT/cap/${p}${SUF}_random.json 2>&1 | grep -v "$F" | tail -8
      $PY scripts/diagnostics/competitors/cap_probe_ellip.py flat $p runs/il3/ckpt/g1_flat_flatcfg_ellip10.pt $OUT/cap/${p}${SUF}_policy.json 2>&1 | grep -v "$F" | tail -8
    done ;;
esac
echo "== stage $STAGE done $(date)"
