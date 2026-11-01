#!/bin/bash
# MetalSim evals of the so101-smooth stream, one GPU-queue job, sequential. From the MetalSim root:
#   scripts/gpu_run.sh so101-smooth-evals render 30 -- bash runs/film/so101_lift_smooth/eval/run_metalsim_evals.sh <so101Sim root> <name>...
# names: expert_7v4 expert_12v orig_12v orig_7v4 isaac_scaled_7v4 isaac_unscaled_raw_7v4
set -u
S=runs/film/so101_lift_smooth/eval
SO=$1; shift
echo "[evals] $(date '+%F %T') mujoco_warp $(git -C upstream/mujoco_warp log --oneline -1)"
for n in "$@"; do
  servo=${n##*_}; c=${n%_*}
  case $c in
    expert) pol=expert ;;
    orig) pol=runs/film/so101_lift/policy/so101_lift_ppo.pt ;;
    *) pol=runs/film/so101_lift_smooth/policy/candidates/$c.pt ;;
  esac
  echo "[evals] $(date +%T) $n: $pol $servo"
  .venv/bin/python $S/eval_metalsim.py $pol $servo $S/metalsim/$n $S/cpu/$n/reference.json $SO 2>&1 | grep -v "^Module \|took .* ms" 
done
echo "[evals] $(date +%T) done"
