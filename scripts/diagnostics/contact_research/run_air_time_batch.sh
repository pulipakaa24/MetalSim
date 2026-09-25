#!/bin/bash
# B(1): Isaac's and our checkpoints on the MuJoCo Warp G1 flat task, gait statistics per contact preset.
# usage: scripts/gpu_run.sh air_time_batchN timing 15 -- bash scripts/diagnostics/contact_research/run_air_time_batch.sh N
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python; S=scripts/diagnostics/contact_research/air_time_rollout.py; O=runs/contact_research/air_time
mkdir -p $O
echo "== batch $1 $(date)"; python3 scripts/gpu_lock.py status
I=runs/contact_research/isaac_model_1000_metalsim.pt; I5=runs/contact_research/isaac_model_500_metalsim.pt
U=runs/policies/g1_flat_rslrl/g1_flat_rslrl_it1000.pt; U4=runs/policies/g1_flat_rslrl/g1_flat_rslrl_it400.pt
run() { name=$1; shift; t0=$(date +%s); $PY $S "$@" --out $O/$name.json > $O/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s)-t0 )) s"; }
if [ "$1" = 1 ]; then
  run smoke $I --envs 16 --steps 60
  run isaac1000_default $I
  run isaac1000_tau10_impact $I --contact_tuning tau10_impact_hardlimits
  run isaac1000_tau5 $I --contact_tuning tau5_imp99_hardlimits
  run isaac1000_default_stoch $I --stochastic
  run isaac1000_default_seed2 $I --seed 2
else
  run ours1000_default $U
  run ours1000_tau10_impact $U --contact_tuning tau10_impact_hardlimits
  run ours400_default $U4
  run ours400_tau10_impact $U4 --contact_tuning tau10_impact_hardlimits
  run isaac500_default $I5
  run isaac1000_tau10_impact_stoch $I --contact_tuning tau10_impact_hardlimits --stochastic
fi
