#!/bin/bash
# speculative contacts, task 2: the full PARITY 1.7 protocol for the final rule: PhysX friction gate + friction rows at the normal impedance (MJW_SPEC_FRICTION=live MJW_SPEC_FRICIMP=d0) on the
# frozen prototype worktree 27f1fcd: rec x2 (landing deterministic, torso-impact floor), transfer (4-env protocol), closed loop
# seeds 1 and 2, cap probe; tau 5 ms (ellip10_tau5_specgap10mm) and tau 10 ms impact (ellip10_specgap10mm) bases.
cd /Users/aditya/robosim
export MJW=/Users/aditya/robosim/upstream/mujoco_warp-spec-27f1fcd MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=live MJW_SPEC_FRICIMP=d0
S=scripts/diagnostics/penetration/sweep.sh; PY=.venv/bin/python; A=scripts/diagnostics/contact_research/air_time_rollout.py; C=runs/contact_research/isaac_model_1000_metalsim.pt
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) mjw $(git -C $MJW rev-parse --short HEAD) MJW_SPEC_FRICTION=$MJW_SPEC_FRICTION"
TAG=_fin $S rec ellip10_tau5_specgap10mm ellip10_specgap10mm
TAG=_fin_r2 $S rec ellip10_tau5_specgap10mm
TAG=_fin $S transfer ellip10_tau5_specgap10mm ellip10_specgap10mm
TAG=_fin $S air ellip10_tau5_specgap10mm ellip10_specgap10mm
for p in ellip10_tau5_specgap10mm ellip10_specgap10mm; do
  PYTHONPATH=$MJW $PY $A $C --contact_tuning $p --seed 2 --out runs/penetration/air/isaac1000_${p}_fin_spec_seed2.json 2>&1 | grep -v "^Warp\|^Module\|^   " | tail -3
done
TAG=_fin $S cap ellip10_tau5_specgap10mm

# over-travel probe of the final rule (transfer protocol at 4 and 64 envs, per-substep contact logging, gait)
for n in 4 64; do PYTHONPATH=$MJW $PY scripts/diagnostics/speculative/overtravel_probe.py ellip10_tau5_specgap10mm runs/speculative/probe2/spec5_fin_n$n.npz --envs $n 2>&1 | grep "^it\|wrote"; done
for n in 4 64; do PYTHONPATH=$MJW $PY scripts/diagnostics/speculative/overtravel_probe.py ellip10_specgap10mm runs/speculative/probe2/spec10_fin_n$n.npz --envs $n 2>&1 | grep "^it\|wrote"; done
echo "=== end probe $(date)"
