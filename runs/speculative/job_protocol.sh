#!/bin/bash
# speculative contacts, task 2: the full PARITY 1.7 protocol for the PhysX-rule friction gate (MJW_SPEC_FRICTION=live) on the
# frozen prototype worktree d99186b: rec x2 (landing deterministic, torso-impact floor), transfer (4-env protocol), closed loop
# seeds 1 and 2, cap probe; tau 5 ms (ellip10_tau5_specgap10mm) and tau 10 ms impact (ellip10_specgap10mm) bases.
cd /Users/aditya/robosim
export MJW=/Users/aditya/robosim/upstream/mujoco_warp-spec-d99186b MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=live
S=scripts/diagnostics/penetration/sweep.sh; PY=.venv/bin/python; A=scripts/diagnostics/contact_research/air_time_rollout.py; C=runs/contact_research/isaac_model_1000_metalsim.pt
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) mjw $(git -C $MJW rev-parse --short HEAD) MJW_SPEC_FRICTION=$MJW_SPEC_FRICTION"
TAG=_live $S rec ellip10_tau5_specgap10mm ellip10_specgap10mm
TAG=_live_r2 $S rec ellip10_tau5_specgap10mm
TAG=_live $S transfer ellip10_tau5_specgap10mm ellip10_specgap10mm
TAG=_live $S air ellip10_tau5_specgap10mm ellip10_specgap10mm
for p in ellip10_tau5_specgap10mm ellip10_specgap10mm; do
  PYTHONPATH=$MJW $PY $A $C --contact_tuning $p --seed 2 --out runs/penetration/air/isaac1000_${p}_live_spec_seed2.json 2>&1 | grep -v "^Warp\|^Module\|^   " | tail -3
done
TAG=_live $S cap ellip10_tau5_specgap10mm
echo "=== end $(date)"
