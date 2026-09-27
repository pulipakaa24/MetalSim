#!/bin/bash
# second-seed closed-loop rollouts (noise floor of the air-time / slide / short-phase numbers)
cd /Users/aditya/robosim; PY=.venv/bin/python; A=scripts/diagnostics/contact_research/air_time_rollout.py; C=runs/contact_research/isaac_model_1000_metalsim.pt
for p in recommended ellip10_feet_tau5 ellip10_feet_tau5_dr2; do $PY $A $C --contact_tuning $p --seed 2 --out runs/penetration/air/isaac1000_${p}_seed2.json > /dev/null 2>&1; done
MJW_SPECULATIVE_GAP=1 PYTHONPATH=upstream/mujoco_warp-spec $PY $A $C --contact_tuning ellip10_tau5_specgap10mm --seed 2 --out runs/penetration/air/isaac1000_ellip10_tau5_specgap10mm_spec_seed2.json > /dev/null 2>&1
echo done
