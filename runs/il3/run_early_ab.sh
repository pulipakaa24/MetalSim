#!/bin/bash
# Early-transient blow-ups (iterations 10-14 of both rough runs): 20-iteration trainings per single-item variant, seed 0,
# same learner, then per-env classification from a 12-iteration checkpoint.
cd "$(dirname "$0")/../.."
export SKIP_TESTS=1
R=runs/il3/run_train2.sh
$R rough rough_il3 recommended isaaclab3_every_substep_cap20 20 0 early_fixed
$R rough rough_il3 recommended isaaclab3_every_substep_cap20_hardlimits 20 0 early_hardlimits
$R rough rough_il3 recommended isaaclab3_every_substep_cap20_nogap 20 0 early_nogap
$R rough rough_il3 recommended isaaclab3_every_substep_cap20_mjcontact 20 0 early_mjcontact
$R rough rough_il3 recommended isaaclab3_collide_every_substep 20 0 early_cap100
$R rough rough_il3 default none 20 0 early_contactdefault
$R rough rough_il3 recommended none 20 0 early_recommended
EXTRA="--il3_events 0" $R rough rough_il3 recommended isaaclab3_every_substep_cap20 20 0 early_noevents
$R rough rough_isaac recommended isaaclab3_every_substep_cap20 20 0 early_rough232task
$R rough rough_il3 recommended isaaclab3_every_substep_cap20 12 0 early_fixed_it12
source .venv/bin/activate
for P in isaaclab3_every_substep_cap20 isaaclab3_every_substep_cap20_hardlimits contact:default; do
  python runs/il3/blowup_probe.py rough runs/il3/ckpt/early_fixed_it12.pt $P 300 runs/il3/blowup_early_${P/:/_}.json >> runs/il3/early_ab.log 2>&1
done
echo "exit $? $(date)" >> runs/il3/early_ab.log
