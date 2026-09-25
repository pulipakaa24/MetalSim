#!/bin/bash
# round 3: four-column cost split per step (base, +register Cholesky, +sparse L'DL of M, +sparse Jacobian),
# profile of the combined configuration, early-step checks, 20-iteration training probes
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
S1='{"metal_register_cholesky_max":48,"label":"step1 reg-chol"}'
S2='{"metal_register_cholesky_max":48,"m_dense_max":0,"label":"step2 +sparse LDL of M"}'
S3='{"metal_register_cholesky_max":48,"m_dense_max":0,"jacobian":"sparse","label":"step3 +sparse Jacobian"}'
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
for V in '{"label":"base"}' "$S1" "$S2" "$S3" '{"label":"base repeat"}'; do
  MJW_TP_VARIANT="$V" python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
done
echo "=== end $(date)"
} > runs/mjw_tp/costsplit3.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
MJW_TP_VARIANT="$S3" python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]"
MJW_TP_VARIANT="$S3" WP_METAL_PROFILE=1 python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --kernels 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]"
echo "=== end $(date)"
} > runs/mjw_tp/profile3.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
C=scripts/diagnostics/g1_tp_check.py
MJW_TP_VARIANT='{}' python $C 512 50 --out runs/mjw_tp/traj_base3.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]"
for V in "$S1" "$S2" "$S3"; do
  MJW_TP_VARIANT="$V" python $C 512 50 --ref runs/mjw_tp/traj_base3.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]"
done
echo "=== end $(date)"
} > runs/mjw_tp/checks3.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
for V in '{}' "$S1" "$S2" "$S3"; do
  MJW_TP_VARIANT="$V" python scripts/diagnostics/g1_tp_train_probe.py 4096 20 2>&1 | grep "^variant\|^it \|Error\|Traceback\|anomal"
done
echo "=== end $(date)"
} > runs/mjw_tp/probe3.log 2>&1
