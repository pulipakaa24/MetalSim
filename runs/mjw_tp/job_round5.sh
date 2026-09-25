#!/bin/bash
# round 5: step 3 = MuJoCo Warp fork: one-world-per-thread sparse L'DL factor/solve on Metal + skip no-op launches
# (gravcomp with no gravcomp bodies, tendon damping with no tendons). A/B within the fork via has_gravcomp.
cd /Users/aditya/robosim && source .venv/bin/activate
F='^\[\|Error\|Traceback'
S2='{"metal_register_cholesky_max":48,"m_dense_max":0'
{
echo "=== start $(date) power: $(pmset -g batt | head -1)"; python3 scripts/gpu_lock.py status
for V in "$S2,\"label\":\"step3 (fork: serial LDL + no-op skips)\"}" "$S2,\"gravcomp_launch\":true,\"label\":\"step3 with the gravcomp launch restored\"}" \
         "$S2,\"block_dim\":{\"sparse_ldl_serial\":64},\"label\":\"step3 ldl block 64\"}" "$S2,\"block_dim\":{\"sparse_ldl_serial\":16},\"label\":\"step3 ldl block 16\"}" \
         '{"label":"base (fork as now)"}'; do
  MJW_TP_VARIANT="$V" python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep "$F"
done
echo "=== end $(date)"
} > runs/mjw_tp/costsplit5.log 2>&1
{
echo "=== start $(date)"; python3 scripts/gpu_lock.py status
MJW_TP_VARIANT="$S2}" python scripts/diagnostics/g1_tp_check.py 512 50 --ref runs/mjw_tp/traj_base3.npz 2>&1 | grep -v "^Module\|^Warp\|^   [\"CDK/]\|^     [\"/]"
MJW_TP_VARIANT="$S2}" WP_METAL_PROFILE=1 python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --kernels 2>&1 | sed -n '/top kernels/,$p'
echo "=== end $(date)"
} > runs/mjw_tp/checks5.log 2>&1
