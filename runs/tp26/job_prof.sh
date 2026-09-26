#!/bin/bash
# tp26 job 1: where the G1 step spends its time under the task's contact preset (recommended) vs the headline setting
# (contact_cfg null): step-profile segments (graph mode) and per-kernel eager attribution (WP_METAL_PROFILE=1), plus the
# register Cholesky cost split (factor / solve / both) at n = 32, 43, 48.
cd /Users/aditya/robosim && source .venv/bin/activate
G='^  \(physics\|whole\|outside\|solver\|mjw.step\|fwd_\|sensor\|sum of\)\|^task\|^variant\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
python -c "import mujoco_warp, warp; print('imports', mujoco_warp.__file__, warp.__file__)"
echo "--- cholesky parts $(date +%H:%M:%S)"; python scripts/diagnostics/metal_cholesky_parts.py 32 33 40 43 48 2>&1 | tee runs/tp26/chol_parts.log | grep "^n="
echo "--- segments R $(date +%H:%M:%S)"; MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20 > runs/tp26/prof_R_seg.log 2>&1; grep "$G" runs/tp26/prof_R_seg.log
echo "--- kernels R $(date +%H:%M:%S)"; WP_METAL_PROFILE=1 MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 3 --kernels > runs/tp26/prof_R_kern.log 2>&1; grep -A 14 "substep: solver.solve" runs/tp26/prof_R_kern.log | head -16; grep -A 6 "^--- physics" runs/tp26/prof_R_kern.log
echo "--- kernels D $(date +%H:%M:%S)"; WP_METAL_PROFILE=1 MJW_TP_VARIANT='{"contact_cfg":null}' python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 3 --kernels > runs/tp26/prof_D_kern.log 2>&1; grep -A 14 "substep: solver.solve" runs/tp26/prof_D_kern.log | head -16; grep -A 6 "^--- physics" runs/tp26/prof_D_kern.log
echo "=== end $(date) power: $(pmset -g batt | head -1)"
