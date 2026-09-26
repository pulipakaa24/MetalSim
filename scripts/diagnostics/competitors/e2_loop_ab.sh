#!/bin/bash
# Full-loop cost of elliptic impratio 10 vs the recommended preset on the G1 task (g1_tp_variants.py, mode 2 live in
# the shared checkout), two repeats interleaved.  scripts/gpu_run.sh e2_loop_ab timing 14 -- scripts/diagnostics/competitors/e2_loop_ab.sh
cd "$(dirname "$0")/../../.."
echo "mujoco_warp: $(.venv/bin/python -c 'import mujoco_warp;print(mujoco_warp.__file__)') $(git -C upstream/mujoco_warp log --oneline -1)"
for rep in 1 2; do
  for cfg in recommended tau10_impact_hardlimits_ellip10; do
    echo "== rep $rep $cfg $(date +%H:%M:%S)"
    MJW_TP_VARIANT="{\"contact_cfg\":\"$cfg\",\"label\":\"$cfg rep$rep\"}" .venv/bin/python scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep -v "^Module\|^Warp\|^   Dev\|^   CUDA\|^   Kern\|^     "
  done
done
