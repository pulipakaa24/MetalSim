#!/bin/bash
# Full-loop cost of the Newton iteration cap (10 = the task's, 20) for the recommended and the elliptic impratio-10
# presets on the G1 task (g1_tp_variants.py; a runtime-registered solver preset "cap20" = iterations 20, nothing else),
# two repeats interleaved.  scripts/gpu_run.sh e2_loop_cap timing 12 -- scripts/diagnostics/competitors/e2_loop_cap.sh
cd "$(dirname "$0")/../../.."
echo "mujoco_warp: $(.venv/bin/python -c 'import mujoco_warp;print(mujoco_warp.__file__)') $(git -C upstream/mujoco_warp log --oneline -1)"
for rep in 1 2; do
  for cfg in recommended tau10_impact_hardlimits_ellip10; do
    for cap in 10 20; do
      echo "== rep $rep $cfg cap$cap $(date +%H:%M:%S)"
      SC=""; [ "$cap" = 20 ] && SC=',"solver_cfg":"cap20"'
      MJW_TP_VARIANT="{\"contact_cfg\":\"$cfg\"$SC,\"label\":\"$cfg cap$cap rep$rep\"}" .venv/bin/python -c "
from metalsim.physics import solver_presets as sp
sp.PRESETS['cap20'] = sp.SolverPreset(iterations=20)
import runpy, sys; sys.argv = ['scripts/diagnostics/g1_tp_variants.py', '4096']; runpy.run_path(sys.argv[0], run_name='__main__')" 2>&1 | grep -v "^Module\|^Warp\|^   Dev\|^   CUDA\|^   Kern\|^     "
    done
  done
done
