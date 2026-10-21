#!/bin/bash
# Bring this Mac's shared MuJoCo Warp checkout (upstream/mujoco_warp, imported editable by .venv) to the pin
# f1b35e2 (fork metalsim, 2026-10-13: equality block ending at njmax). Fast-forward only. Run it through the GPU queue
# so no job imports mid-update:
#   scripts/gpu_run.sh sync_mjw_f1b35e2 timing 2 -- runs/sync/sync_shared_mjw_f1b35e2.sh > runs/sync/sync_mjw_f1b35e2.log 2>&1
set -ex
cd .
R=$(git -C upstream/mujoco_warp remote -v | awk '/pulipakaa24\/mujoco_warp/ {print $1; exit}')
git -C upstream/mujoco_warp fetch -q "$R" metalsim
git -C upstream/mujoco_warp merge -q --ff-only f1b35e2b3ddc80b197b0c665bf6d71523d175daf
git -C upstream/mujoco_warp log --oneline -1
for v in .venv .venv-newtonfork; do
  [ -x $v/bin/python ] || continue
  $v/bin/python -c "import warp as wp; wp.config.quiet=True; wp.init(); import mujoco_warp, os, subprocess
print('$v', mujoco_warp.__file__, subprocess.run(['git','-C',os.path.dirname(mujoco_warp.__file__),'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip(), 'metal', wp.is_metal_available())"
done
grep -n "efcid + 3 > njmax_in\|efcid + 6 > njmax_in" upstream/mujoco_warp/mujoco_warp/_src/constraint.py
