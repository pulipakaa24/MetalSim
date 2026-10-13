#!/bin/bash
# Metal timing counterparts of metalsim/parity/isaac_side/isaaclab3/stage_mpm_timing.sh: the same scenes and lengths,
# CUDA stopping rule (newton_compat loop="cuda"), Newton branch + chunk graph, no per-solve logging, one timing-class
# GPU job per run (waits for an idle device). Output runs/mpm/metal_timing/<id>/.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.   # Warp and Newton: the forks installed in .venv-newton152 by scripts/setup_newton152.sh
PY=.venv-newton152/bin/python
R=runs/mpm/metal_timing; mkdir -p $R
rec() { local id=$1; shift; [ -f $R/$id/meta.json ] && { echo "skip $id"; return; }
        scripts/gpu_run.sh mpmt_$id timing 20 -- $PY metalsim/parity/isaac_side/record_mpm.py --device metal:0 \
            --chunk-graph --gpu-exit --no-log --out $R/$id "$@" > $R/$id.log 2>&1; echo "$id exit $?"; grep "gpu_run\|^done" $R/$id.log; }
for ex in granular multi_material twoway_coupling anymal; do
  rec T_${ex}_metal --scene example:mpm_$ex --seconds 2 --stride 1000
  # the example's own whole-step capture (the L4's "captured" column); the solver loop is GPU-gated inside it
  rec T_${ex}_metal_capture --scene example:mpm_$ex --seconds 2 --stride 1000 --set capture=true
done
rec T_column_a2_metal --scene column --set aspect=2 --stride 1000
rec T_column_a8_metal --scene column --set aspect=8 --stride 1000
rec T_plate_drag_metal --scene plate --set 'mode="drag"' --set travel=0.3 --stride 1000
echo METAL_TIMING_DONE
