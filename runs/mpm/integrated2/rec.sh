#!/bin/bash
# rec.sh LABEL WARP NEWTON : the six floor recordings (record_variant_suite.sh "floors" settings), one GPU job each
A=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/runs/a2
R=/Users/aditya/robosim
L=$1; WT=$2; NT=$3
O=$A/rec/$L; mkdir -p $O
export PYTHONPATH=$WT:$NT:$R WARP_CACHE_PATH=$A/kc/rec_$L PYTHONDONTWRITEBYTECODE=1
PY=$R/.venv-newton152/bin/python
rec() { local id=$1; shift; [ -f "$O/$id/meta.json" ] && { echo "skip $id"; return; }
  $R/scripts/gpu_run.sh a2r_${L}_$id check 30 -- $PY $R/metalsim/parity/isaac_side/record_mpm.py --device metal:0 \
    --chunk-graph --gpu-exit --out "$O/$id" "$@" > "$O/$id.log" 2>&1; echo "$L $id exit $? $(tail -1 "$O/$id.log" | cut -c1-100)"; }
rec F_granular --scene example:mpm_granular --seconds 1 --stride 6
rec F_multi_material --scene example:mpm_multi_material --seconds 1 --stride 6
rec F_twoway --scene example:mpm_twoway_coupling --seconds 1 --stride 2
rec F_anymal --scene example:mpm_anymal --seconds 1 --stride 2
rec F_plate_drag --scene plate --set 'mode="drag"' --set travel=0.3 --stride 15
rec F_column_a2 --scene column --set aspect=2
