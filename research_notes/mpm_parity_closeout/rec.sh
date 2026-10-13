#!/bin/bash
# rec.sh OUT_DIR WARP_TREE SCENE_ID [extra record_mpm.py args]   -- one Metal floor-scene recording through gpu_run.sh
# SCENE_ID in granular multi_material twoway anymal plate_drag column_a2 (record_variant_suite.sh "floors" settings)
set -u
OUT=$1; WARP=$2; S=$3; shift 3
WT=/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c
MS=${METALSIM_TREE:-/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b}
NEWTON=${NEWTON_TREE:-/Users/aditya/robosim/upstream/newton-integrated}
export PYTHONPATH=$WARP:$NEWTON:$MS
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$WT/runs/mpm/closeout/kcache_$(basename "$OUT")}
case $S in
  granular) A=(--scene example:mpm_granular --seconds 1 --stride 6);;
  multi_material) A=(--scene example:mpm_multi_material --seconds 1 --stride 6);;
  twoway) A=(--scene example:mpm_twoway_coupling --seconds 1 --stride 2);;
  anymal) A=(--scene example:mpm_anymal --seconds 1 --stride 2);;
  plate_drag) A=(--scene plate --set 'mode="drag"' --set travel=0.3 --stride 15);;
  column_a2) A=(--scene column --set aspect=2);;
esac
mkdir -p "$OUT"
[ -f "$OUT/F_$S/meta.json" ] && { echo "skip $OUT/F_$S"; exit 0; }
/Users/aditya/robosim/scripts/gpu_run.sh cl_$(basename "$OUT")_$S ${KIND:-check} 30 -- /Users/aditya/robosim/.venv-newton152/bin/python \
  $MS/metalsim/parity/isaac_side/record_mpm.py --device metal:0 --chunk-graph --gpu-exit --metal-fast-math ${FM:-off} \
  --out "$OUT/F_$S" "${A[@]}" "$@" > "$OUT/F_$S.log" 2>&1
echo "$OUT/F_$S exit $? $(tail -1 "$OUT/F_$S.log" | cut -c1-200)"
