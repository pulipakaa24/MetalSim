#!/bin/bash
# record_metal_suite.sh (and the Metal side of stage_mpm_floors.sh) for an explicit (Warp, Newton, metalsim) tree
# combination, with absolute paths (scripts/gpu_run.sh runs jobs from the main checkout) and a per-Warp-tree kernel
# cache. Same run ids and settings as the L4 stages, so parity_report.py / floors_report.py read the output directly.
#   record_variant_suite.sh OUT_DIR WARP_TREE NEWTON_TREE METALSIM_TREE
# Env: STAGES (default "floors r0 r1 r3 r4 r6 r7"); EXTRA (more record_mpm.py arguments, e.g. --no-log);
# existing runs are skipped.
set -u
OUT=$1; WARP=$2; NEWTON=$3; MAIN=$4
ROOT=${METALSIM_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}   # the checkout whose scripts/gpu_run.sh and .venv-newton152 are used
PY=$ROOT/.venv-newton152/bin/python
# '-' for WARP_TREE / NEWTON_TREE: the Warp / Newton installed in .venv-newton152 (scripts/setup_newton152.sh)
PP=$MAIN; [ "$NEWTON" != "-" ] && PP=$NEWTON:$PP; [ "$WARP" != "-" ] && PP=$WARP:$PP
export PYTHONPATH=$PP
KC=installed; [ "$WARP" != "-" ] && KC=$(basename "$WARP")
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$(dirname "$OUT")/kcache_$KC}
REC=$MAIN/metalsim/parity/isaac_side/record_mpm.py
STAGES=${STAGES:-"floors r0 r1 r3 r4 r6 r7"}
mkdir -p "$OUT"
rec() { local id=$1; shift; [ -f "$OUT/$id/meta.json" ] && { echo "skip $id"; return; }
        $ROOT/scripts/gpu_run.sh mpmv_$id ${KIND:-render} 30 -- $PY $REC --device metal:0 --chunk-graph --gpu-exit \
            --out "$OUT/$id" "$@" ${EXTRA:-} > "$OUT/$id.log" 2>&1; echo "$id exit $? $(tail -1 "$OUT/$id.log" | cut -c1-160)"; }
for S in $STAGES; do case $S in
floors)  # stage_mpm_floors.sh scenes (the L4 baseline is <scene>_base)
  rec F_granular --scene example:mpm_granular --seconds 1 --stride 6
  rec F_multi_material --scene example:mpm_multi_material --seconds 1 --stride 6
  rec F_twoway --scene example:mpm_twoway_coupling --seconds 1 --stride 2
  rec F_anymal --scene example:mpm_anymal --seconds 1 --stride 2
  rec F_plate_drag --scene plate --set 'mode="drag"' --set travel=0.3 --stride 15
  rec F_column_a2 --scene column --set aspect=2
  ;;
r0)
  rec R0_a2_host_1 --scene column --set aspect=2
  rec R0_a2_host_2 --scene column --set aspect=2
  rec R0_granular_host_1 --scene example:mpm_granular --seconds 1 --stride 6
  rec R0_granular_host_2 --scene example:mpm_granular --seconds 1 --stride 6
  ;;
r1)
  for a in 0.5 1 2 3 5 8; do rec R1_a${a} --scene column --set aspect=$a; done
  for a in 0.5 1 2 3 5 8; do for s in 0 1 2 3 4 5 6 7 8 9; do
    rec R1_a${a}_j${s} --scene column --set aspect=$a --set jitter=0.5 --set seed=$s --stride 10
  done; done
  ;;
r3)
  rec R3_granular       --scene example:mpm_granular       --seconds 3 --stride 6
  rec R3_multi_material --scene example:mpm_multi_material --seconds 3 --stride 6
  rec R3_snow_ball      --scene example:mpm_snow_ball      --seconds 2 --stride 6
  rec R3_viscous        --scene example:mpm_viscous        --seconds 2 --stride 12
  rec R3_beam_twist     --scene example:mpm_beam_twist     --seconds 2 --stride 12
  ;;
r4) rec R4_twoway --scene example:mpm_twoway_coupling --seconds 3 --stride 2 ;;
r6) rec R6_anymal --scene example:mpm_anymal --seconds 3 --stride 2 ;;
r7)
  rec R7_plate_intrude --scene plate --set 'mode="intrude"' --stride 15
  rec R7_plate_drag    --scene plate --set 'mode="drag"' --set travel=0.3 --stride 15
  ;;
esac; done
echo SUITE_DONE
