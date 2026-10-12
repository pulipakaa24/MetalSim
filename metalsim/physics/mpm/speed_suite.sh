#!/bin/bash
# Timing-class ms/frame of the MPM speed scenes for one (Warp tree, Newton tree, metalsim tree) combination, one
# timing-class GPU job per scene (scripts/gpu_run.sh waits for an idle device). Every path is absolute because
# gpu_run.sh runs the job from the main checkout.
#   speed_suite.sh OUT_DIR WARP_TREE NEWTON_TREE METALSIM_TREE [REP]
# Env: SCENES (default below), FRAMES (60), WARMUP (5), WARP_CACHE_PATH (default OUT_DIR/../kcache_<warp tree name>,
# so a modified Warp tree never shares generated Metal sources with another tree).
set -u
OUT=$1; WARP=$2; NEWTON=$3; MAIN=$4; REP=${5:-1}
ROOT=${METALSIM_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}   # the checkout whose scripts/gpu_run.sh and .venv-newton152 are used
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$ROOT/.venv-newton152/bin/python
# '-' for WARP_TREE / NEWTON_TREE: the Warp / Newton installed in .venv-newton152 (scripts/setup_newton152.sh)
PP=$MAIN; [ "$NEWTON" != "-" ] && PP=$NEWTON:$PP; [ "$WARP" != "-" ] && PP=$WARP:$PP
export PYTHONPATH=$PP
KC=installed; [ "$WARP" != "-" ] && KC=$(basename "$WARP")
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$(dirname "$OUT")/kcache_$KC}
FRAMES=${FRAMES:-60}; WARMUP=${WARMUP:-5}
SCENES=${SCENES:-"granular multi_material twoway twoway_capture anymal anymal_capture plate_drag column_a8 column_a2"}
mkdir -p "$OUT"
for s in $SCENES; do
  case $s in
    granular) a=(example:mpm_granular) ;;
    multi_material) a=(example:mpm_multi_material) ;;
    twoway) a=(example:mpm_twoway_coupling) ;;
    twoway_capture) a=(example:mpm_twoway_coupling --set capture=true) ;;
    anymal) a=(example:mpm_anymal) ;;
    anymal_capture) a=(example:mpm_anymal --set capture=true) ;;
    plate_drag) a=(plate --set 'mode="drag"' --set travel=0.3) ;;
    column_a8) a=(column --set aspect=8) ;;
    column_a2) a=(column --set aspect=2) ;;
    *) echo "unknown scene $s"; continue ;;
  esac
  j=$OUT/${s}_r$REP.json
  [ -f "$j" ] && { echo "skip $s"; continue; }
  $ROOT/scripts/gpu_run.sh spd_${s} timing 15 -- $PY $HERE/probes/speed_bench.py "${a[@]}" \
      --frames $FRAMES --warmup $WARMUP --json "$j" > "$OUT/${s}_r$REP.log" 2>&1
  echo "$s exit $? $(tail -1 "$OUT/${s}_r$REP.log" | cut -c1-200)"
done
