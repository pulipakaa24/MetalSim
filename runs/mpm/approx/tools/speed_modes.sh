#!/bin/bash
# speed_modes.sh REP MODE...  -- timing-class ms/frame per fast-math mode (speed_suite.sh's settings: 60 frames after 5)
# Env: KIND (timing), FRAMES (60), TAG (output dir suffix; KIND=check FRAMES=3 TAG=_warm pre-warms the kernel cache)
set -u
REP=$1; shift
W=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
ROOT=/Users/aditya/robosim
PY=$ROOT/.venv-newton152/bin/python
export PYTHONPATH=$W/_forks/warp:$ROOT/upstream/newton-integrated:$W
export WARP_CACHE_PATH=$W/runs/mpm/kcache_cfm
SCENES=${SCENES:-"granular multi_material twoway_capture anymal_capture plate_drag column_a2 column_a8"}
KIND=${KIND:-timing}; FRAMES=${FRAMES:-60}
for s in $SCENES; do
  for MODE in "$@"; do  # modes interleaved per scene
    OUT=$W/runs/mpm/speed_fm/$MODE${TAG:-}; mkdir -p $OUT
    case $s in
      granular) a=(example:mpm_granular) ;;
      multi_material) a=(example:mpm_multi_material) ;;
      twoway_capture) a=(example:mpm_twoway_coupling --set capture=true) ;;
      anymal_capture) a=(example:mpm_anymal --set capture=true) ;;
      plate_drag) a=(plate --set 'mode="drag"' --set travel=0.3) ;;
      column_a8) a=(column --set aspect=8) ;;
      column_a2) a=(column --set aspect=2) ;;
    esac
    j=$OUT/${s}_r$REP.json
    [ -f "$j" ] && { echo "skip $MODE $s"; continue; }
    $ROOT/scripts/gpu_run.sh spdfm_${MODE}_${s} $KIND 10 -- $PY $W/metalsim/physics/mpm/probes/speed_bench.py "${a[@]}" \
        --frames $FRAMES --warmup 5 --metal-fast-math $MODE --json "$j" > "$OUT/${s}_r$REP.log" 2>&1
    echo "$MODE $s exit $? $(tail -1 "$OUT/${s}_r$REP.log" | cut -c1-160)"
  done
done
echo SPEED_DONE
