#!/bin/bash
# Coupled-scene re-timing on the 16-inch M4 Max (2026-10-13), main 52469d93, Warp 95f6f121, Newton 3fd1f052.
# Two protocols, each one exclusive timing-class GPU job per run (gpu_run.sh waits for < 10 % utilisation):
#   L  = the L4's protocol (stage_mpm_timing.sh / time_metal_suite.sh): record_mpm.py, 2 s, no log, median of wall_s
#   S  = speed_suite.sh's protocol: probes/speed_bench.py, 60 frames after 5 warmup
# Scenes: two-way coupling and ANYmal on sand, eager and with the example's whole-step graph capture.
# Modes: metal_fast_math off (the default, parity mode) and relaxed (section 8's speed mode).
# Repetitions interleaved: for rep { for mode { for scene { L; S } } }.
#   run.sh REPS
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd)          # this worktree (metalsim sources)
R=$W/runs/mpm/coupled_retime; PY=/Users/aditya/robosim/.venv-newton152/bin/python
GR=/Users/aditya/robosim/scripts/gpu_run.sh
export PYTHONPATH=$W
REPS=${1:-3}
for rep in $(seq 1 $REPS); do
 for mode in off relaxed; do
  for s in twoway twoway_capture anymal anymal_capture; do
    case $s in
      twoway) ex=mpm_twoway_coupling; cap=() ;;
      twoway_capture) ex=mpm_twoway_coupling; cap=(--set capture=true) ;;
      anymal) ex=mpm_anymal; cap=() ;;
      anymal_capture) ex=mpm_anymal; cap=(--set capture=true) ;;
    esac
    id=L_${s}_${mode}_r$rep
    if [ ! -f $R/$id/meta.json ]; then
      $GR cr_$id timing 15 -- $PY $W/metalsim/parity/isaac_side/record_mpm.py --device metal:0 --chunk-graph --gpu-exit \
          --no-log --metal-fast-math $mode --scene example:$ex --seconds 2 --stride 1000 ${cap[@]+"${cap[@]}"} --out $R/$id \
          > $R/$id.log 2>&1
      echo "$id exit $? $(grep '^done' $R/$id.log | cut -c1-120)"
    fi
    id=S_${s}_${mode}_r$rep
    if [ ! -f $R/$id.json ]; then
      $GR cr_$id timing 15 -- $PY $W/metalsim/physics/mpm/probes/speed_bench.py example:$ex ${cap[@]+"${cap[@]}"} \
          --frames 60 --warmup 5 --metal-fast-math $mode --json $R/$id.json > $R/$id.log 2>&1
      echo "$id exit $? $(tail -1 $R/$id.log | cut -c1-120)"
    fi
  done
 done
done
echo CR_DONE
