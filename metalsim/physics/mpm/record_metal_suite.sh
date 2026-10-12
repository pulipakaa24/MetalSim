#!/bin/bash
# Metal counterparts of metalsim/parity/isaac_side/isaaclab3/stage_mpm.sh (same run ids, same scene settings), recorded
# with the oracle semantics (float32 accumulation, CUDA stopping rule) + the Newton branch + the chunk graph, all of
# which are bitwise equal to the oracle (tests/test_mpm.py). Output runs/mpm/metal/<id>/. Each run takes the GPU queue
# separately. STAGES (default "r0 r1 r3 r4 r6 r7") selects groups; existing runs are skipped.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.   # Warp and Newton: the forks installed in .venv-newton152 by scripts/setup_newton152.sh
PY=.venv-newton152/bin/python
R=runs/mpm/metal; mkdir -p $R
STAGES=${STAGES:-"r0 r1 r3 r4 r6 r7"}
rec() { local id=$1; shift; [ -f $R/$id/meta.json ] && { echo "skip $id"; return; }
        scripts/gpu_run.sh mpm_$id render 30 -- $PY metalsim/parity/isaac_side/record_mpm.py --device metal:0 \
            --chunk-graph --gpu-exit --out $R/$id "$@" > $R/$id.log 2>&1; echo "$id exit $?"; tail -1 $R/$id.log; }

for S in $STAGES; do case $S in
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
echo METAL_SUITE_DONE
