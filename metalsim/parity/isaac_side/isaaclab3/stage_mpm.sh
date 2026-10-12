#!/bin/bash
# MPM references on the L4: Newton 1.5.2 SolverImplicitMPM as shipped in Isaac Lab 3.0-EA (docs/research/mpm_2026-10-09.md).
# Needs ~/parity3_scripts/mpm/{record_mpm.py,record_mpm_il3.py,scenes.py,newton_compat.py}. Output ~/parity3/mpm/<id>/, tarball ~/parity3/mpm_refs.tgz.
#   R0  noise floors on the a=2 column: host-run graph loop twice; Newton's own conditional-graph loop; float32 accumulation
#   R1  column collapse a in {0.5,1,2,3,5,8}: Isaac Lab cell_center lattice (trajectory tier) + 10 jittered seeds each (statistical tier)
#   R3  material coverage: Newton's granular / multi_material / snow_ball / viscous / beam_twist examples
#   R2  the a=2 and a=5 columns through Isaac Lab's MPMObject/MPMSolverCfg (vs the stock-loop Newton-direct runs)
#   R4  two-way rigid coupling example;  R6  ANYmal on sand example
#   R7  resistive-force-theory plate: vertical intrusion and horizontal drag through a granular bed (force on the plate)
# STAGES (default "r0 r1 r2 r3 r4 r6 r7") selects groups.
source $HOME/parity3_scripts/common.sh
exec >> $OUT/stage_mpm.log 2>&1
set -x
STAGES=${STAGES:-"r0 r1 r2 r3 r4 r6 r7"}
M=$HOME/parity3_scripts/mpm; R=$OUT/mpm; mkdir -p $R
rm -f $OUT/MPM_DONE
nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import warp_nn" 2>/dev/null || uv pip install "warp-nn[onnx]==0.3.0"
python -c "import newton, warp; print('newton', newton.__version__, 'warp', warp.__version__)"
rec() { local id=$1; shift; wait_for_gpu; [ -f $R/$id/meta.json ] && { echo "skip $id"; return; }
        python $M/record_mpm.py --device cuda:0 --out $R/$id "$@" > $R/$id.log 2>&1; echo "$id exit $?"; tail -1 $R/$id.log; }

for S in $STAGES; do case $S in
r0)
  rec R0_a2_host_1 --scene column --set aspect=2
  rec R0_a2_host_2 --scene column --set aspect=2
  rec R0_a2_stock  --scene column --set aspect=2 --loop native
  rec R0_a2_f32    --scene column --set aspect=2 --accumulate float32
  rec R0_granular_host_1 --scene example:mpm_granular --seconds 1 --stride 6
  rec R0_granular_host_2 --scene example:mpm_granular --seconds 1 --stride 6
  rec R0_granular_f32    --scene example:mpm_granular --seconds 1 --stride 6 --accumulate float32
  ;;
r1)
  for a in 0.5 1 2 3 5 8; do
    rec R1_a${a} --scene column --set aspect=$a
  done
  for a in 0.5 1 2 3 5 8; do for s in 0 1 2 3 4 5 6 7 8 9; do
    rec R1_a${a}_j${s} --scene column --set aspect=$a --set jitter=0.5 --set seed=$s --stride 10
  done; done
  ;;
r2)
  for a in 2 5; do id=R2_il3_a$a; wait_for_gpu
    [ -f $R/$id/meta.json ] || { python $M/record_mpm_il3.py --viz none --aspect $a --out $R/$id > $R/$id.log 2>&1; echo "$id exit $?"; tail -1 $R/$id.log; }
  done
  rec R0_a5_stock --scene column --set aspect=5 --loop native
  ;;
r3)
  rec R3_granular       --scene example:mpm_granular       --seconds 3 --stride 6
  rec R3_multi_material --scene example:mpm_multi_material --seconds 3 --stride 6
  rec R3_snow_ball      --scene example:mpm_snow_ball      --seconds 2 --stride 6
  rec R3_viscous        --scene example:mpm_viscous        --seconds 2 --stride 12
  rec R3_beam_twist     --scene example:mpm_beam_twist     --seconds 2 --stride 12
  ;;
r4)
  rec R4_twoway --scene example:mpm_twoway_coupling --seconds 3 --stride 2
  ;;
r7)
  rec R7_plate_intrude --scene plate --set 'mode="intrude"' --stride 15
  rec R7_plate_drag    --scene plate --set 'mode="drag"' --set travel=0.3 --stride 15
  ;;
r6)
  rec R6_anymal --scene example:mpm_anymal --seconds 3 --stride 2
  ;;
esac; done
tar czf $OUT/mpm_refs.tgz -C $OUT mpm
touch $OUT/MPM_DONE
