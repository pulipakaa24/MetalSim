#!/bin/bash
# CUDA's own floors for the MPM parity suite (reports/CUDA and Metal float parity.md): every mesh-collider scene with
# float32 accumulation (as on Metal) and the CUDA stopping rule, as a baseline and under perturbations that leave the
# algorithm unchanged:
#   F_arith  precise   (no fast_math: exact / and sqrt)
#   F_tie    nofma     (fuse_fp off), cubin (toolkit ptxas instead of the driver's), sah (SAH mesh BVH instead of LBVH)
# Output ~/parity3/mpm_floors/<scene>_<variant>/, tarball ~/parity3/mpm_floors.tgz.
source $HOME/parity3_scripts/common.sh
exec >> $OUT/stage_mpm_floors.log 2>&1
set -x
M=$HOME/parity3_scripts/mpm; R=$OUT/mpm_floors; mkdir -p $R
rm -f $OUT/MPM_FLOORS_DONE
rec() { local id=$1; shift; wait_for_gpu; [ -f $R/$id/meta.json ] && { echo "skip $id"; return; }
        python $M/record_mpm.py --device cuda:0 --accumulate float32 --out $R/$id "$@" > $R/$id.log 2>&1; echo "$id exit $?"; tail -1 $R/$id.log; }
declare -A SC=(
  [granular]="--scene example:mpm_granular --seconds 1 --stride 6"
  [multi_material]="--scene example:mpm_multi_material --seconds 1 --stride 6"
  [twoway]="--scene example:mpm_twoway_coupling --seconds 1 --stride 2"
  [anymal]="--scene example:mpm_anymal --seconds 1 --stride 2"
  [plate_drag]="--scene plate --set mode=\"drag\" --set travel=0.3 --stride 15"
  [column_a2]="--scene column --set aspect=2"
)
for s in granular multi_material twoway anymal plate_drag column_a2; do
  eval "args=(${SC[$s]})"
  rec ${s}_base "${args[@]}"
  rec ${s}_precise "${args[@]}" --precise-cuda
  rec ${s}_nofma "${args[@]}" --cuda-nofma
  rec ${s}_cubin "${args[@]}" --cuda-cubin
  rec ${s}_sah "${args[@]}" --cuda-sah
done
tar czf $OUT/mpm_floors.tgz -C $OUT mpm_floors
touch $OUT/MPM_FLOORS_DONE
