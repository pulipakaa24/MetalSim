#!/bin/bash
# Floor *distribution* for the MPM parity suite: stage_mpm_floors.sh's settings (float32 accumulation, CUDA stopping
# rule, same scenes.py / newton_compat.py copies) with more legitimate CUDA perturbations per scene, algorithm unchanged:
#   base2            unperturbed rerun (determinism check against mpm_floors/<scene>_base)
#   j<s>             every initial particle coordinate moved by k ulp, k in {-1,0,1}, seed s (record_mpm.py --jitter-ulp)
#   <compiler>_j<s>  jitter combined with a compiler / BVH variant (nofma, cubin, sah, precise)
#   nofma_cubin, precise_sah   compiler-variant combinations
# Output ~/parity3/mpm_floors2/<scene>_<variant>/, tarball ~/parity3/mpm_floors2.tgz.
source $HOME/parity3_scripts/common.sh
exec >> $OUT/stage_mpm_floors2.log 2>&1
set -x
M=$HOME/parity3_scripts/mpm/floors2; R=$OUT/mpm_floors2; mkdir -p $R
rm -f $R/DONE
rec() { local id=$1; shift; wait_for_gpu; [ -f $R/$id/meta.json ] && { echo "skip $id"; return; }
        timeout 1800 python $M/record_mpm.py --device cuda:0 --accumulate float32 --out $R/$id "$@" > $R/$id.log 2>&1; echo "$id exit $?"; tail -1 $R/$id.log; }
declare -A SC=(
  [granular]="--scene example:mpm_granular --seconds 1 --stride 6"
  [multi_material]="--scene example:mpm_multi_material --seconds 1 --stride 6"
  [twoway]="--scene example:mpm_twoway_coupling --seconds 1 --stride 2"
  [anymal]="--scene example:mpm_anymal --seconds 1 --stride 2"
  [plate_drag]="--scene plate --set 'mode=\"drag\"' --set travel=0.3 --stride 15"
  [column_a2]="--scene column --set aspect=2"
)
for s in ${SCENES:-twoway plate_drag multi_material granular anymal column_a2}; do
  eval "args=(${SC[$s]})"
  rec ${s}_base2 "${args[@]}"
  for k in 1 2 3 4 5 6; do rec ${s}_j$k "${args[@]}" --jitter-ulp 1 --seed $k; done
  rec ${s}_nofma_j7 "${args[@]}" --cuda-nofma --jitter-ulp 1 --seed 7
  rec ${s}_cubin_j8 "${args[@]}" --cuda-cubin --jitter-ulp 1 --seed 8
  rec ${s}_sah_j9 "${args[@]}" --cuda-sah --jitter-ulp 1 --seed 9
  rec ${s}_precise_j10 "${args[@]}" --precise-cuda --jitter-ulp 1 --seed 10
  rec ${s}_nofma_cubin "${args[@]}" --cuda-nofma --cuda-cubin
  rec ${s}_precise_sah "${args[@]}" --precise-cuda --cuda-sah
done
tar czf $OUT/mpm_floors2.tgz -C $OUT mpm_floors2
touch $R/DONE
