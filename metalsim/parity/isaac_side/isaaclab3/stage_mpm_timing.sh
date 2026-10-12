#!/bin/bash
# MPM timing on the L4 (Isaac Lab 3.0-EA's Newton 1.5.2), for the Metal speed comparison at matched settings.
#   T_<scene>_eager    Newton's own solver loop (conditional CUDA graph inside the strain solve), step run eagerly:
#                      what Isaac Lab's NewtonMPMManager does when it cannot capture the step (dense / rebuilt grids)
#   T_<scene>_capture  the same with the example's whole-step graph capture (the best case on CUDA)
# No per-solve logging (no host reads). Output ~/parity3/mpm_timing/<id>/, tarball ~/parity3/mpm_timing.tgz.
source $HOME/parity3_scripts/common.sh
exec >> $OUT/stage_mpm_timing.log 2>&1
set -x
M=$HOME/parity3_scripts/mpm; R=$OUT/mpm_timing; mkdir -p $R
rm -f $OUT/MPM_TIMING_DONE
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
rec() { local id=$1; shift; wait_for_gpu; [ -f $R/$id/meta.json ] && { echo "skip $id"; return; }
        python $M/record_mpm.py --device cuda:0 --loop native --no-log --out $R/$id "$@" > $R/$id.log 2>&1; echo "$id exit $?"; tail -1 $R/$id.log; }
for ex in granular multi_material twoway_coupling anymal; do
  rec T_${ex}_eager   --scene example:mpm_$ex --seconds 2 --stride 1000
  rec T_${ex}_capture --scene example:mpm_$ex --seconds 2 --stride 1000 --set capture=true
done
rec T_column_a2_eager --scene column --set aspect=2 --stride 1000
rec T_column_a8_eager --scene column --set aspect=8 --stride 1000
rec T_plate_drag_eager --scene plate --set 'mode="drag"' --set travel=0.3 --stride 1000
tar czf $OUT/mpm_timing.tgz -C $OUT mpm_timing
touch $OUT/MPM_TIMING_DONE
