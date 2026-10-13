#!/bin/bash
# Rigid-side members of the MPM floor ensemble (MPM.md section 11): stage_mpm_floors2.sh's ANYmal settings with every
# initial joint coordinate of the robot moved by k ulp, k in {-1, 0, 1} (record_mpm.py --jitter-body-ulp), seeds
# 1-6. In the first frames the floor2 ensemble moves the robot only through its no-FMA runs; these runs give the
# rigid side the same 1-ulp perturbation the particles already get. Output ~/parity3/mpm_floors3/anymal_bj<k>/.
source $HOME/parity3_scripts/common.sh
exec >> $OUT/stage_mpm_floors3.log 2>&1
set -x
M=$HOME/parity3_scripts/mpm/floors2; R=$OUT/mpm_floors3; mkdir -p $R
rm -f $R/DONE
for k in 1 2 3 4 5 6; do
  wait_for_gpu
  [ -f $R/anymal_bj$k/meta.json ] && { echo "skip anymal_bj$k"; continue; }
  timeout 1800 python $M/record_mpm.py --device cuda:0 --accumulate float32 --out $R/anymal_bj$k \
      --scene example:mpm_anymal --seconds 1 --stride 2 --jitter-body-ulp 1 --seed $k > $R/anymal_bj$k.log 2>&1
  echo "anymal_bj$k exit $?"; tail -1 $R/anymal_bj$k.log
done
touch $R/DONE
