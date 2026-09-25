#!/bin/bash
# Stage 3: Isaac Lab 3.0's own rsl_rl training (rsl-rl-lib 5.4.1, the task's G1FlatPPORunnerCfg), 4096 envs, seed 0,
# 1500 iterations, stock task config (incl. 3.0's push_robot / add_base_mass events), on each backend; then rough on Newton.
# Usage: stage3_train.sh "flat:newton_mjwarp flat:isaacsim_physx rough:newton_mjwarp"
source $HOME/parity3_scripts/common.sh
exec >> $OUT/stage3_train.log 2>&1
set -x
JOBS=${1:-"flat:newton_mjwarp flat:isaacsim_physx rough:newton_mjwarp"}
mkdir -p $OUT/train
for J in $JOBS; do
  T=${J%%:*}; B=${J##*:}
  case $T in flat) TASK=Isaac-Velocity-Flat-G1; EXP=g1_flat;; rough) TASK=Isaac-Velocity-Rough-G1; EXP=g1_rough;; esac
  wait_for_gpu; t0=$(date +%s)
  log=$OUT/train/train_${T}_$B.log
  timeout 21600 isaaclab train --rl_library rsl_rl --task $TASK --num_envs 4096 --max_iterations 1500 --seed 0 \
     --run_name ${B} --viz none physics=$B $NOVIS > $log 2>&1
  echo "train $T $B exit $? in $(( $(date +%s) - t0 )) s"
  run=$(ls -td logs/rsl_rl/$EXP/*_${B} 2>/dev/null | head -1)
  [ -n "$run" ] && rm -rf $OUT/train/rsl_rl_${T}_$B && cp -r $run $OUT/train/rsl_rl_${T}_$B
  sed 's/\x1b\[[0-9;]*m//g' $log | sed -n '/Learning iteration/,/Iteration time/p' | grep -a -v '^\s*$' > $OUT/train/train_${T}_${B}_terms.txt
  touch $OUT/TRAIN_${T}_${B}_DONE
done
