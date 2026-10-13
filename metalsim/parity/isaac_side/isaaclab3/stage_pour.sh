#!/bin/bash
# Franka-Pour on the L4 with Isaac Lab 3.0-EA (release/3.0.0 ae37b028) and Isaac's own task code.
# Isaac's task does not spawn with the published Isaac 6.1 Franka (its spawn hook asserts a second finger mimic and
# arm collision proxies the asset's default composition lacks; stage_pour_smoke.sh records the failure), so this
# applies franka_pour_published_asset.patch (variant Colliders=convex_hulls, keep the asset's one finger mimic, no
# hand hull: the model metalsim composes) for the run and reverts it afterwards.
#   STAGES="smoke bench train" (default smoke bench)
#   smoke: 4 envs, 1 iteration.  bench: `isaaclab benchmark runtime` (random actions, step only) at 2048 envs.
#   train: rsl_rl training at 2048 envs for ITERS iterations (default 3) with the task's runner config; rsl_rl logs
#          collection (rollout + inference) and learning time per iteration.
# Logs: ~/parity3/pour/
source $HOME/parity3_scripts/common.sh
mkdir -p $OUT/pour
exec >> $OUT/pour/stage_pour.log 2>&1
set -x
STAGES=${STAGES:-"smoke bench"}
ITERS=${ITERS:-3}
NUM=${NUM:-2048}            # train: env count (2048 = the README's; it does not fit the L4's 24 GB with CUDA graphs)
EXTRA=${EXTRA:-}            # train: Hydra overrides, e.g. env.sim.physics.use_cuda_graph=false
TAG=${TAG:-}
SEED=${SEED:-0}             # train: rsl_rl / env seed (Isaac physics is not run-to-run deterministic by default)
P=$HOME/parity3_scripts/franka_pour_published_asset.patch
git apply --check $P && git apply $P || { echo "patch did not apply"; exit 1; }
trap 'git apply -R $P' EXIT
nvidia-smi --query-gpu=name,driver_version --format=csv
for S in $STAGES; do
  wait_for_gpu; t0=$(date +%s)
  case $S in
    smoke) timeout 2400 isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour --num_envs 4 --max_iterations 1 \
             --seed 0 --run_name smoke_patched --viz none > $OUT/pour/smoke_patched.log 2>&1 ;;
    bench) timeout 5400 isaaclab benchmark runtime --task IsaacContrib-Franka-Pour --num_envs 2048 --num_steps 20 \
             --warmup_steps 3 --seed 0 --visualizer none --benchmark_formatter json,summary \
             --output_path $OUT/pour/bench_2048 > $OUT/pour/bench_2048.log 2>&1 ;;
    train) timeout 21600 isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour --num_envs $NUM \
             --max_iterations $ITERS --seed $SEED --run_name l4_${NUM}$TAG --viz none $EXTRA > $OUT/pour/train_${NUM}$TAG.log 2>&1 ;;
  esac
  echo "$S exit $? in $(( $(date +%s) - t0 )) s"
done
touch $OUT/pour/STAGE_POUR_DONE
