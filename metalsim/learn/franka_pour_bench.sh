#!/bin/zsh
# Franka-Pour throughput on this Mac at NUM envs (default 2048), three timing-class jobs through the GPU queue, each
# under scripts/watchdog.py: step only (random actions), rollout + inference, the full PPO loop (one 96-step
# iteration after one warm-up iteration). Usage: metalsim/learn/franka_pour_bench.sh [GPU_RUN] [PYTHON] [NUM] [TAG]
set -u
W="$(cd "$(dirname "$0")/../.." && pwd)"
GPU_RUN=${1:-$W/scripts/gpu_run.sh}
PY=${2:-$W/.venv-newton152/bin/python}
NUM=${3:-2048}
TAG=${4:-mac}
OUT=$W/runs/franka_pour
mkdir -p $OUT
for m in step rollout; do
  $GPU_RUN pour_bench_$m timing 6 -- python3 $W/scripts/watchdog.py --timeout 1200 --cwd $W -- \
    $PY -m metalsim.learn.franka_pour_ppo bench --num-envs $NUM --mode $m --steps 20 --warmup 3 \
    --json $OUT/bench_${TAG}_${NUM}_$m.json > $OUT/bench_${TAG}_${NUM}_$m.log 2>&1
done
$GPU_RUN pour_bench_ppo timing 12 -- python3 $W/scripts/watchdog.py --timeout 1500 --cwd $W -- \
  $PY -m metalsim.learn.franka_pour_ppo bench --num-envs $NUM --mode ppo --steps 1 --warmup 1 \
  --json $OUT/bench_${TAG}_${NUM}_ppo.json > $OUT/bench_${TAG}_${NUM}_ppo.log 2>&1
echo done
