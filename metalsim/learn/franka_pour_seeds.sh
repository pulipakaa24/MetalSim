#!/bin/bash
# Franka-Pour seed study: the matched 1,024-env, 60-iteration configuration (PARITY.md §1b.5) for the given seeds,
# one queued `train` job per seed, each under scripts/watchdog.py. Learning runs only (no timing claims).
#   franka_pour_seeds.sh QUEUE_CHECKOUT PYTHON WORKDIR SEED...
#   e.g. metalsim/learn/franka_pour_seeds.sh ~/metalsim-fresh/MetalSim ~/metalsim-fresh/MetalSim/.venv-newton152/bin/python \
#        ~/metalsim-fresh/agents/pour 1 2
set -u
Q=$1; PY=$2; W=$3; shift 3
mkdir -p "$W/runs/franka_pour/seeds"
cd "$Q"
for s in "$@"; do
  out=runs/franka_pour/seeds/s$s
  scripts/gpu_run.sh pour_seed$s train 120 -- python3 "$W/scripts/watchdog.py" --timeout 20000 --stall 300 --rss-gb 48 \
    --cwd "$W" -- "$PY" -m metalsim.learn.franka_pour_ppo train --num-envs 1024 --iterations 60 --seed "$s" \
    --out "$out" > "$W/$out.log" 2>&1
  echo "seed $s exit $?" >> "$W/runs/franka_pour/seeds/status.txt"
done
