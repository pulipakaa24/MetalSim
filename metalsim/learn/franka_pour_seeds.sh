#!/bin/bash
# Franka-Pour seed study: the matched 1,024-env, 60-iteration configuration (PARITY.md §1b.5) for the given seeds,
# one queued `train` job per seed, each under scripts/watchdog.py. Learning runs only (no timing claims).
#   franka_pour_seeds.sh QUEUE_CHECKOUT PYTHON WORKDIR SEED...
#   QUEUE_CHECKOUT: the checkout whose GPU queue (scripts/gpu_run.sh) the jobs go through; WORKDIR: the checkout whose
#   code runs and where runs/franka_pour/seeds/ is written (it may be the same checkout). From a checkout's root, e.g.
#   metalsim/learn/franka_pour_seeds.sh "$PWD" "$PWD/.venv-newton152/bin/python" "$PWD" 1 2
set -u
Q=$1; PY=$2; W=$3; shift 3
mkdir -p "$W/runs/franka_pour/seeds"
cd "$Q"
# Resumable: a seed whose log already reaches iteration 59 is skipped; otherwise it continues from its last checkpoint
# (written every 5 iterations; the resume restores networks, optimizer, curriculum, RNG and logger state, and starts
# the envs from fresh rows). If the machine's user kills the job, re-run this script with the same arguments.
for s in "$@"; do
  out=runs/franka_pour/seeds/s$s
  last=$(tail -1 "$W/$out/log.jsonl" 2>/dev/null | python3 -c "import json,sys; t=sys.stdin.read(); print(json.loads(t)['it'] if t else -1)")
  [ "$last" -ge 59 ] && { echo "seed $s already complete" >> "$W/runs/franka_pour/seeds/status.txt"; continue; }
  RES=(); [ -f "$W/$out/model_last.pt" ] && RES=(--resume "$out/model_last.pt")
  scripts/gpu_run.sh pour_seed$s train 120 -- python3 "$W/scripts/watchdog.py" --timeout 20000 --stall 300 --rss-gb 48 \
    --cwd "$W" -- "$PY" -m metalsim.learn.franka_pour_ppo train --num-envs 1024 --iterations 60 --seed "$s" \
    --save-interval 5 --out "$out" ${RES[@]+"${RES[@]}"} >> "$W/$out.log" 2>&1
  echo "seed $s exit $? ($(date +%H:%M))" >> "$W/runs/franka_pour/seeds/status.txt"
done
