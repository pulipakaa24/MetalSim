#!/bin/bash
# Cooperative long Franka-Pour training on a shared GPU queue: runs the trainer as kind `train` and, whenever another
# job is waiting in the queue, pauses it at the end of the current iteration (STOP file, checkpoint written), lets the
# queue run the others, and re-queues the resume behind them, until the iteration target is reached.
#   franka_pour_yield.sh QUEUE_CHECKOUT PYTHON WORKDIR OUT NUM_ENVS ITERATIONS NAME
set -u
Q=$1; PY=$2; W=$3; OUT=$4; NUM=$5; ITERS=$6; NAME=$7
LOG=$W/$OUT.log
cd "$Q"
while true; do
  done_it=$(tail -1 "$W/$OUT/log.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.loads(sys.stdin.read() or '{\"it\": -1}')['it'])")
  [ "$done_it" -ge $((ITERS - 1)) ] && { echo "[yield] done at iteration $done_it" >> "$LOG"; exit 0; }
  RES=""; [ -f "$W/$OUT/model_last.pt" ] && RES="--resume $OUT/model_last.pt"
  scripts/gpu_run.sh "$NAME" train 600 -- python3 "$W/scripts/watchdog.py" --timeout 200000 --stall 300 --rss-gb 48 \
    --cwd "$W" -- "$PY" -m metalsim.learn.franka_pour_ppo train --num-envs "$NUM" --iterations "$ITERS" \
    --out "$OUT" $RES >> "$LOG" 2>&1 &
  RUN=$!
  while kill -0 $RUN 2>/dev/null; do
    sleep 120
    holder=$(python3 scripts/gpu_lock.py status | head -1)
    waiting=$(python3 scripts/gpu_lock.py status | grep "waiting \[" | grep -vc " $NAME ")
    if echo "$holder" | grep -q "\"$NAME\"" && [ "$waiting" -gt 0 ] && [ ! -f "$W/$OUT/STOP" ]; then
      echo "[yield] $waiting job(s) waiting: pausing after this iteration ($(date +%H:%M:%S))" >> "$LOG"
      touch "$W/$OUT/STOP"
    fi
  done
  rm -f "$W/$OUT/STOP"
  sleep 30
done
