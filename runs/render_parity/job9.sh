#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
for M in legacy physical; do
  rm -f runs/render_parity/cartpole_smoke_$M.log
  .venv/bin/python -m metalsim.learn.train_cartpole_rgb --envs 1024 --tier 2 --render_mode $M --steps 655360 --seed 42 --log runs/render_parity/cartpole_smoke_$M.log > /dev/null 2>runs/render_parity/cartpole_smoke_$M.err
  echo "-- $M"; cat runs/render_parity/cartpole_smoke_$M.log; tail -3 runs/render_parity/cartpole_smoke_$M.err
done
