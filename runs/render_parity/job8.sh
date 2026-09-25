#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python -m pytest tests/test_render_tier2.py tests/test_render_tier0.py -q 2>&1 | tail -2
echo "== batch cost (1024 envs, step incl. physics+render+reward)"
.venv/bin/python -c "
import warp as wp; wp.config.quiet=True
from metalsim.learn.cartpole_rgb import benchmark
n=1024
for mode,spp in (('legacy',1),('legacy',4),('physical',4)):
    r=benchmark(n, steps=50, render=True, tier=2, spp=spp, max_bounces=2, render_mode=mode)
    print(f'tier 2 {mode} spp {spp}: {r:,.0f} env-steps/s = {n/r*1e3:.1f} ms per 1024-env step')
" 2>&1 | grep tier
echo "== smoke training, same seed, legacy then physical"
for M in legacy physical; do
  timeout 150 .venv/bin/python -m metalsim.learn.train_cartpole_rgb --envs 1024 --tier 2 --render_mode $M --steps 700000 --seed 42 --log runs/render_parity/cartpole_smoke_$M.log > /dev/null 2>&1
  echo "-- $M"; tail -12 runs/render_parity/cartpole_smoke_$M.log
done
