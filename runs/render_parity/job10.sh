#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python runs/render_parity/cp_img.py 2>&1 | grep -E "^(legacy|physical)|Traceback|Error"
M=physical; rm -f runs/render_parity/cartpole_smoke_physical_ex025.log
.venv/bin/python -m metalsim.learn.train_cartpole_rgb --envs 1024 --tier 2 --steps 655360 --seed 42 --log runs/render_parity/cartpole_smoke_physical_ex025.log > /dev/null 2>runs/render_parity/cartpole_smoke_physical_ex025.err
cat runs/render_parity/cartpole_smoke_physical_ex025.log; tail -3 runs/render_parity/cartpole_smoke_physical_ex025.err
.venv/bin/python -m pytest tests/test_render_tier2.py tests/test_render_tier0.py -q 2>&1 | tail -1
