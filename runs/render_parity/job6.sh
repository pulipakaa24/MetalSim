#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python -m metalsim.render.bench_cost --fidelity_rl 2>&1 | grep -v Warning | grep -v Module | tail -20
.venv/bin/python -m pytest tests/test_render_tier0.py tests/test_render_tier2.py tests/test_scene_usd.py -q 2>&1 | tail -4
