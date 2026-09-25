#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python -m pytest tests/test_render_tier0.py tests/test_render_tier2.py tests/test_scene_usd.py -q 2>&1 | tail -8
.venv/bin/python -m pytest tests/test_render_tier2.py -q -s -k "usd_units or rtx_tonemap or lambertian or furnace" 2>&1 | grep -E "expected|mean|passed|failed|Error"
.venv/bin/python runs/render_parity/smoke.py 2>&1 | grep -v Warning
