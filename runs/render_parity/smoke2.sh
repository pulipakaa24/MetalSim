#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python -m pytest tests/test_render_tier2.py -q -s -k "rtx_tonemap" 2>&1 | grep -E "mean|passed|failed|Error"
.venv/bin/python runs/render_parity/smoke.py 2>&1 | grep -v Warning | tail -8
runs/render_parity/job2.sh
