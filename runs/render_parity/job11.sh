#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
ps aux | grep python | grep -v grep | grep -v gpu_lock | awk '{print "other python:", $2, $6/1e6 " GB RSS", $12, $13}'
.venv/bin/python runs/render_parity/repro_nan.py 2>&1 | grep -E "^it|accum|Insufficient|Error|Traceback|nan"
.venv/bin/python -m pytest tests/test_render_tier2.py -q 2>&1 | tail -1
