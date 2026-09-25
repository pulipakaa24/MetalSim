#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python -m pytest tests -q -x 2>&1 | grep -v "Module \|^Warp\|CUDA not\|Devices:\|\"cpu\"\|\"metal:0\"\|Kernel cache\|Library/Caches" | tail -40
