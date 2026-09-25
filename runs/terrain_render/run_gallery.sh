#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python docs/gallery/scripts/15_g1_rough_boxes.py 2>&1 | grep -v "Module \|Warn"
