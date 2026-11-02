#!/bin/bash
# one E0 torch wait per process (a timed-out command buffer can poison later submissions of the same process)
S=runs/gpu_watchdog_2026-11-01
for X in "$@"; do <repo>/.venv/bin/python $S/e0_wait_timeout.py $X 2>&1 | grep -v "^Warp \|^   \|^     "; done
