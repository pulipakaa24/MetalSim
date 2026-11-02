#!/bin/bash
# J1: no contention. E0 wait sweep (torch, then Warp), then the instrumented G1 flat probe alone.
S=runs/gpu_watchdog_2026-11-01
PY=<repo>/.venv/bin/python
cd <repo>
echo "== E0 torch $(date +%H:%M:%S)"; $PY $S/e0_wait_timeout.py 1 2 4 8 16 > $S/j1_e0_torch.out 2>&1; echo "rc $?"
echo "== E0 warp $(date +%H:%M:%S)"; $PY $S/e0_wait_timeout.py --warp 4 16 > $S/j1_e0_warp.out 2>&1; echo "rc $?"
echo "== G1 flat alone $(date +%H:%M:%S)"; rm -f $S/j1_flat.log*
$PY $S/g1_probe.py 4096 flat train 2 $S/j1_flat.log - 0.0025 > $S/j1_flat.out 2>&1; echo "rc $?"
echo "== done $(date +%H:%M:%S)"
