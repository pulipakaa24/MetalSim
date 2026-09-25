#!/bin/bash
# rerun: tron1_sim with its corrected protocol (both configs), and repeat throughput for tron1_wf / panda (order new, old, new)
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
S=tron1_sim,tron1_wf,panda
.venv/bin/python scripts/diagnostics/fast_factorization_scenes.py new --out runs/fastfact/new2a.npz --scenes $S --n 64 --ntp 4096
.venv/bin/python scripts/diagnostics/fast_factorization_scenes.py old --out runs/fastfact/old2.npz --ref runs/fastfact/new2a.npz --scenes $S --n 64 --ntp 4096
.venv/bin/python scripts/diagnostics/fast_factorization_scenes.py new --out runs/fastfact/new2b.npz --ref runs/fastfact/old2.npz --scenes $S --n 64 --ntp 4096
