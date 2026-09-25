#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python scripts/diagnostics/fast_factorization_scenes.py old --out runs/fastfact/old.npz --n 64 --ntp 4096
.venv/bin/python scripts/diagnostics/fast_factorization_scenes.py new --out runs/fastfact/new.npz --ref runs/fastfact/old.npz --n 64 --ntp 4096
