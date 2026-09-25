#!/bin/bash
# protocol replay with the RTX-parity tier 2 (default --tier2_mode rtx) and the compare against RTX RT and PT
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
.venv/bin/python -m metalsim.parity.record_g1 --isaac runs/parity/isaac/parity_out2/rt --out runs/parity/metalsim2 --physics_dt 0.0025 2>&1 | grep -v Warning | tail -5
.venv/bin/python -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/rt --metalsim runs/parity/metalsim2 --out runs/parity/report_rt4 > runs/parity/compare_rt4.log 2>&1
.venv/bin/python -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/pt --metalsim runs/parity/metalsim2 --out runs/parity/report_pt4 > runs/parity/compare_pt4.log 2>&1
echo compare done
