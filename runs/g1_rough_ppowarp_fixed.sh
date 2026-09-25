#!/bin/bash
# Rough like-for-like run (Isaac G1RoughEnvCfg exactly, reward_cfg rough_isaac); gate on the formula test first.
cd /Users/aditya/robosim
.venv/bin/python -m pytest tests/test_g1_task_terms.py -q -k rough_isaac >> runs/g1_rough_ppowarp_fixed.log 2>&1 || { echo "rough_isaac test failed; not training" >> runs/g1_rough_ppowarp_fixed.log; exit 1; }
exec .venv/bin/python -m metalsim.learn.g1_velocity 4096 rough train 1500 runs/g1_rough_ppowarp_fixed.log runs/policies/g1_rough_ppowarp_fixed.pt 0.0025 --seed 0 --reward_cfg rough_isaac
