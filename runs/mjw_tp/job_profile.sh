#!/bin/bash
# profile job: baseline cost split + segment graphs + per-kernel attribution (G1 flat, MuJoCo Warp, 4096, 2.5 ms)
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date) power: $(pmset -g batt | head -1)"
python3 scripts/gpu_lock.py status
echo "=== baseline cost split (scripts/diagnostics/g1_engine_bench.py 4096 mjwarp 0.0025)"
python scripts/diagnostics/g1_engine_bench.py 4096 mjwarp 0.0025
echo "=== segment graphs"
python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --reps 20
echo "=== per-kernel"
WP_METAL_PROFILE=1 python scripts/diagnostics/g1_step_profile.py 4096 0.0025 --kernels
echo "=== baseline cost split, repeat"
python scripts/diagnostics/g1_engine_bench.py 4096 mjwarp 0.0025
echo "=== end $(date)"
