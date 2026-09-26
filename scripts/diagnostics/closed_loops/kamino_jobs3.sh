#!/bin/zsh
# Remaining cl_kamino sub-steps as one short step per queue ticket (the 8 s / 1024-world stress took > 6 min per
# run at ~100 ms per step with host-synchronised PADMM iterations and held the GPU; killed 2026-09-26 02:06).
# Usage: scripts/gpu_run.sh cl_k3_STEP low 12 -- scripts/diagnostics/closed_loops/kamino_jobs3.sh STEP
cd "$(dirname "$0")/../../.."
python3 scripts/gpu_lock.py status | head -1
O=runs/closed_loops/kamino; K=scripts/diagnostics/closed_loops/kamino_probe.py; F=.venv-newtonfork/bin/python
TO=(python3 scripts/diagnostics/closed_loops/run_to.py 600)
IL=(--tol 1e-4 --iters 100 --alpha 0.1 --graph_conditionals 0)
case $1 in
  stress_fourbar) ${TO[@]} $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 2 --worlds 1024 --torque sigma3 --out $O/fork_fourbar_sigma3_w1024_dt0.0025_metal.npz ;;
  stress_leg)     ${TO[@]} $F $K --mech leg --device metal:0 --dt 0.0025 --T 2 --worlds 1024 --torque sigma3 --out $O/fork_leg_sigma3_w1024_dt0.0025_metal.npz ;;
  stress_leg_5ms) ${TO[@]} $F $K --mech leg --device metal:0 --dt 0.005 --T 2 --worlds 1024 --torque sigma3 --out $O/fork_leg_sigma3_w1024_dt0.005_metal.npz ;;
  bench)          ${TO[@]} $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 0.01 --worlds 1024 --bench_steps 100 --capture 1
                  ${TO[@]} $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 0.01 --worlds 1024 --bench_steps 20 --capture 0
                  ${TO[@]} $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 0.01 --worlds 1024 --bench_steps 100 --capture 1 ${IL[@]} ;;
  il3)            ${TO[@]} $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 5 ${IL[@]} --out $O/fork_fourbar_none_dt0.0025_metal_il3.npz ;;
  example)        ${TO[@]} $F -m newton.examples kamino_basic_fourbar --viewer null --device metal:0 --num-frames 50 --world-count 16 --test ;;
esac
echo "kamino_jobs3 $1 done"
