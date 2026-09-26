#!/bin/zsh
# Queued (GPU queue, kind low): every SolverKamino run of the closed-loop study, CPU device and Metal.
#   scripts/gpu_run.sh cl_kamino low 45 -- scripts/diagnostics/closed_loops/kamino_jobs.sh
cd "$(dirname "$0")/../../.."
python3 scripts/gpu_lock.py status
O=runs/closed_loops/kamino; mkdir -p $O
K=scripts/diagnostics/closed_loops/kamino_probe.py; TO="python3 scripts/diagnostics/closed_loops/run_to.py"
F=.venv-newtonfork/bin/python; N=.venv-newton152/bin/python
# 1. CPU vs Metal agreement, passive four-bar (fork 1.7.0.dev), 2.5 ms, 5 s
for dev in metal:0; do  # CPU-device counterparts ran unqueued-allowed on 2026-09-26 (runs/closed_loops/cpu)
  ${=TO} 600 $F $K --mech fourbar --device $dev --dt 0.0025 --T 5 --out $O/fork_fourbar_none_dt0.0025_${dev%:0}.npz
done
# 2. Newton 1.5.2 (Isaac Lab 3.0 EA's pin): same run
for dev in metal:0; do  # CPU-device counterparts ran unqueued-allowed on 2026-09-26 (runs/closed_loops/cpu)
  ${=TO} 600 $N $K --mech fourbar --device $dev --dt 0.0025 --T 5 --out $O/152_fourbar_none_dt0.0025_${dev%:0}.npz
done
# 3. leg passive, both steps; four-bar at 5 ms; Metal and CPU
for dev in metal:0; do
  ${=TO} 600 $F $K --mech leg --device $dev --dt 0.0025 --T 2 --out $O/fork_leg_none_dt0.0025_${dev%:0}.npz
  ${=TO} 600 $F $K --mech leg --device $dev --dt 0.005 --T 2 --out $O/fork_leg_none_dt0.005_${dev%:0}.npz
  ${=TO} 600 $F $K --mech fourbar --device $dev --dt 0.005 --T 5 --out $O/fork_fourbar_none_dt0.005_${dev%:0}.npz
done
# 4. Baumgarte alpha sensitivity (bilateral), Metal, four-bar passive 2.5 ms
for al in 0.0 0.1 0.5; do
  ${=TO} 600 $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 5 --alpha $al --out $O/fork_fourbar_none_dt0.0025_metal_alpha$al.npz
done
# 5. sigma3 torques, world 0 trajectory (four-bar vs reference) + 1024-world stress (closure, blow-ups), Metal
${=TO} 600 $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 5 --torque sigma3 --out $O/fork_fourbar_sigma3_dt0.0025_metal.npz
for mech in fourbar leg; do for dt in 0.0025 0.005; do
  ${=TO} 900 $F $K --mech $mech --device metal:0 --dt $dt --T 8 --worlds 1024 --torque sigma3 --out $O/fork_${mech}_sigma3_w1024_dt${dt}_metal.npz
done; done
# 6. throughput, 1024 worlds: graph-captured (capture_while on Metal) and eager
for mech in fourbar leg; do
  ${=TO} 600 $F $K --mech $mech --device metal:0 --dt 0.0025 --T 0.05 --worlds 1024 --bench_steps 200 --capture 1
  ${=TO} 600 $F $K --mech $mech --device metal:0 --dt 0.0025 --T 0.05 --worlds 1024 --bench_steps 40 --capture 0
done
${=TO} 600 $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 0.05 --worlds 4096 --bench_steps 200 --capture 1
# (CPU 1024-world bench dropped: CPU timing is not a claim here)
# 6b. Isaac Lab 3.0's Kamino settings (isaaclab_newton kamino_manager_cfg.py: PADMM tol 1e-4, 100 it., bilateral
#     Baumgarte alpha 0.1, graph conditionals off): passive four-bar, 1024-world stress, throughput
IL="--tol 1e-4 --iters 100 --alpha 0.1 --graph_conditionals 0"
${=TO} 600 $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 5 ${=IL} --out $O/fork_fourbar_none_dt0.0025_metal_il3.npz
${=TO} 600 $F $K --mech leg --device metal:0 --dt 0.0025 --T 2 ${=IL} --out $O/fork_leg_none_dt0.0025_metal_il3.npz
${=TO} 900 $F $K --mech leg --device metal:0 --dt 0.0025 --T 8 --worlds 1024 --torque sigma3 ${=IL} --out $O/fork_leg_sigma3_w1024_dt0.0025_metal_il3.npz
${=TO} 600 $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 0.05 --worlds 1024 --bench_steps 200 --capture 1 ${=IL}
${=TO} 600 $F $K --mech leg --device metal:0 --dt 0.0025 --T 0.05 --worlds 1024 --bench_steps 200 --capture 1 ${=IL}
# 7. Newton's own example (fork), Metal
${=TO} 600 $F -m newton.examples kamino_basic_fourbar --viewer null --device metal:0 --num-frames 50 --world-count 16 --test
echo "kamino_jobs done"
