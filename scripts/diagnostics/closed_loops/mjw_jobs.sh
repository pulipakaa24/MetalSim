#!/bin/zsh
# Queued (GPU queue, kind low): MuJoCo Warp (CPU device and Metal) and the 1024-world MuJoCo C runs of the
# closed-loop study.   scripts/gpu_run.sh cl_mjw low 40 -- scripts/diagnostics/closed_loops/mjw_jobs.sh
cd "$(dirname "$0")/../../.."
python3 scripts/gpu_lock.py status
O=runs/closed_loops/mjw; mkdir -p $O
P=.venv/bin/python; S=scripts/diagnostics/closed_loops/mjw_loops.py; TO="python3 scripts/diagnostics/closed_loops/run_to.py"
# 1. trajectory agreement C / MJW-CPU / MJW-Metal, default solref, 1 world
for mech torque T in fourbar none 5 leg none 2 fourbar sigma3 5; do for dt in 0.0025 0.005; do
  for dev in metal:0; do  # CPU device: runs/closed_loops/cpu
    ${=TO} 600 $P $S traj --engine mjw --device $dev --mech $mech --torque $torque --T $T --dt $dt --out $O/${mech}_${torque}_dt${dt}_${dev%:0}.npz
  done
done; done
# 2. solref sensitivity on Metal (matches the C sweep?), four-bar passive 2.5 ms
for sr in "0.005,1" "0.01,1"; do for si in "0.9,0.95,0.001,0.5,2" "0.99,0.999,0.001,0.5,2"; do
  ${=TO} 600 $P $S traj --engine mjw --device metal:0 --mech fourbar --T 5 --dt 0.0025 --solref $sr --solimp $si --out $O/fourbar_none_dt0.0025_sr${sr%,1}_si${si%%,*}_metal.npz
done; done
# 3. 3-sigma torque stress, 1024 worlds, 8 s (400 control steps at 50 Hz): MJW Metal and MuJoCo C
for mech in fourbar leg; do for dt in 0.0025 0.005; do
  for sr in "0.02,1" "$(python3 -c "print(2*$dt)"),1"; do
    ${=TO} 900 $P $S traj --engine mjw --device metal:0 --mech $mech --torque sigma3 --worlds 1024 --T 8 --dt $dt --solref $sr --out $O/${mech}_sigma3_w1024_dt${dt}_sr${sr%,1}_metal.npz
  done
  # MuJoCo C 1024-world counterpart runs outside the queue (pure CPU): see closed_loops note
done; done
# 4. throughput (graph replay, env-steps/s), with and without the loop-closing equality
for mech in fourbar leg; do for w in 1024 4096; do
  ${=TO} 300 $P $S bench --engine mjw --device metal:0 --mech $mech --worlds $w --dt 0.0025 --bench_steps 2000
  ${=TO} 300 $P $S bench --engine mjw --device metal:0 --mech $mech --worlds $w --dt 0.0025 --bench_steps 2000 --no_eq
done; done
echo "mjw_jobs done"
