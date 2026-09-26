#!/bin/zsh
# MuJoCo C (pure CPU, no Warp) solref/solimp sweep on the closed-loop mechanisms; outputs runs/closed_loops/c/*.npz
cd "$(dirname "$0")/../../.."
P=.venv/bin/python; S=scripts/diagnostics/closed_loops/mjw_loops.py; O=runs/closed_loops/c; mkdir -p $O
for mech torque T in fourbar none 5 fourbar sigma3 5 leg none 2; do
  for dt in 0.0025 0.005; do
    for sr in "0.02,1" "0.01,1" "$(python3 -c "print(2*$dt)"),1"; do
      for si in "0.9,0.95,0.001,0.5,2" "0.99,0.999,0.001,0.5,2"; do
        $P $S traj --engine c --mech $mech --torque $torque --T $T --dt $dt --solref $sr --solimp $si --out $O/${mech}_${torque}_dt${dt}_sr${sr%,1}_si${si%%,*}.npz > /dev/null
      done
    done
  done
done
