#!/bin/bash
# ab17b.sh   Short timing follow-up (queue courtesy: few jobs). 2048 worlds eager A/B x2, 256 worlds deterministic A/B x1.
D=runs/rigid17
J=$D/tools/job.sh
for rep in 1 2; do
  for T in at new; do
    $J ab_${T}_2048_r$rep n17_$T timing --timeout 1800 --stall 120 -- --worlds 2048 --state-in $D/out/state2048.npz \
        --max-triangle-pairs 8388608 --steps 10 > /dev/null
  done
done
for T in at new; do
  $J det_${T}_256_r1 n17_$T timing --timeout 1800 --stall 120 -- --worlds 256 --det --state-in $D/out/state256.npz \
      --steps 10 > /dev/null
done
