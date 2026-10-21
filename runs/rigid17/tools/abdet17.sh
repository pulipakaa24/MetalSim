#!/bin/bash
# abdet17.sh   Deterministic-mode collide A/B at 256 worlds (Newton's default max_triangle_pairs: the 1.7 line's
# deterministic packing allows at most 2^20 buffered contacts): A = n17_at, B = n17_new; contact hashes must agree.
D=runs/rigid17
for rep in 1 2 3; do
  for T in at new; do
    $D/tools/job.sh det_${T}_256_r$rep n17_$T timing --timeout 1800 --stall 120 -- --worlds 256 --det \
        --state-in $D/out/state256.npz --steps 20 > /dev/null
  done
done
