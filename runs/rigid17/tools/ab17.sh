#!/bin/bash
# ab17.sh   Interleaved A/B of Franka-Pour collide on the Newton 1.7 line, pinned Warp build:
# A = n17_at (barrier fix + 64-bit atomics), B = n17_new (+ per-thread midphase, 32 threadgroups/core, bounded sort).
# Same saved in-contact state per world count (settled with A); 3 repetitions, A and B alternating; eager and --det.
D=runs/rigid17
J=$D/tools/job.sh
for W in 256 2048; do
  [ -f $D/out/state$W.npz ] && [ "$W" = 256 ] || $J settle${W}_at n17_at check --timeout 1800 --stall 120 -- \
      --worlds $W --settle 60 --state-out $D/out/state$W.npz --max-triangle-pairs 8388608 > /dev/null
done
for W in 256 2048; do
  for rep in 1 2 3; do
    for T in at new; do
      $J ab_${T}_${W}_r$rep n17_$T timing --timeout 1800 --stall 120 -- --worlds $W --state-in $D/out/state$W.npz \
          --max-triangle-pairs 8388608 --steps 20 > /dev/null
      $J abdet_${T}_${W}_r$rep n17_$T timing --timeout 1800 --stall 120 -- --worlds $W --det --state-in \
          $D/out/state$W.npz --max-triangle-pairs 8388608 --steps 20 > /dev/null
    done
  done
done
python3 - <<'EOF'
import json, glob, os, statistics as st
D = "runs/rigid17/out"
rows = {}
for f in sorted(glob.glob(f"{D}/ab*_r*.json")):
    r = json.load(open(f)); k = os.path.basename(f).rsplit("_r", 1)[0]
    rows.setdefault(k, []).append((r["collide_ms"]["median"], r["rigid_contacts"], r["contact_sha"]))
for k, v in rows.items():
    print(k, "median of reps %.2f ms" % st.median(x[0] for x in v), [round(x[0], 2) for x in v], "contacts", {x[1] for x in v}, "sha", {x[2] for x in v})
EOF
