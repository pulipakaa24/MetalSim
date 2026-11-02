#!/bin/bash
# J2: E0 threshold refinement (one wait per process), then the instrumented G1 flat probe next to
# (1) one compute load, (2) two compute loads, (3) a 30 GB touched-but-idle memory load.
S=runs/gpu_watchdog_2026-11-01
PY=<repo>/.venv/bin/python
cd <repo>
echo "== E0 refine $(date +%H:%M:%S)"; $S/e0_one.sh 5 6 7 > $S/j2_e0.out 2>&1
probe() {  # name
  rm -f $S/$1.log*
  echo "== $1 start $(date +%H:%M:%S)"
  $PY $S/g1_probe.py 4096 flat train 2 $S/$1.log - 0.0025 > $S/$1.out 2>&1; echo "== $1 rc $? $(date +%H:%M:%S)"
}
# (1) one compute load
$PY $S/load_compute.py 90 load1 > $S/j2_c1_load1.out 2>&1 & L1=$!
sleep 3; probe j2_c1; kill $L1 2>/dev/null; wait $L1 2>/dev/null
sleep 3
# (2) two compute loads
$PY $S/load_compute.py 90 load1 > $S/j2_c2_load1.out 2>&1 & L1=$!
$PY $S/load_compute.py 90 load2 > $S/j2_c2_load2.out 2>&1 & L2=$!
sleep 3; probe j2_c2; kill $L1 $L2 2>/dev/null; wait $L1 $L2 2>/dev/null
sleep 3
# (3) 30 GB memory load, idle
$PY $S/load_memory.py 30 120 > $S/j2_m30_load.out 2>&1 & M=$!
for i in $(seq 1 60); do grep -q allocated $S/j2_m30_load.out 2>/dev/null && break; sleep 1; done
cat $S/j2_m30_load.out
probe j2_m30; kill $M 2>/dev/null; wait $M 2>/dev/null
echo "== done $(date +%H:%M:%S)"
