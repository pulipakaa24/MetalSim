#!/bin/bash
# Rigid parity harness (runs/fma_rigid_parity/harness, metrics.py) on the closeout Warp tree (warp_push, 43ffc6dd):
# metal_cuda_libm off (= the 152808b7 kernels outside warp.fem) vs on, two runs each, interleaved, one GPU job each.
W=.
R=$W/research_notes/mpm_parity_closeout/rigid_suite.sh
NEW=$W/_forks/warp_push
run() {
  # a finished run (pytest summary line) is kept
  tail -1 $W/runs/fma_rigid_parity/closeout_$1.log 2>/dev/null | /usr/bin/grep -q " passed" && { echo "$1 kept"; return; }
  WP_METAL_CUDA_LIBM=$2 scripts/gpu_run.sh cl_rigsuite_$1 check 15 -- $R $NEW closeout_$1 \
    > $W/runs/fma_rigid_parity/closeout_$1.log 2>&1
  echo "$1 $(tail -1 $W/runs/fma_rigid_parity/closeout_$1.log)"
}
run libmoff1 0
run libmon1 1
run libmoff2 0
run libmon2 1
echo SUITE_DONE
