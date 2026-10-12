#!/bin/bash
# multi_material ms/frame with the closeout tree, metal_cuda_libm on vs off (interleaved, 3 repetitions)
W=/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c
MS=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
PY=/Users/aditya/robosim/.venv-newton152/bin/python
B=$MS/metalsim/physics/mpm/probes/speed_bench.py
O=$W/runs/mpm/closeout/speed; mkdir -p $O
for rep in 1 2 3; do
  for v in 1 0; do
    K=$W/runs/mpm/closeout/kcache_tile; [ $v = 0 ] && K=$W/runs/mpm/closeout/kcache_tileonly
    WP_METAL_CUDA_LIBM=$v PYTHONPATH=$W/_forks/warp:/Users/aditya/robosim/upstream/newton-integrated:$MS WARP_CACHE_PATH=$K \
      $PY $B example:mpm_multi_material --frames 60 --warmup 5 --metal-fast-math off --json $O/mm_libm${v}_$rep.json > $O/mm_libm${v}_$rep.log 2>&1
    echo "libm=$v rep $rep: $(tail -1 $O/mm_libm${v}_$rep.log | cut -c1-110)"
  done
done
