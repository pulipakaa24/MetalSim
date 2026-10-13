#!/bin/bash
# Interleaved ms/frame A/B: Warp before (agent-ae86 fork copy, 11aa54c8) vs after (this closeout's changes), off mode.
# (macOS bash 3.2: no associative arrays)
W=/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c
MS=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
PY=/Users/aditya/robosim/.venv-newton152/bin/python
B=$MS/metalsim/physics/mpm/probes/speed_bench.py
O=${OUT:-$W/runs/mpm/closeout/speed}; mkdir -p $O
for rep in 1 2; do
  for s in "example:mpm_multi_material" "example:mpm_granular" "column --set aspect=2" "example:mpm_anymal"; do
    for v in before after; do
      if [ $v = before ]; then T=$MS/_forks/warp; K=$W/runs/mpm/closeout/kcache_speed_before
      else T=$W/_forks/warp; K=$W/runs/mpm/closeout/kcache_tile; fi
      tag=$(echo "$s" | tr ' :=' '___')
      PYTHONPATH=$T:/Users/aditya/robosim/upstream/newton-integrated:$MS WARP_CACHE_PATH=$K \
        $PY $B $s --frames 60 --warmup 5 --metal-fast-math off --json $O/${tag}_${v}_$rep.json > $O/${tag}_${v}_$rep.log 2>&1
      echo "$v $s rep $rep: $(tail -1 $O/${tag}_${v}_$rep.log | cut -c1-120) | $(tail -1 $O/${tag}_${v}_$rep.log | /usr/bin/grep -o '"warp": "[^"]*"')"
    done
  done
done
