#!/bin/bash
# Metal tiling decisions for the six floor scenes (tile_decisions.py), compared with the L4's.
W=.
export PYTHONPATH=$W/_forks/warp:upstream/newton-integrated:.
export WARP_CACHE_PATH=$W/runs/mpm/closeout/kcache_tile
PY=.venv-newton152/bin/python
T=$W/research_notes/mpm_parity_closeout/tile_decisions.py
O=$W/runs/mpm/closeout/tiles_metal; mkdir -p $O
for s in mpm_multi_material mpm_granular mpm_twoway_coupling mpm_anymal; do
  $PY $T --device metal:0 --scene example:$s --steps 4 --out $O/$s.json 2>&1 | tail -1
done
$PY $T --device metal:0 --scene column --set aspect=2 --steps 4 --out $O/column_a2.json 2>&1 | tail -1
$PY $T --device metal:0 --scene plate --set 'mode="drag"' --set travel=0.3 --steps 4 --out $O/plate_drag.json 2>&1 | tail -1
for f in $W/runs/parity/isaaclab3_mpm_closeout/tiles/*.json; do
  b=$(basename $f); cmp -s <($PY -c "import json,sys;print(json.load(open(sys.argv[1])))" $f) <($PY -c "import json,sys;print(json.load(open(sys.argv[1])))" $O/$b) && echo "$b identical" || echo "$b DIFFERS"
done
