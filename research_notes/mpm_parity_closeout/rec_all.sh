#!/bin/bash
# rec_all.sh OUT_DIR WARP_TREE [scenes...]   -- the six floor scenes (rec.sh), sequentially
OUT=$1; WARP=$2; shift 2
SC=${*:-"multi_material granular column_a2 plate_drag twoway anymal"}
D=$(dirname "$0")
for s in $SC; do "$D/rec.sh" "$OUT" "$WARP" "$s"; done
echo ALL_DONE
