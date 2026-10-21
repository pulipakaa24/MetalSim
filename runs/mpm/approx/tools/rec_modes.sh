#!/bin/bash
# rec_modes.sh STAGES MODE...  -- Metal recordings (record_variant_suite.sh run ids) per fast-math mode, Warp branch
# metalsim-cuda-fast-math; output runs/mpm/fm_<MODE>/
W=.
ST=$1; shift
for MODE in "$@"; do
  KIND=check STAGES="$ST" EXTRA="--metal-fast-math $MODE" WARP_CACHE_PATH=$W/runs/mpm/kcache_cfm \
    bash $W/metalsim/physics/mpm/record_variant_suite.sh $W/runs/mpm/fm_$MODE $W/_forks/warp \
    upstream/newton-integrated $W
done
echo REC_MODES_DONE
