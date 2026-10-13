#!/bin/bash
S=/private/tmp/claude-501/-Users-aditya-robosim/3a553738-d88f-4d8d-ac36-4ecff3ac4bc1/scratchpad/rigid_fma
cd /Users/aditya/robosim; O=runs/fma_rigid_parity/rollouts; mkdir -p $O
tree() { case $1 in B) echo warp-innate;; C) echo warp-innate-ordered;; F) echo warp-innate-fpcontract;; esac; }
for terr in flat rough; do for r in 1 2; do for k in B C F; do
  PYTHONPATH=upstream/$(tree $k):.:$S WARP_CACHE_PATH=$S/cache_$(tree $k) .venv/bin/python $S/g1_rollout.py $O/${k}${r}_$terr.npz 512 $terr 200 2>&1 | /usr/bin/grep -v "^Module\|^Warp\|^ \|CUDA"
done; done; done
