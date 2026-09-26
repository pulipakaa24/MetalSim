#!/bin/zsh
# Tier-2 environment-map port: bit-for-bit check of the parity presets (HEAD worktree vs working tree),
# gallery frame, and the tier-2 test file after / before. Run through the GPU queue:
#   scripts/gpu_run.sh envmap_port render 20 -- zsh runs/render/envmap/run_all.sh
set -u
cd /Users/aditya/robosim
OUT=runs/render/envmap
WT=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/head
PY=.venv/bin/python
echo "== $(date) before: HEAD worktree $(git -C $WT rev-parse --short HEAD)"
PYTHONPATH=$WT $PY scripts/diagnostics/tier2_envmap_regression.py $OUT/before.npz 2>&1 | grep -v "^ \|Warp\|CUDA\|Kernel cache"
echo "== $(date) after: working tree"
$PY scripts/diagnostics/tier2_envmap_regression.py $OUT/after.npz 2>&1 | grep -v "^ \|Warp\|CUDA\|Kernel cache"
echo "== $(date) compare"
$PY scripts/diagnostics/tier2_envmap_regression.py --compare $OUT/before.npz $OUT/after.npz
echo "== $(date) gallery frame"
$PY docs/gallery/scripts/16_g1_hdri.py assets/polyhaven/hdri/train/lebombo.hdr docs/gallery/g1_hdri_tier2.png 2>&1 | grep -v "^ \|Warp\|CUDA\|Kernel cache"
echo "== $(date) pytest after (working tree)"
$PY -m pytest tests/test_render_tier2.py -q -s 2>&1 | grep -v "^ \|Warp\|CUDA\|Kernel cache" > $OUT/pytest_after.log; tail -3 $OUT/pytest_after.log
echo "== $(date) pytest before (HEAD worktree test file + code)"
PYTHONPATH=$WT $PY -m pytest $WT/tests/test_render_tier2.py -q -p no:cacheprovider 2>&1 | grep -v "^ \|Warp\|CUDA\|Kernel cache" > $OUT/pytest_before.log; tail -3 $OUT/pytest_before.log
echo "== $(date) done"
