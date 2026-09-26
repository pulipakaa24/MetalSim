#!/bin/zsh
# "Before" half of the bit-for-bit check: render the parity presets from the previous renderer (HEAD worktree
# 98a41c1, fetched G1 USD symlinked in) and compare with after.npz. Enqueue after the queue resumes:
#   scripts/gpu_run.sh envmap_before render 5 -- zsh runs/render/envmap/run_before.sh
set -u
cd /Users/aditya/robosim
OUT=runs/render/envmap
WT=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/head
echo "== $(date) before: HEAD worktree $(git -C $WT rev-parse --short HEAD)"
PYTHONPATH=$WT .venv/bin/python scripts/diagnostics/tier2_envmap_regression.py $OUT/before.npz 2>&1 | grep -v "Warning: in Bindings\|^ \|Warp\|CUDA\|Kernel cache"
echo "== $(date) compare"
.venv/bin/python scripts/diagnostics/tier2_envmap_regression.py --compare $OUT/before.npz $OUT/after.npz
echo "== $(date) done"
