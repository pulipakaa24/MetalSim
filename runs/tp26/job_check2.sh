#!/bin/bash
# tp26 checks 2: fused vs unfused per-iteration launches within the worktree (isolates the fusion), g1_tp_check.py
# state-difference protocol (512 envs, 50 steps, the task default preset). The unfused run is the reference file.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp:/Users/aditya/robosim/upstream/mujoco_warp-tp
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-tp rev-parse --short HEAD)"
run() { PYTHONPATH=$WT MJW_METAL_FUSE_UPDATE=$1 MJW_TP_VARIANT="{\"label\":\"$2\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_check.py','512','50']+sys.argv[1:]; runpy.run_path('scripts/diagnostics/g1_tp_check.py', run_name='__main__')" $3 2>&1 | grep -v "^Module\|^Warp\|^   [^s]\|^$"; }
echo "--- unfused (reference) $(date +%H:%M:%S)"; run 0 "worktree unfused" "--out runs/tp26/check2_unfused.npz"
echo "--- fused vs unfused $(date +%H:%M:%S)"; run 1 "worktree fused" "--ref runs/tp26/check2_unfused.npz --out runs/tp26/check2_fused.npz"
echo "=== end $(date)"
