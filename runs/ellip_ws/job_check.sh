#!/bin/bash
# ellip_ws: state-difference protocol (g1_tp_check.py 512 envs, 50 steps, the task default = elliptic ellip10 cap 20) on the
# frozen worktree mujoco_warp-ellip-test (5d0d165) / warp-innate-tp-test (9abceff9): unfused per-iteration launches
# (MJW_METAL_FUSE_UPDATE=0, the reference file) vs fused (default), fused + htot fusion (MJW_METAL_FUSE_HTOT=1) and
# fused + extrapolated warm start (MJW_WARMSTART_EXTRAP=1). Each run also prints its own two-instance floor and the MuJoCo C protocol.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test rev-parse --short HEAD)"
run() { PYTHONPATH=$WT env $1 MJW_TP_VARIANT="{\"label\":\"$2\"}" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_check.py','512','50']+sys.argv[1:]; runpy.run_path('scripts/diagnostics/g1_tp_check.py', run_name='__main__')" $3 2>&1 | grep -v "^Module\|^Warp\|^   [^s]\|^$\|^linesearch\|^To disable\|^solver iterations"; }
echo "--- U unfused (reference) $(date +%H:%M:%S)"; run "MJW_METAL_FUSE_UPDATE=0" "ellip unfused" "--out runs/ellip_ws/check_unfused.npz"
echo "--- F fused vs unfused $(date +%H:%M:%S)"; run "MJW_METAL_FUSE_UPDATE=1" "ellip fused" "--ref runs/ellip_ws/check_unfused.npz --out runs/ellip_ws/check_fused.npz"
echo "--- H fused+htot vs unfused $(date +%H:%M:%S)"; run "MJW_METAL_FUSE_HTOT=1" "ellip fused htot" "--ref runs/ellip_ws/check_unfused.npz --out runs/ellip_ws/check_htot.npz"
echo "--- E fused+extrap vs unfused $(date +%H:%M:%S)"; run "MJW_WARMSTART_EXTRAP=1" "ellip fused extrap" "--ref runs/ellip_ws/check_unfused.npz --out runs/ellip_ws/check_extrap.npz"
echo "=== end $(date)"
