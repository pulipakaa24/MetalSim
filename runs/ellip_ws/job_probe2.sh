#!/bin/bash
# ellip_ws: cap / warm-start probe for the line-search noise floor 8 eps (cap_probe_variant.py, cap_probe_ellip.py's protocol), frozen
# worktree, flat terrain, the task default (elliptic ellip10, cap 20): states from the elliptic-trained policy
# (runs/il3/ckpt/g1_flat_flatcfg_ellip10_it1000.pt) and from random actions (robots fall and lie), 4096 envs.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-ellip-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test5
echo "=== start $(date) frozen warp $(git -C upstream/warp-innate-ellip-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test5 rev-parse --short HEAD)"
run() { PYTHONPATH=$WT VARIANT=lsfloor LS_FLOOR_EPS=8 python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['cap_probe_variant.py','flat','recommended','$1','runs/ellip_ws/probe_lsfloor8_$2.json']; runpy.run_path('scripts/diagnostics/competitors/cap_probe_variant.py', run_name='__main__')" > runs/ellip_ws/probe_lsfloor8_$2.log 2>&1; grep -v '^{"step"\|^linesearch\|^To disable\|^solver iterations\|^Module\|^Warp\|^   ' runs/ellip_ws/probe_lsfloor8_$2.log | tail -60; }
echo "--- policy states $(date +%H:%M:%S)"; run runs/il3/ckpt/g1_flat_flatcfg_ellip10_it1000.pt policy
echo "--- random actions $(date +%H:%M:%S)"; run random random
echo "=== end $(date)"
