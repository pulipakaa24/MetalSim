#!/bin/bash
# the GPU tests on the committed tree (clean export) + the BoxWindow fix, MuJoCo Warp fork a8e6485 (clean export)
cd /private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/head_tree
.venv/bin/python scripts/gpu_lock.py status
export PYTHONPATH=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/head_tree:/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/mjw_a8e6485
.venv/bin/python -c "import metalsim, mujoco_warp; print(metalsim.__file__, mujoco_warp.__file__)"
.venv/bin/python -m pytest tests/test_terrain.py tests/test_g1_task_terms.py -q -p no:cacheprovider 2>&1 | grep -E "passed|failed|^E  |FAILED" | head -40
