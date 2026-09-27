#!/bin/bash
# speculative contacts, correctness (c): the fork's suite on Metal from the frozen prototype worktree e636feb, without the
# prototype switch, with MJW_SPECULATIVE_GAP=1 (gap-free test models: the rule must be inert), and with the final rule's switches
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate:/Users/aditya/robosim/upstream/mujoco_warp-spec-e636feb
echo "=== start $(date) warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-spec-e636feb rev-parse --short HEAD)"
cd upstream/mujoco_warp-spec-e636feb
F='^linesearch\|^To disable\|^solver iterations\|^nvmax'
echo "--- no switch $(date +%H:%M:%S)"; PYTHONPATH=$WT python -m pytest mujoco_warp/_src -q -p no:cacheprovider -rf 2>&1 | grep -v "$F" | grep "^FAILED\|passed\|failed"
echo "--- MJW_SPECULATIVE_GAP=1 $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_SPECULATIVE_GAP=1 python -m pytest mujoco_warp/_src -q -p no:cacheprovider -rf 2>&1 | grep -v "$F" | grep "^FAILED\|passed\|failed"
echo "--- MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=live MJW_SPEC_FRICIMP=d0 $(date +%H:%M:%S)"; PYTHONPATH=$WT MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=live MJW_SPEC_FRICIMP=d0 python -m pytest mujoco_warp/_src -q -p no:cacheprovider -rf 2>&1 | grep -v "$F" | grep "^FAILED\|passed\|failed"
echo "=== end $(date)"
