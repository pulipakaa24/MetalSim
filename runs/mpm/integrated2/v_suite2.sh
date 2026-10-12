#!/bin/bash
# Final-state verification (each job through the GPU queue): recordings integrated (I) / final (N) and a second N run of
# the coupled scenes, Warp Metal test groups, tests/test_mpm.py, rigid repeat checks, rigid suite.
A=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/runs/a2
W=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/upstream
R=/Users/aditya/robosim
G=$R/scripts/gpu_run.sh
O=$A/out; mkdir -p $O $A/kc
export PYTHONDONTWRITEBYTECODE=1
# rigid suite with the final Warp and the fixed Newton 1.7
SR=$A/suite; mkdir -p $SR
$G a2v_suite2 check 60 -- env PYTHONPATH=$W/warp-i2:$W/newton-rigid:$R:$R/runs/fma_rigid_parity/harness WARP_CACHE_PATH=$A/kc/suite \
  CAPTURE_OUT=$O/suite_final.jsonl bash -c "cd $R && $R/.venv/bin/python -m pytest -p capture_locals -q -rA --ignore=tests/test_mpm.py --ignore=tests/test_gpu_queue.py -p no:cacheprovider tests" > $O/suite_final.log 2>&1
echo "suite: $(tail -1 $O/suite_final.log)"; /usr/bin/grep "^FAILED" $O/suite_final.log
echo "end $(date +%H:%M:%S)"
