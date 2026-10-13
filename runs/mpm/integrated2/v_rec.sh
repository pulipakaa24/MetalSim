#!/bin/bash
# Final-state verification (each job through the GPU queue): recordings integrated (I) / final (N) and a second N run of
# the coupled scenes, Warp Metal test groups, tests/test_mpm.py, rigid repeat checks, rigid suite.
A=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/runs/a2
W=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/upstream
R=/Users/aditya/robosim
G=$R/scripts/gpu_run.sh
O=$A/out; mkdir -p $O $A/kc
export PYTHONDONTWRITEBYTECODE=1
bash $A/rec.sh I $R/upstream/warp-innate-integrated $R/upstream/newton-integrated
bash $A/rec.sh N $W/warp-i2 $W/newton-i2
$R/.venv/bin/python $A/py/bitcmp.py $A/rec/I $A/rec/N > $O/bitwise_I_vs_N.txt; cat $O/bitwise_I_vs_N.txt
# coupled scenes a second time (run to run)
bash $A/rec.sh N2 $W/warp-i2 $W/newton-i2
$R/.venv/bin/python $A/py/bitcmp.py $A/rec/N $A/rec/N2 > $O/bitwise_N_vs_N2.txt; cat $O/bitwise_N_vs_N2.txt
echo "end $(date +%H:%M:%S)"
