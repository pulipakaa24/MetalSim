#!/bin/bash
# Item-4: A/B again (2 reps, interleaved) bracketed by a GPU-side probe (gs_pass.py: us per Gauss-Seidel color pass
# in graph replay, sustained clock) on B, to tell a GPU-side slowdown from a host-side one.
H=/Users/aditya/robosim/.claude/worktrees/agent-a7a2cb997dc44ebcc
D=$H/runs/mpm/regress_check; T=$H/upstream/t4; O=$D/out3; mkdir -p $O
PY=/Users/aditya/robosim/.venv-newton152/bin/python
export PYTHONDONTWRITEBYTECODE=1
gs() { (cd $H && WARP_CACHE_PATH=$D/kc_B PYTHONPATH=$T/warp-B:$T/newton-B:$H $PY metalsim/physics/mpm/probes/gs_pass.py \
   example:mpm_granular --seconds 2 --json $O/gs_$1.json > $O/gs_$1.log 2>&1); echo "gs_pass $1: $(tail -1 $O/gs_$1.log | cut -c1-300) $(date +%H:%M:%S)"; }
gs pre
OUT=$O $D/ab.sh run 2 2>&1 | sed 's/^/ab /'
gs post
