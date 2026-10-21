# usage: source env.sh TREE   -- Newton 1.7-line trees for the rigid-scale port (item 1 of the 2026-10-11 loose ends).
# TREE = a directory name under $WT/upstream/ holding a Newton checkout (e.g. n17_base, n17_fix, n17_perf).
# Python: .venv-newtonfork (Newton 1.7 deps, MuJoCo Warp fork 3.14, the shared Warp fork checkout); the tree's Newton
# shadows the installed one through PYTHONPATH.
R=.
WT=$R/.claude/worktrees/agent-a7a2cb997dc44ebcc
PY=$R/.venv-newtonfork/bin/python
export PYTHONPATH=$WT/upstream/t4/warp-B:$WT/upstream/$1:$WT     # Warp: the pinned fork commit 152808b7, built (t4/warp-B)
export PANDA_XML=$R/upstream/mujoco_menagerie/franka_emika_panda/panda.xml
SP=$WT/runs/rigid17/tools/collide17.py
export WARP_CACHE_PATH=$WT/runs/rigid17/kcache/kcB_$1
