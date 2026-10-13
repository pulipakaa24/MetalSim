# usage: source env.sh base|new|dev   (base = shared integrated trees; new = _forks/newton; dev = _forks/newton_dev)
R=/Users/aditya/robosim
WT=$R/.claude/worktrees/agent-a6ef6785c0d23ee50
T=$WT/runs/rigid_perf/tools
PY=$R/.venv-newton152/bin/python
if [ "$1" = base ]; then
  export PYTHONPATH=$R/upstream/warp-innate-integrated:$R/upstream/newton-integrated:$WT
elif [ "$1" = hang ] || [ "$1" = hangfix ]; then
  export PYTHONPATH=$WT/_forks/mjw:$R/upstream/warp-innate-integrated:$WT/_forks/newton_$1:$WT
elif [ "$1" = dev ]; then
  export PYTHONPATH=$WT/_forks/mjw:$R/upstream/warp-innate-integrated:$WT/_forks/newton_dev:$WT
else
  export PYTHONPATH=$WT/_forks/mjw:$R/upstream/warp-innate-integrated:$WT/_forks/newton:$WT
fi
export PANDA_XML=$R/upstream/mujoco_menagerie/franka_emika_panda/panda.xml
SP=$WT/metalsim/physics/mpm/probes/scale_pour.py
export WARP_CACHE_PATH=$WT/runs/rigid_perf/kcache/kc_$1
