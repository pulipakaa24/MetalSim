#!/bin/bash
W=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
P=$W/metalsim/physics/mpm/probes
O=$W/runs/mpm/approx
PY=/Users/aditya/robosim/.venv-newton152/bin/python
export PYTHONPATH=$W/_forks/warp
export WARP_CACHE_PATH=$O/kcache
$PY $P/metal_approx_probe.py --json $O/probe.json "$@" || exit 1
$PY $P/metal_compose_check.py --json $O/compose_metal.json
