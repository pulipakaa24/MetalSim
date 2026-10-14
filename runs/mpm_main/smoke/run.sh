#!/bin/bash
# Box-on-sand smoke test of the BatchSim + MPM coupler (args passed to smoke_box.py), under the watchdog.
W=/Users/aditya/robosim/.claude/worktrees/agent-ab17e9d0efefed741
cd "$W" || exit 1
PYTHONPATH=$W exec python3 $W/scripts/watchdog.py --stall 90 --timeout 420 -- \
  /Users/aditya/robosim/.venv/bin/python "$W/research_notes/MPM in main simulator/smoke_box.py" "$@"
