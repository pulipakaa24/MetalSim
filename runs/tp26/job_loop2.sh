#!/bin/bash
# tp26 job 6b: Warp policy layer-kernel mappings (raw log kept: runs/tp26/policy_cost.log).
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
python scripts/diagnostics/warp_policy_cost.py 4096 > runs/tp26/policy_cost.log 2>&1; grep "act()\|mapping\|Error\|Traceback" runs/tp26/policy_cost.log
echo "=== end $(date) power: $(pmset -g batt | head -1)"
