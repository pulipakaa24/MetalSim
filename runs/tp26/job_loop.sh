#!/bin/bash
# tp26 job 6: the learning loop outside physics: Warp policy inference cost and bit-identical layer-kernel mappings,
# and the PPO update's time split (torch on MPS), G1 flat at 4096.
cd /Users/aditya/robosim && source .venv/bin/activate
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- warp policy cost $(date +%H:%M:%S)"; python scripts/diagnostics/warp_policy_cost.py 4096 2>&1 | grep -v "^Module\|^Warp\|^   \|^$"
echo "--- ppo update profile $(date +%H:%M:%S)"; python scripts/diagnostics/ppo_update_profile.py 4096 2>&1 | grep -v "^Module\|^Warp\|^   \|^$"
echo "=== end $(date) power: $(pmset -g batt | head -1)"
