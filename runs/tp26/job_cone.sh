#!/bin/bash
# tp26 job 10: elliptic cone term as rank-1 factor updates (worktree, MJW_ELLIPTIC_CONE_UPDATE 1 vs 0, incremental mode 2
# in both): Metal physics check (elliptic_check.sh, go2 + g1, mode 3 vs mode 2 reference) and throughput on the G1 task
# (ellip10), Go2, humanoid, SO-101 (elliptic_bench.sh, elliptic cone, interleaved 0 1 0 1).
cd /Users/aditya/robosim && source .venv/bin/activate
WTM=/Users/aditya/robosim/upstream/mujoco_warp-tp
export PYTHONPATH=/Users/aditya/robosim/upstream/warp-innate-tp
D=scripts/diagnostics/competitors
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) wt warp $(git -C upstream/warp-innate-tp rev-parse --short HEAD) mjw $(git -C $WTM rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
echo "--- check mode 2 (reference) $(date +%H:%M:%S)"; MJW_ELLIPTIC_CONE_UPDATE=0 MODELS="go2 g1" bash $D/elliptic_check.sh $WTM tp26_mode2 2>&1 | grep -v "^Module" | tail -6
echo "--- check mode 3 vs mode 2 $(date +%H:%M:%S)"; MJW_ELLIPTIC_CONE_UPDATE=1 MODELS="go2 g1" bash $D/elliptic_check.sh $WTM tp26_mode3 tp26_mode2 2>&1 | grep -v "^Module" | tail -12
for r in 0 1 0 1; do
  echo "--- bench MJW_ELLIPTIC_CONE_UPDATE=$r $(date +%H:%M:%S)"; MJW_ELLIPTIC_CONE_UPDATE=$r CONES=elliptic PROFILE=0 bash $D/elliptic_bench.sh $WTM cone_upd_$r g1task go2 humanoid so101 2>&1 | grep "LABEL\|==\|steps_per_s\|STEP\|Error\|Traceback" | cut -c1-200
done
echo "--- profile g1task mode 3 $(date +%H:%M:%S)"; MJW_ELLIPTIC_CONE_UPDATE=1 CONES=elliptic BENCH=0 PROFILE=1 bash $D/elliptic_bench.sh $WTM cone_upd_prof g1task 2>&1 | grep -v "^Module" | tail -14 | cut -c1-160
echo "=== end $(date) power: $(pmset -g batt | head -1)"
