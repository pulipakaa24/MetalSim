#!/bin/bash
# ellip_ws: the full A/B of runs/tp26/job_ellip.sh on the frozen worktree mujoco_warp-ellip-test2 (a00c21f: elliptic fused
# per-iteration launch with the htot fusion, default knobs) / warp-innate-tp-test (9abceff9): g1_tp_variants.py 4096 (physics
# only, env step, rollout + inference, PPO update, loop), recommended_pyramidal (cap 10) vs recommended (elliptic ellip10, cap 20),
# interleaved P E P E.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-tp-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test2
F='^\[\|Error\|Traceback\|imports'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-tp-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test2 rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
run() { PYTHONPATH=$WT MJW_TP_VARIANT="$2" python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__); import runpy, sys; sys.argv=['g1_tp_variants.py','4096']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/ellip_ws/ab_$1.log 2>&1; grep "$F" runs/ellip_ws/ab_$1.log; }
for r in 1 2; do
  echo "--- P$r pyramidal cap 10 $(date +%H:%M:%S)"; run P$r '{"contact_cfg":"recommended_pyramidal","label":"P'$r'_pyr_cap10"}'
  echo "--- E$r elliptic ellip10 cap 20 $(date +%H:%M:%S)"; run E$r '{"contact_cfg":"recommended","label":"E'$r'_ellip10_cap20"}'
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
