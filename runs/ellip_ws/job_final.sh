#!/bin/bash
# ellip_ws: final A/B (runs/tp26/job_ellip.sh protocol), frozen worktrees warp-innate-ellip-test (6bceb39d) / mujoco_warp-ellip-test6
# (433c305: elliptic fused launch with htot fusion at 128 lanes, ICB early exit; line-search floor off by default):
# g1_tp_variants.py 4096, recommended_pyramidal cap 10 (P) vs recommended = elliptic ellip10 cap 20 without (E0) and with the
# line-search floor at 8 eps (E8), interleaved P E0 E8 P E0 E8.
cd /Users/aditya/robosim && source .venv/bin/activate
WT=/Users/aditya/robosim/upstream/warp-innate-ellip-test:/Users/aditya/robosim/upstream/mujoco_warp-ellip-test6
F='^\[\|Error\|Traceback'
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) frozen warp $(git -C upstream/warp-innate-ellip-test rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp-ellip-test6 rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
run() { PYTHONPATH=$WT env $3 MJW_TP_VARIANT="$2" python -c "import runpy, sys; sys.argv=['g1_tp_variants.py','4096']; runpy.run_path('scripts/diagnostics/g1_tp_variants.py', run_name='__main__')" > runs/ellip_ws/final_$1.log 2>&1; grep "$F" runs/ellip_ws/final_$1.log; }
for r in 1 2; do
  echo "--- P$r pyramidal cap 10 $(date +%H:%M:%S)"; run P$r '{"contact_cfg":"recommended_pyramidal","label":"P'$r'_pyr_cap10"}' "MJW_LS_NOISE_FLOOR=0"
  echo "--- E0_$r elliptic ellip10 cap 20 $(date +%H:%M:%S)"; run E0_$r '{"contact_cfg":"recommended","label":"E0_'$r'_ellip10_cap20"}' "MJW_LS_NOISE_FLOOR=0"
  echo "--- E8_$r elliptic ellip10 cap 20, LS floor 8 $(date +%H:%M:%S)"; run E8_$r '{"contact_cfg":"recommended","label":"E8_'$r'_ellip10_cap20_ls8"}' "MJW_LS_NOISE_FLOOR=8"
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
