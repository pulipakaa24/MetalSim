#!/bin/bash
# PARITY §1.4 rough throughput rows after the heightfield contact fix: interleaved A/B, pinned fork 8eb51fe (meshes on the
# plane path) vs the fix 93e5ce1 (meshes on GJK/EPA). RUN ON THE 16-INCH (the timing machine), through the timing queue,
# from the repo root of a checkout at the bumped pin (upstream/mujoco_warp 93e5ce1, upstream/warp-metalsim 52c09cdd):
#
#   git -C upstream/mujoco_warp archive 8eb51fe | (mkdir -p /tmp/mjw_8eb51fe && tar -x -C /tmp/mjw_8eb51fe)
#   scripts/gpu_run.sh rr_timing_rough timing 15 -- bash runs/rough_rerun_2026-10-31/timing16/rough_timing.sh default
#   scripts/gpu_run.sh rr_timing_rough_hfield timing 15 -- bash runs/rough_rerun_2026-10-31/timing16/rough_timing.sh hfield
#
# MODE default: the task's defaults today (boxes_local collision + exact scan, contact_cfg "recommended" = elliptic): the
#   §1.4 "step only" command (python -m metalsim.learn.g1_velocity N rough, 4096 and 2048) and the cost split of
#   runs/tp26/job_rough2.sh (scripts/diagnostics/g1_tp_variants.py: physics only / full env step / rollout + inference /
#   full PPO loop; the README's rough loop number).
# MODE hfield: the §1.4 rows' own setting (runs/bench12.log, 2026-09-24: heightfield collision on every cell, grid scan,
#   MuJoCo-default contacts, pyramidal cone, actuator effort limit): step only via step_only.py, cost split via
#   g1_tp_variants.py under task_defaults.py.
# Output: runs/rough_rerun_2026-10-31/timing16/<mode>_<host>_<time>.log (power and device utilisation at both ends).
set -u
MODE=${1:-default}; BASE=${BASE:-/tmp/mjw_8eb51fe}; RRD=runs/rough_rerun_2026-10-31
PY=.venv/bin/python; OUT=$RRD/timing16/${MODE}_$(scutil --get LocalHostName 2>/dev/null || hostname)_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$OUT") 2>&1
pw() { echo "power: $(pmset -g batt | head -1 | sed "s/^Now drawing from //") | $(pmset -g | grep powermode | tr -s ' ') | util $(scripts/gpu_util.sh)% | $(date +%T)"; }
echo "=== $MODE start $(date) on $(sysctl -n hw.model); metalsim $(git rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) warp $(git -C upstream/warp-metalsim rev-parse --short HEAD)"; pw
[ -d "$BASE/mujoco_warp" ] || { echo "missing pinned copy $BASE (see header)"; exit 2; }
which_mjw() { PYTHONPATH=$1 $PY -c "import mujoco_warp; from mujoco_warp._src import collision_convex as c; print('mujoco_warp', mujoco_warp.__file__, 'HFIELD_PLANE_CONTACTS', c.HFIELD_PLANE_CONTACTS)"; }
H='{"terrain_collision": "hfield", "scan_surface": "grid", "contact_cfg": "default", "effort_limit": "actuator"}'
for rep in 1 2; do for t in base fix; do
  W=""; [ $t = base ] && W=$BASE
  echo "--- $t rep $rep"; pw; which_mjw "$W"
  if [ "$MODE" = default ]; then
    for n in 4096 2048; do PYTHONPATH=$W $PY -m metalsim.learn.g1_velocity $n rough 2>&1 | grep -E "^G1|^N=|Error|Traceback"; done
    PYTHONPATH=$W MJW_TP_TERRAIN=rough MJW_TP_VARIANT="{\"contact_cfg\":\"recommended\",\"label\":\"rough_${t}${rep}\"}" \
      $PY scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep -E "^\[|Error|Traceback"
  else
    for n in 4096 2048; do PYTHONPATH=$W RR_TASK_SETTINGS="$H" $PY $RRD/timing16/step_only.py $n 2>&1 | grep -E "^G1|^N=|Error|Traceback"; done
    PYTHONPATH=$W RR_TASK_DEFAULTS="$H" MJW_TP_TERRAIN=rough MJW_TP_VARIANT="{\"label\":\"rough_hfield_${t}${rep}\"}" \
      $PY $RRD/task_defaults.py scripts/diagnostics/g1_tp_variants.py 4096 2>&1 | grep -E "^\[|Error|Traceback"
  fi
done; done
echo "=== end $(date)"; pw
