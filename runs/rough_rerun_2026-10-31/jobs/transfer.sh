#!/bin/bash
# transfer.sh OUTDIR TAG MODE label=checkpoint ...   (MODE full | smoke)
# PARITY §1.5 rough-transfer protocols, MetalSim side (scripts/diagnostics/g1_rough_transfer.py), under the task settings
# those records were measured with (MuJoCo-default contacts, actuator effort limit: task_defaults.py explains why):
#   A protocol24: levels 3 and 6, 4 envs in columns 0/4/9/14 (pyramid stairs, inverted stairs, boxes, random rough), cell
#                 origin, command 0.5 m/s, 400 steps, mean action; terrain_collision hfield + grid scan, the setting of
#                 runs/g1_rough_transfer.jsonl (e88db43f, 2026-09-25 07:36; "Isaac's checkpoints 7 / 24 in MetalSim")
#   B reps8:      the same with --reps 8 (rep 0 = protocol start, 7 starts jittered by up to +-5 cm), boxes_local + exact
#                 scan, the setting of runs/terrain_walls/transfer_reps.sh / transfer_reps.jsonl ("14 / 96" wall cells,
#                 "ours 1 / 96 -> 4 / 96"); runs/terrain_walls/transfer_reps.sh passed exactly these flags
# fall = pelvis < 0.3 m above the local ground at step 400 (g1_rough_transfer.py FALL_HAG). MODE smoke: level 3 only; B
# with --reps 2 --steps 40. Outputs: OUTDIR/TAG_protocol24.jsonl, OUTDIR/TAG_reps8.jsonl (earlier files moved to *.prev).
set -u
trap 'kill -TERM $P 2>/dev/null; wait $P 2>/dev/null; exit 143' TERM INT HUP
OUT=$1; TAG=$2; MODE=$3; shift 3
CK=(); for s in "$@"; do CK+=(--ckpt "$s"); done
export RR_TASK_DEFAULTS='{"contact_cfg": "default", "effort_limit": "actuator"}'
PY=.venv/bin/python; T=runs/rough_rerun_2026-10-31/task_defaults.py; X=scripts/diagnostics/g1_rough_transfer.py
IP=runs/parity/isaac/rough/play_rough
LV="3 6"; RB="--reps 8"; [ "$MODE" = smoke ] && { LV=3; RB="--reps 2 --steps 40"; }
mkdir -p "$OUT"; RC=0
for f in "$OUT/${TAG}_protocol24.jsonl" "$OUT/${TAG}_reps8.jsonl"; do [ -e "$f" ] && mv "$f" "$f.prev"; done
run() { echo "--- $(date +%T) $*"; "$@" & P=$!; wait $P; local rc=$?; echo "--- rc $rc"; [ $rc -ne 0 ] && RC=$rc; }
run $PY $T $X --isaac_play $IP --level $LV "${CK[@]}" --terrain_collision hfield --scan_surface grid --out "$OUT/${TAG}_protocol24.jsonl"
run $PY $T $X --isaac_play $IP --level $LV $RB "${CK[@]}" --terrain_collision boxes_local --scan_surface exact --out "$OUT/${TAG}_reps8.jsonl"
exit $RC
