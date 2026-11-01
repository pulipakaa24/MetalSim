#!/bin/bash
# run_il3fix.sh: copy of run_all.sh for rough_il3 only (FROM=il3fix_rough_s0), on MuJoCo Warp 787a5b3a (EPA horizon +
# heightfield margin-shrink fix, handoff/hfield-epa.md Next 2) instead of 93e5ce18; started by il3_handover.sh.
# Rough-terrain re-runs required by the heightfield contact fix (MuJoCo Warp fork 93e5ce1, branch metalsim):
# docs/research/hfield_contacts_2026-10-31.md §9, items 1-4. Hand-off: runs/readme_media/handoff/rough-reruns.md.
# Each job runs through scripts/gpu_run.sh (the GPU queue) and job.sh (provenance header, then exec); one seed per job, so
# waiting short jobs are granted in between. A failed job is recorded in status.jsonl and the pipeline goes on.
#
#   nohup ... run_all.sh                 full pipeline (outputs here)
#   SMOKE=1 run_all.sh                   tiny version of every job type, plus 4096-env sizing runs (outputs in smoke/)
#   FROM=<job> / SKIP="<job> <job>"      start at a job / skip jobs (restarts)
# Stop safely: stop.sh (touches STOP so no new job starts, terminates the running job through its gpu_run.sh, which
# kills the job and releases the GPU lock, then the driver).
#
# Original commands (exact sources in the hand-off): 2.3.2 rough seeds = runs/g1_rough_ppowarp_fixed.sh (seed 0) and the
# seed 1 / 2 log headers; rough_il3 = runs/il3/run_train3.sh rough rough_il3 recommended isaaclab3_every_substep_cap20 1500 0
# il3fix_rough_s0. Flags below map the settings of those dates onto today's CLI (options added later are set to the
# behaviour the original code had; git commits in the hand-off).
set -u
unset PYTHONPATH            # jobs choose their MuJoCo Warp copy explicitly (gpu_job PYTHONPATH argument)
R=~/MetalSim; cd "$R" || exit 1
RR=runs/rough_rerun_2026-10-31
SMOKE=${SMOKE:-0}; FROM=${FROM:-}; SKIP=${SKIP:-}
if [ "$SMOKE" = 1 ]; then O=$RR/smoke; P=rr_smoke_; else O=$RR; P=rr_; fi
L=$O/logs; STATUS=$O/status.jsonl; DLOG=$O/driver.log
PY=.venv/bin/python
mkdir -p $L $O/ckpt $O/transfer $O/isaac_export/metalsim_ckpts_rough
echo $$ > $O/driver.pid
started=0; [ -z "$FROM" ] && started=1

ts() { date '+%Y-%m-%dT%H:%M:%S'; }
power() { pmset -g batt | head -1 | sed -e "s/^Now drawing from '//" -e "s/'.*$//"; }
dlog() { echo "$(ts) $*" >> $DLOG; }
status() {   # job state kind log queued end rc
  python3 - "$@" >> $STATUS <<'EOF'
import json, os, sys
job, state, kind, log, queued, end, rc = sys.argv[1:8]
d = {"job": job, "state": state, "kind": kind, "log": log, "queued": queued or None, "start": None, "end": end or None,
     "rc": int(rc) if rc not in ("", "null") else None}
m = log + ".meta.json"
if state != "queued" and os.path.exists(m):
    try: d.update(json.load(open(m)))
    except Exception: pass
print(json.dumps(d))
EOF
}
want() {     # FROM / SKIP / STOP handling
  local n=$1
  [ "$started" = 0 ] && [ "$n" = "$FROM" ] && started=1
  [ "$started" = 0 ] && { dlog "$n: before FROM=$FROM, skipped"; return 1; }
  case " $SKIP " in *" $n "*) dlog "$n: in SKIP, skipped"; status $n skipped - - "" "$(ts)" null; return 1;; esac
  [ -e $O/STOP ] && { dlog "$n: STOP file present, not started"; status $n skipped - - "" "$(ts)" null; return 1; }
  return 0
}
gpu_job() {  # NAME KIND MINUTES PYTHONPATH EXPECT_MJW -- command...   (PYTHONPATH "-" = the installed checkout)
  local n=$1 kind=$2 min=$3 pp=$4 exp=$5; shift 5; [ "$1" = "--" ] && shift
  want $n || return 0
  local log=$L/$n.out q; q=$(ts)
  [ -e $log ] && mv $log $log.prev
  status $n queued $kind $log "$q" "" null
  dlog "$n: queued ($kind, $min min, power $(power)) -> $log"
  [ "$pp" = - ] && pp=""
  scripts/gpu_run.sh $P$n $kind $min -- /usr/bin/env RR_PYTHONPATH="$pp" RR_EXPECT_MJW="$exp" $RR/job.sh $P$n $log -- "$@" >> $DLOG 2>&1
  local rc=$?
  status $n $([ $rc = 0 ] && echo done || echo failed) $kind $log "$q" "$(ts)" $rc
  dlog "$n: finished rc $rc (power $(power))"
}
rotate() { for f in "$@"; do [ -e "$f" ] && mv "$f" "$f.prev"; done; return 0; }

# --- settings -----------------------------------------------------------------------------------------------------------
# 2.3.2 rough preset (reward_cfg rough_isaac = Isaac Lab 2.3.2 G1RoughEnvCfg), as on 2026-09-25 03:11-14:49: no contact
# tuning (contact_cfg from b1473429 14:52; "default" = MuJoCo's), base and feet-slide velocity at the frame origin
# (12b4e33b 13:56, 1209b4e4 19:42), PPOWarp without the initial reset (7cf968ae 09-26 04:13), effort limit on the
# actuator (18180101 09-26 05:07). Seed 0 (03:11) predates the ray-order fix (e88db43f 07:36: "ij") and the box terrain
# + exact scan (4b09b044 09:07); seeds 1 and 2 log "height-scan ray order xy, terrain collision boxes_local, height scan exact".
# DEVIATION: --initial_reset 1 (the originals started every env's first episode from qpos0, the start-state defect fixed in
# 7cf968ae, PARITY §1.5 note). On the fixed fork the qpos0 start overflows njmax 256 on boxes_local (smoke: NEFC in 8 of 4096
# worlds at iteration 1; pinned fork with the same start: none, smoke/exp/), which the latched guard (870d0d7f) aborts.
C232="--reward_cfg rough_isaac --contact_cfg default --base_velocity origin --feet_slide_velocity origin --initial_reset 1 --effort_limit actuator"
SURF_S0="--scan_ordering ij --terrain_collision hfield --scan_surface grid"
SURF_S12="--scan_ordering xy --terrain_collision boxes_local --scan_surface exact"
# rough_il3 (il3fix_rough_s0, 2026-09-26 03:00): header "rough_il3, height-scan ray order xy, terrain collision hfield, height
# scan exact, solver_cfg isaaclab3_every_substep_cap20, contact_cfg recommended, base velocity com, feet-slide velocity com";
# "recommended" was then tau10_impact_hardlimits = today's "recommended_pyramidal" (1a1cf0db 09-26 11:00); no initial reset,
# actuator effort limit (both later). The isaaclab3 preset owns cone / contact / limit fields either way (solver_presets.py:85-151).
# DEVIATION: --initial_reset 1 here too (consistency; the original's early transient, 823 of its 1,010 blow-ups at iterations
# 10-14, was that qpos0 start: isaaclab3 note §6.7).
CIL3="--reward_cfg rough_il3 --contact_cfg recommended_pyramidal --solver_cfg isaaclab3_every_substep_cap20 --base_velocity com --feet_slide_velocity com --initial_reset 1 --effort_limit actuator"
OLDCK="metalsim_ours_it1500=$RR/ckpt_old/g1_rough_ppowarp_fixed_it1500.pt nvidia_pretrained_rough_it2999=assets/isaac/pretrained/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt"
PIN=$RR/code/mjw_8eb51fe; PINC=8eb51fe6     # MuJoCo Warp fork before the fix (git archive), for the transfer A/B
FIX=787a5b3a; FIXPP=$RR/code/mjw_787a5b3   # EPA-fixed fork tip after review fixes (handoff/hfield-epa.md, epa-fixes.md), git archive with COMMIT file
if [ "$SMOKE" = 1 ]; then N=64; IT=3; TMODE=smoke; TMIN=10; else N=4096; IT=1500; TMODE=full; TMIN=110; fi

train() {    # NAME SEED SURFACE-FLAGS CONFIG-FLAGS   -> $O/logs/NAME.log (the training log), $O/ckpt/NAME.pt (+ _it<k>.pt every 100)
  local n=$1 s=$2 surf=$3 cfg=$4 nn=${5:-$N} it=${6:-$IT} ck=${7:-yes}
  local tl=$L/$n.log c=$O/ckpt/$n.pt; [ "$ck" = no ] && c=-
  want $n || return 0
  rotate $tl $tl.anomalies.jsonl
  # train_wrap.py = python -m metalsim.learn.g1_velocity with EPA_HORIZON reported but not fatal (see its docstring)
  started=1; gpu_job $n train $TMIN $FIXPP $FIX -- $PY $RR/train_wrap.py $nn rough train $it $tl $c 0.0025 --seed $s $surf $cfg
}
export_ck() {  # NAME TAG SEED: our checkpoints of training NAME -> isaac_export/metalsim_ckpts_rough/rerun_<TAG>_it<k>.pt
  local n=$1 tag=$2 pairs=()
  if [ "$SMOKE" = 1 ]; then pairs=("$O/ckpt/$n.pt=$O/isaac_export/metalsim_ckpts_rough/rerun_${tag}_final.pt")
  else for k in 500 1000 1500; do pairs+=("$O/ckpt/${n}_it$k.pt=$O/isaac_export/metalsim_ckpts_rough/rerun_${tag}_it$k.pt"); done; fi
  gpu_job export_$tag render 3 - $FIX -- $RR/jobs/export.sh "${pairs[@]}"
}

# --- pipeline -----------------------------------------------------------------------------------------------------------
dlog "=== pipeline start (SMOKE=$SMOKE FROM=$FROM SKIP=$SKIP) pid $$ power $(power); metalsim $(git rev-parse --short HEAD), mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD), warp $(git -C upstream/warp-metalsim rev-parse --short HEAD)"
# 1. tests/test_terrain.py on the new pin (+ the training gates)
gpu_job tests render 15 - $FIX -- $RR/jobs/tests.sh $TMODE
# 2. transfer into MetalSim: Isaac's L4 rough checkpoints are missing on this Mac (hand-off); our seed-0 it1500 (the only old
#    rough checkpoint left) and NVIDIA's pretrained Isaac rough policy, on the fixed fork and on the pinned fork (A/B)
gpu_job transfer_old_fix render 20 - $FIX -- $RR/jobs/transfer.sh $O/transfer old_fix $TMODE $OLDCK
gpu_job transfer_old_pin render 20 $PIN $PINC -- $RR/jobs/transfer.sh $O/transfer old_pin $TMODE $OLDCK
# 3. rough training, 2.3.2 preset, three seeds x 1500 (one seed per job); exports for Isaac; seed 0's checkpoints through
#    the transfer protocols (the original "ours" rows were seed 0's it500 / it1000 / it1500)
train g1_rough_ppowarp_fixed_s0 0 "$SURF_S0" "$C232"
export_ck g1_rough_ppowarp_fixed_s0 s0
if [ "$SMOKE" = 1 ]; then NEWCK="metalsim_rerun_s0_final=$O/ckpt/g1_rough_ppowarp_fixed_s0.pt"
else NEWCK=""; for k in 500 1000 1500; do NEWCK="$NEWCK metalsim_rerun_s0_it$k=$O/ckpt/g1_rough_ppowarp_fixed_s0_it$k.pt"; done; fi
gpu_job transfer_new_s0 render 20 - $FIX -- $RR/jobs/transfer.sh $O/transfer new_s0 $TMODE $NEWCK
train g1_rough_ppowarp_fixed_s1 1 "$SURF_S12" "$C232"
export_ck g1_rough_ppowarp_fixed_s1 s1
if [ "$SMOKE" != 1 ]; then
  train g1_rough_ppowarp_fixed_s2 2 "$SURF_S12" "$C232"
  export_ck g1_rough_ppowarp_fixed_s2 s2
fi
# 4. Isaac Lab 3.0 like-for-like rough run (rough_il3, seed 0)
train il3fix_rough_s0 0 "" "$CIL3"
# smoke only: 4096-env sizing runs (memory at full size, early per-iteration rate for the duration estimates)
if [ "$SMOKE" = 1 ]; then
  train size4096_s0cfg 0 "$SURF_S0" "$C232" 4096 8 no
  train size4096_s1cfg 1 "$SURF_S12" "$C232" 4096 8 no
  train size4096_il3 0 "" "$CIL3" 4096 8 no
fi
dlog "=== pipeline finished, power $(power)"
status _pipeline finished - - "" "$(ts)" null
rm -f $O/driver.pid
