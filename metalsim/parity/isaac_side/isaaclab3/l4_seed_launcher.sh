#!/bin/bash
# Standalone (session-independent) start-retry + launch + fetch for the Franka-Pour Isaac seeds on an existing L4 VM.
#   l4_seed_launcher.sh VM ZONE SEEDS VERIFY [FALLBACK_SEEDS FALLBACK_PEER]
#     VERIFY=1: run fingerprint.sh on the VM and launch only on an exact match with the seed-0 reference (below).
#     FALLBACK_SEEDS / FALLBACK_PEER: after SEEDS finish, if the launcher of VM FALLBACK_PEER never got its VM up
#     (status RETRYING or FAILED), abort that launcher and run FALLBACK_SEEDS here too.
# Never creates, resizes, moves or deletes instances: only `instances start` of the named VM in its own zone, and
# `instances stop` of it after a failed check. Start attempts every 5 min, for at most 12 h from launch.
# State: $D/status_<VM>.txt (one line: STATE | time | details), log $D/launcher_<VM>.log, abort flag $D/ABORT_<VM>.
# Run detached: python3 -c 'import os,sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' <this> ARGS... &
VM=$1; ZONE=$2; SEEDS=$3; VERIFY=$4; FB_SEEDS=${5:-}; FB_PEER=${6:-}
D=${L4_SEEDS_DIR:-$PWD/runs/parity/isaaclab3_pour/l4_seeds}   # run from the repo root, or set L4_SEEDS_DIR
S=$D/scripts                       # snapshot of the scripts staged to the VM (copied in by the caller)
G="--zone $ZONE --project mcmc-193822 --quiet"
SSH="gcloud compute ssh $VM $G --tunnel-through-iap"
STATUS=$D/status_$VM.txt; LOG=$D/launcher_$VM.log
PER_SEED_H=5.3                      # seed 0 on the L4: 314 s per iteration x 60 + start-up
REF_DRIVER=580.178.04; REF_HEAD=ae37b028; REF_RSL=5.4.1; REF_MTP="max_triangle_pairs: int = 1_000_000"
ts() { date '+%F %T %Z'; }
st() { echo "$1 | $(ts) | $VM ($ZONE) seeds [$SEEDS] | $2" > "$STATUS"; echo "$(ts) $1: $2" >> "$LOG"; }
mkdir -p "$D"; echo "$(ts) launcher pid $$ args: $*" >> "$LOG"
deadline=$(( $(date +%s) + 12*3600 )); i=0; nonstock=0

# 1. start retries
while :; do
  [ -e "$D/ABORT_$VM" ] && { st FAILED "aborted ($(cat "$D/ABORT_$VM"))"; exit 0; }
  i=$((i+1))
  out=$(gcloud compute instances start $VM $G 2>&1); rc=$?
  if [ $rc -eq 0 ]; then echo "$(ts) try $i: STARTED" >> "$LOG"; break; fi
  if echo "$out" | grep -qiE "reauth|gcloud auth login|credential|unauthenticated"; then
    st FAILED "gcloud auth expired at try $i, user must re-run gcloud auth login: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"; exit 3
  fi
  if echo "$out" | grep -q STOCKOUT; then r=STOCKOUT; nonstock=0
  else r="error: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"; nonstock=$((nonstock+1)); fi
  echo "$(ts) try $i: $r" >> "$LOG"
  [ $nonstock -ge 6 ] && { st FAILED "6 non-stockout start errors in a row, last: $r"; exit 2; }
  [ $(( $(date +%s) + 300 )) -gt $deadline ] && { st FAILED "VM never started: $i tries over 12 h (last: $r)"; exit 2; }
  st RETRYING "try $i failed ($r); next in 5 min; gives up at $(date -r $deadline '+%F %T')"
  sleep 300
done
if [ -e "$D/ABORT_$VM" ]; then gcloud compute instances stop $VM $G >> "$LOG" 2>&1; st FAILED "started but aborted ($(cat "$D/ABORT_$VM")); VM stopped"; exit 0; fi
st BOOTING "VM started after $i tries; waiting for ssh"
for j in $(seq 1 30); do $SSH --command true < /dev/null > /dev/null 2>&1 && break; sleep 30; done

# 2. fingerprint (always recorded; gating when VERIFY=1)
$SSH --command 'bash -s' < $S/fingerprint.sh > $D/fingerprint_$VM.txt 2>> "$LOG"
sed -i '' "s#/home/[A-Za-z0-9._-]*/#/home/<vm-user>/#g" $D/fingerprint_$VM.txt
if [ "$VERIFY" = 1 ]; then
  fp() { grep "^$1: " $D/fingerprint_$VM.txt | head -1 | cut -d' ' -f2-; }
  diff=""
  [ "$(fp FP_DRIVER)" = "$REF_DRIVER" ] || diff="$diff driver '$(fp FP_DRIVER)' != $REF_DRIVER;"
  case "$(fp FP_ISAACLAB_HEAD)" in $REF_HEAD*) ;; *) diff="$diff IsaacLab3 HEAD '$(fp FP_ISAACLAB_HEAD)' != $REF_HEAD;";; esac
  [ "$(fp FP_ISAACLAB_DIRTY)" = 0 ] || diff="$diff IsaacLab3 tree not clean ($(fp FP_ISAACLAB_DIRTY) entries);"
  [ "$(fp FP_RSL_RL)" = "$REF_RSL" ] || diff="$diff rsl_rl '$(fp FP_RSL_RL)' != $REF_RSL;"
  [ "$(fp FP_OMNICLIENT_7151)" -ge 1 ] 2>/dev/null || diff="$diff Kit/omni client 2.72.3-release.7151 not found;"
  [ "$(fp FP_MTP_CFG)" = "$REF_MTP" ] || diff="$diff '$(fp FP_MTP_CFG)' != '$REF_MTP';"
  [ "$(fp FP_MTP_POUR_OVERRIDES)" = 0 ] || diff="$diff franka_pour overrides max_triangle_pairs;"
  if [ -n "$diff" ]; then
    # leave the VM up for a human (a restart may hit a stockout), but power it off in 2 h if nobody intervenes
    $SSH --command 'sudo shutdown -h +120' < /dev/null >> "$LOG" 2>&1
    st FAILED "fingerprint mismatch, nothing launched; VM powers off in 2 h (sudo shutdown -c to keep it):$diff see $D/fingerprint_$VM.txt"
    exit 4
  fi
  echo "$(ts) fingerprint matches the seed-0 reference" >> "$LOG"
fi

# 3. stage + launch (POWEROFF after the seeds; 30 min grace so this launcher can fetch the logs first)
launch() {  # $1 = seeds
  $SSH --command 'mkdir -p ~/parity3_scripts ~/parity3/pour; rm -f ~/parity3/pour/SEEDS_DONE' < /dev/null >> "$LOG" 2>&1
  $SSH --command 'test -f ~/parity3_scripts/common.sh' < /dev/null || gcloud compute scp $G --tunnel-through-iap $S/common.sh $VM:parity3_scripts/ >> "$LOG" 2>&1
  gcloud compute scp $G --tunnel-through-iap $S/stage_pour.sh $S/stage_pour_seeds.sh $S/franka_pour_published_asset.patch \
    $VM:parity3_scripts/ >> "$LOG" 2>&1 || return 1
  $SSH --command "chmod +x ~/parity3_scripts/stage_pour*.sh; SEEDS='$1' POWEROFF=1 POWEROFF_DELAY=1800 nohup ~/parity3_scripts/stage_pour_seeds.sh > /dev/null 2>&1 & sleep 3; pgrep -f stage_pour_seeds" < /dev/null >> "$LOG" 2>&1
}
fetch() {  # copy the per-seed train logs, VM user scrubbed
  for s in $SEEDS $FB_SEEDS; do
    gcloud compute scp $G --tunnel-through-iap "$VM:parity3/pour/train_1024_curve_s$s.log" $D/ > /dev/null 2>&1 &&
      sed -i '' "s#/home/[A-Za-z0-9._-]*/#/home/<vm-user>/#g" $D/train_1024_curve_s$s.log
  done
  gcloud compute scp $G --tunnel-through-iap "$VM:parity3/pour/stage_pour.log" $D/stage_pour_$VM.log > /dev/null 2>&1 &&
    sed -i '' "s#/home/[A-Za-z0-9._-]*/#/home/<vm-user>/#g" $D/stage_pour_$VM.log
}
last_it() { grep -o "Learning iteration [0-9]*" "$D/train_1024_curve_s$1.log" 2>/dev/null | tail -1 | grep -o "[0-9]*$"; }
run_seeds() {  # $1 = seeds; returns when SEEDS_DONE appears (logs fetched) or the VM went down
  launch "$1" || { st FAILED "launch of seeds [$1] failed (see log); VM left running"; exit 5; }
  t0=$(date +%s); n=$(echo $1 | wc -w)
  fin=$(date -r $(( t0 + $(python3 -c "print(int($n*$PER_SEED_H*3600))") )) '+%F %H:%M %Z')
  st STARTED "seeds [$1] launched at $(date -r $t0 '+%F %T'); expected finish ~$fin (5.3 h per seed)"
  while sleep 300; do
    state=$(gcloud compute instances describe $VM $G --format='value(status)' 2>/dev/null)
    if [ "$state" != RUNNING ]; then
      [ -z "$state" ] && continue   # API hiccup / auth: retry next poll
      st FAILED "VM went $state before SEEDS_DONE; logs as of the last poll in $D"; exit 6
    fi
    fetch
    prog=""; for s in $1; do prog="$prog s$s:it$(last_it $s)"; done
    if $SSH --command 'test -f ~/parity3/pour/SEEDS_DONE' < /dev/null > /dev/null 2>&1; then
      fetch; echo "$(ts) seeds [$1] done:$prog" >> "$LOG"; return 0
    fi
    st TRAINING "seeds [$1] since $(date -r $t0 '+%F %T'), expected finish ~$fin; progress:$prog"
  done
}
run_seeds "$SEEDS"
if [ -n "$FB_SEEDS" ]; then
  peer=$(cut -d' ' -f1 "$D/status_$FB_PEER.txt" 2>/dev/null)
  if [ "$peer" = RETRYING ] || [ "$peer" = FAILED ]; then
    echo "fallback: $VM runs seeds [$FB_SEEDS] at $(ts)" > "$D/ABORT_$FB_PEER"
    # cancel the pending poweroff of the finished run, then run the fallback seeds (they power off at the end)
    $SSH --command 'pkill -f stage_pour_seeds.sh; true' < /dev/null >> "$LOG" 2>&1
    SEEDS_DONE_FIRST=$SEEDS; SEEDS="$SEEDS $FB_SEEDS"
    run_seeds "$FB_SEEDS"
  else
    echo "$(ts) peer $FB_PEER is '$peer': no fallback" >> "$LOG"
  fi
fi
st DONE "seeds [$SEEDS] finished; logs in $D/train_1024_curve_s*.log (VM user scrubbed); VM powers itself off ~30 min after the last seed"
