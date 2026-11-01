#!/bin/bash
# il3_handover.sh: run rough_il3 on the EPA-fixed fork (1eb40d55) instead of 93e5ce18 (handoff/hfield-epa.md, Next 2),
# without editing the running run_all.sh. Waits until run_all.sh has queued export_s2 (seed 2's training is done and
# export_s2 passed its STOP check), touches STOP so the driver skips il3fix_rough_s0 and exits, then removes STOP and
# launches run_il3fix.sh detached (same way as launch.sh) with FROM=il3fix_rough_s0.
R=~/MetalSim; RR=runs/rough_rerun_2026-10-31; O=$R/$RR; cd "$R" || exit 1
LOG=$O/il3_handover.log; DP=$1
log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*" >> $LOG; }
log "watcher pid $$, driver pid $DP"
until grep -q '"job": "export_s2", "state": "queued"' $O/status.jsonl; do
  kill -0 $DP 2>/dev/null || { log "driver $DP exited before export_s2 was queued: nothing done"; exit 1; }
  sleep 1
done
touch $O/STOP; log "export_s2 queued: STOP touched"
while kill -0 $DP 2>/dev/null; do sleep 5; done
log "driver exited; last status lines:"; tail -n 4 $O/status.jsonl | cut -c1-160 >> $LOG
grep -q '"job": "il3fix_rough_s0", "state": "skipped"' $O/status.jsonl || { log "il3fix_rough_s0 not recorded as skipped: not launching (check by hand)"; exit 1; }
rm -f $O/STOP
env -u PYTHONPATH FROM=il3fix_rough_s0 SKIP= nohup /usr/bin/perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' \
  /usr/bin/caffeinate -i -s /bin/bash $RR/run_il3fix.sh >> $O/driver.out 2>&1 < /dev/null &
LP=$!; disown $LP 2>/dev/null
for i in $(seq 1 15); do [ -s $O/driver.pid ] && break; sleep 1; done
log "launched run_il3fix.sh: launcher $LP, driver $(cat $O/driver.pid 2>/dev/null)"
