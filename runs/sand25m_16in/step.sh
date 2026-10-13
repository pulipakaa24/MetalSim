#!/bin/bash
# step.sh NAME STALL_S [render_anymal args...]: one ramp step on this Mac. Health before (device / renderer / tiler
# utilisation < 10 % within 120 s, no process stuck exiting, recoveryCount), the step alone as an exclusive timing job
# through scripts/gpu_run.sh under the watchdog, health after. Exit 0 = passed and healthy; 3 = not healthy before
# (nothing run); 4 = not healthy after (queue paused with scripts/gpu_pause.sh); other = the step's exit code.
NAME=$1; STALL=$2; shift 2
W=/Users/aditya/robosim/.claude/worktrees/agent-a8f21c8b02205ff25
L=$W/runs/sand25m_16in
agx() { ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o "\"$1 Utilization %\"=[0-9]*" | head -1 | sed 's/.*=//'; }
rcount() { ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"recoveryCount"=[0-9]*' | head -1 | sed 's/.*=//'; }
health() {
  local d r t i E
  for i in $(seq 1 24); do
    d=$(agx Device); r=$(agx Renderer); t=$(agx Tiler)
    [ "${d:-100}" -lt 10 ] && [ "${r:-100}" -lt 10 ] && [ "${t:-100}" -lt 10 ] && break
    sleep 5
  done
  E=$(ps -axo stat=,pid=,comm= | awk '$1 ~ /E/' | tr '\n' ';')
  echo "[health] $(date +%H:%M:%S) $1: device $d% renderer $r% tiler $t% (after $((i * 5 - 5)) s), recoveryCount $(rcount), E-state [$E]"
  [ "${d:-100}" -lt 10 ] && [ "${r:-100}" -lt 10 ] && [ "${t:-100}" -lt 10 ] && [ -z "$E" ]
}
health "before $NAME" || { echo "[step] STOP $(date +%H:%M:%S): not healthy before $NAME (nothing run)"; exit 3; }
echo "[step] $(date +%H:%M:%S) queue $NAME: $*"
cd /Users/aditya/robosim
scripts/gpu_run.sh sand25m16_$NAME timing 15 -- /usr/bin/python3 $W/scripts/particle_demo/watchdog.py --timeout 900 --stall $STALL \
  --log $L/$NAME.log -- /Users/aditya/robosim/.venv/bin/python $L/ramp.py --sim /Users/aditya/robosim/runs/demo/anymal_sim \
  --out $L/out_$NAME --grains 27 --frames 0:1 --camera track --ground-planes 4 "$@" > $L/$NAME.queue.log 2>&1
rc=$?
echo "[step] $(date +%H:%M:%S) $NAME exit $rc"
grep -a "\[timing\]\|\[render\] frame\|\[ramp\]\|Error\|refused\|watchdog\]" $L/$NAME.log | sed "s/^/  [$NAME] /"
grep -a "\[gpu_run\]" $L/$NAME.queue.log | sed "s/^/  /"
if ! health "after $NAME"; then
  /Users/aditya/robosim/scripts/gpu_pause.sh
  echo "[step] STOP $(date +%H:%M:%S): NOT HEALTHY after $NAME (exit $rc); queue paused"; exit 4
fi
exit $rc
