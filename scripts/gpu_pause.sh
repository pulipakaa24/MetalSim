#!/bin/bash
# Pause the GPU queue after the current job: a placeholder ticket at the front of the timing class takes the lock
# as soon as the holder releases and keeps it until scripts/gpu_resume.sh. Waiting jobs stay queued in order.
cd "$(dirname "$0")/.."
PIDF=runs/.gpu_pause.pid
if [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "already pausing/paused (pid $(cat $PIDF))"; exit 0; fi
nohup bash -c 'python3 scripts/gpu_lock.py acquire PAUSE --kind timing --minutes 99999 --pid $$ --front --cmd "queue paused by the user"; trap "python3 scripts/gpu_lock.py release PAUSE --pid \$\$" EXIT; while true; do sleep 60; done' > runs/gpu_pause.log 2>&1 &
echo $! > $PIDF
echo "pause requested (pid $!): the GPU will be held after the current job releases; scripts/gpu_resume.sh to continue"
