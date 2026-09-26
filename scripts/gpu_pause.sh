#!/bin/bash
# Pause the GPU queue after the current job: a placeholder ticket at the front of the timing class takes the lock
# as soon as the holder releases and keeps it until scripts/gpu_resume.sh. Waiting jobs stay queued in order.
# The placeholder runs in its own process group so resume can kill the acquire and the holder together.
cd "$(dirname "$0")/.."
PIDF=runs/.gpu_pause.pid
if [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "already pausing/paused (pid $(cat $PIDF))"; exit 0; fi
python3 - <<'PY' > runs/gpu_pause.log 2>&1 &
import os, subprocess, sys, time
os.setsid()
p = subprocess.run([sys.executable, "scripts/gpu_lock.py", "acquire", "PAUSE", "--kind", "timing", "--minutes", "99999", "--pid", str(os.getpid()), "--front", "--cmd", "queue paused by the user"])
print("PAUSE holds the GPU", time.strftime("%H:%M:%S"), flush=True)
try:
    while True: time.sleep(60)
finally:
    subprocess.run([sys.executable, "scripts/gpu_lock.py", "release", "PAUSE", "--pid", str(os.getpid())])
PY
echo $! > $PIDF
echo "pause requested (pid $!): the GPU will be held after the current job releases; scripts/gpu_resume.sh to continue"
