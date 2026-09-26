#!/bin/bash
# Resume the GPU queue: end the pause placeholder so the next waiting job is granted.
cd "$(dirname "$0")/.."
PIDF=runs/.gpu_pause.pid
[ -f $PIDF ] && { kill "$(cat $PIDF)" 2>/dev/null; rm -f $PIDF; }
python3 scripts/gpu_lock.py release PAUSE 2>/dev/null
rm -f runs/gpu_queue/0.000_0_PAUSE.json
echo "queue resumed"; python3 scripts/gpu_lock.py status | head -3
