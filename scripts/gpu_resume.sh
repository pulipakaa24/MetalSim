#!/bin/bash
# Resume the GPU queue: end the pause placeholder (its whole process group) so the next waiting job is granted.
cd "$(dirname "$0")/.."
PIDF=runs/.gpu_pause.pid
if [ -f $PIDF ]; then p=$(cat $PIDF); kill -- -"$p" 2>/dev/null; kill "$p" 2>/dev/null; rm -f $PIDF; fi
python3 scripts/gpu_lock.py release PAUSE 2>/dev/null
rm -f runs/gpu_queue/*_PAUSE_*.json
echo "queue resumed"; python3 scripts/gpu_lock.py status | head -3
