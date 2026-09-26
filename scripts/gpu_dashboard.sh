#!/bin/bash
# Start (or restart) the GPU queue dashboard server in the background: http://localhost:8765
# Idempotent: a running server on the port is left alone unless --restart is given.
cd "$(dirname "$0")/.."
PORT=${PORT:-8765}; PIDF=runs/.gpu_dashboard.pid
if [ "${1:-}" = "--restart" ] && [ -f $PIDF ]; then kill "$(cat $PIDF)" 2>/dev/null; rm -f $PIDF; sleep 1; fi
if [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "dashboard already running (pid $(cat $PIDF)) at http://localhost:$PORT/"; exit 0; fi
nohup python3 scripts/gpu_dashboard.py --serve $PORT > runs/gpu_dashboard.server.log 2>&1 &
echo $! > $PIDF; sleep 1
kill -0 $! 2>/dev/null && echo "dashboard started (pid $!) at http://localhost:$PORT/" || { echo "failed:"; cat runs/gpu_dashboard.server.log; exit 1; }
