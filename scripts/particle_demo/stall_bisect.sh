#!/bin/bash
# stall_bisect.sh STAGE OUT_DIR: one stage of the bisect of the demo-frame stall on macOS 26 (run it as an exclusive
# timing job through scripts/gpu_run.sh, from a frozen copy of the code: CODE env, default this checkout).
#   1: instance-structure build only (--timing --no-trace), 1 vs 4 ground planes x 50k / 300k / all particles
#   2: one 256x144, 1 spp trace pass at the same sizes (1 plane first, then 4)
#   3: the full 1080p frame (4 planes, 128 spp, OIDN)
# Per configuration: utilisation, processes stuck exiting (E state), the 10 s GPU health probe, then the render under a
# 60 s no-output watchdog. Stops at the first stall or failed probe (exit code != 0).
STAGE=$1; B=$2
C=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
export PYTHONPATH=$C
PY=${PY:-/Users/aditya/robosim/.venv/bin/python}
SIM=${SIM:-/Users/aditya/robosim/runs/demo/anymal_sim}
mkdir -p "$B/out"
health() {
  echo "[health] $1: utilisation $(/Users/aditya/robosim/scripts/gpu_util.sh)%, E-state processes: [$(ps -axo stat=,pid=,comm= | awk '$1 ~ /E/' | tr '\n' ';')]"
  $PY "$C/scripts/particle_demo/gpu_probe.py"
}
run() {
  local name=$1; shift
  health "before $name" || { echo "[stage$STAGE] STOP: probe failed before $name"; exit 3; }
  /usr/bin/python3 "$C/scripts/particle_demo/watchdog.py" --timeout 600 --stall 60 --log "$B/stage${STAGE}_$name.log" -- \
    $PY "$C/scripts/particle_demo/render_anymal.py" --sim "$SIM" --out "$B/out" --grains 27 --frames 0:1 --camera track "$@"
  local rc=$?
  grep -a "\[timing\]\|\[render\] frame\|Error\|assert\|watchdog\]" "$B/stage${STAGE}_$name.log" | sed "s/^/  [$name] /"
  health "after $name"
  if [ $rc -ne 0 ]; then echo "[stage$STAGE] STOP: $name exit $rc"; exit $rc; fi
}
LOW="--width 256 --height 144 --spp 1"
case $STAGE in
1) for c in "1 50000" "4 50000" "1 300000" "4 300000" "1 0" "4 0"; do set -- $c
     run build_p$1_n$2 --timing --no-trace $LOW --ground-planes $1 --max-particles $2; done ;;
2) for c in "1 50000" "1 300000" "1 0" "4 50000" "4 300000" "4 0"; do set -- $c
     run trace_p$1_n$2 --timing $LOW --ground-planes $1 --max-particles $2; done ;;
3) run full_p4 --width 1920 --height 1080 --passes 32 --spp 4 --ground-planes 4 ;;
*) echo "stage 1, 2 or 3"; exit 2 ;;
esac
echo "[stage$STAGE] done"
