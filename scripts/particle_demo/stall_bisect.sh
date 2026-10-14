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
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # the checkout this script is in (its .venv, runs/, queue scripts)
PY=${PY:-$REPO/.venv/bin/python}
SIM=${SIM:-$REPO/runs/demo/anymal_sim}
mkdir -p "$B/out"
case $STAGE in 1|3|slow|narrow|twolevel)   # these build > 6.3 M grains: never on a machine without the validated fix
  if [ -z "$ALLOW_LARGE" ]; then
    echo "[stage$STAGE] refused: builds more than 6.3 M grains; set ALLOW_LARGE=1 only where that is validated"; exit 5
  fi ;;
esac
agx() {   # AGX accelerator utilisation (%) by name: agx Device|Renderer|Tiler
  ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o "\"$1 Utilization %\"=[0-9]*" | head -1 | sed 's/.*=//'
}
health() {   # healthy = device, renderer and tiler utilisation all < 10 % within 120 s (no orphaned GPU work), then the probe
  local d r t i
  for i in $(seq 1 24); do
    d=$(agx Device); r=$(agx Renderer); t=$(agx Tiler)
    if [ "${d:-100}" -lt 10 ] && [ "${r:-100}" -lt 10 ] && [ "${t:-100}" -lt 10 ]; then break; fi
    /bin/sleep 5
  done
  echo "[health] $1: device $d% renderer $r% tiler $t% (after $((i * 5 - 5)) s), E-state processes: [$(ps -axo stat=,pid=,comm= | awk '$1 ~ /E/' | tr '\n' ';')]"
  if ! { [ "${d:-100}" -lt 10 ] && [ "${r:-100}" -lt 10 ] && [ "${t:-100}" -lt 10 ]; }; then
    echo "[health] NOT idle: GPU work without an owner?"; return 4
  fi
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
  local hrc=$?
  if [ $rc -ne 0 ]; then
    if [ -n "$CONTINUE" ] && [ $hrc -eq 0 ]; then echo "[stage$STAGE] $name exit $rc (GPU healthy: continuing)"; return; fi
    echo "[stage$STAGE] STOP: $name exit $rc"; exit $rc
  fi
}
LOW="--width 256 --height 144 --spp 1"
case $STAGE in
1) for c in "1 50000" "4 50000" "1 300000" "4 300000" "1 0" "4 0"; do set -- $c
     run build_p$1_n$2 --timing --no-trace $LOW --ground-planes $1 --max-particles $2; done ;;
2) for c in "1 50000" "1 300000" "1 0" "4 50000" "4 300000" "4 0"; do set -- $c
     run trace_p$1_n$2 --timing $LOW --ground-planes $1 --max-particles $2; done ;;
3) run full_p4 --width 1920 --height 1080 --passes 32 --spp 4 --ground-planes 4 ;;
safe)   # <= 6.3 M grains only: tests, then two-level vs flat at 6.28 M grains with 4 planes (build, trace, images)
   health "before tests" || { echo "[stage$STAGE] STOP: GPU not healthy"; exit 3; }
   ( cd "$C" && /usr/bin/python3 "$C/scripts/particle_demo/watchdog.py" --timeout 600 --stall 120 --log "$B/stagesafe_tests.log" -- \
       $PY -P -m pytest -q -rfEs -p no:cacheprovider -s "$C/tests/test_render_particles.py" )
   trc=$?
   grep -a "passed\|failed\|FAILED\|structures\|watchdog\]" "$B/stagesafe_tests.log" | sed "s/^/  [tests] /"
   [ $trc -ne 0 ] && { health "after tests"; echo "[stage$STAGE] STOP: tests exit $trc"; exit $trc; }
   for st in two_level flat; do
     run ${st}_build_p4_6M --timing --no-trace $LOW --ground-planes 4 --max-particles 300000 --structure $st
     run ${st}_trace_p4_6M --timing $LOW --ground-planes 4 --max-particles 300000 --structure $st
     run ${st}_img_p4_6M --width 480 --height 270 --passes 4 --spp 4 --ground-planes 4 --max-particles 300000 \
         --structure $st --save-hdr --out "$B/img_$st"
   done
   $PY - "$B" <<'PYEOF'
import sys, numpy as np, imageio.v3 as iio
B = sys.argv[1]
a = np.load(f"{B}/img_flat/frame_0000_hdr.npy").astype(np.float32); b = np.load(f"{B}/img_two_level/frame_0000_hdr.npy").astype(np.float32)
pa = iio.imread(f"{B}/img_flat/frame_0000.png").astype(int); pb = iio.imread(f"{B}/img_two_level/frame_0000.png").astype(int)
print(f"[compare] flat vs two-level (4 planes, 6.28 M grains, 480x270, 64 spp, OIDN): hdr identical {np.array_equal(a, b)}, "
      f"mean |diff| / mean {np.abs(a - b).mean() / max(a.mean(), 1e-9):.2e}, max |diff| {np.abs(a - b).max():.3e}; "
      f"png pixels differing {(pa != pb).any(-1).mean() * 100:.2f} %, max 8-bit diff {np.abs(pa - pb).max()}")
PYEOF
   ;;
twolevel)   # the two-level grain structure: tests, the stalled 4-plane build / a trace, flat vs two-level images, full frame
   health "before tests" || { echo "[stage$STAGE] STOP: probe failed"; exit 3; }
   ( cd "$C" && /usr/bin/python3 "$C/scripts/particle_demo/watchdog.py" --timeout 600 --stall 120 --log "$B/stagetwolevel_tests.log" -- \
       $PY -P -m pytest -q -rfEs -p no:cacheprovider -s "$C/tests/test_render_particles.py" )
   trc=$?
   grep -a "passed\|failed\|FAILED\|structures\|watchdog\]" "$B/stagetwolevel_tests.log" | sed "s/^/  [tests] /"
   [ $trc -ne 0 ] && { echo "[stage$STAGE] STOP: tests exit $trc"; exit $trc; }
   run tl_build_p4_n0 --timing --no-trace $LOW --ground-planes 4 --structure two_level
   run tl_trace_p4_n0 --timing $LOW --ground-planes 4 --structure two_level
   run cmp_flat_p1 --width 480 --height 270 --passes 4 --spp 4 --ground-planes 1 --structure flat --save-hdr --out "$B/cmp_flat"
   run cmp_tl_p1 --width 480 --height 270 --passes 4 --spp 4 --ground-planes 1 --structure two_level --save-hdr --out "$B/cmp_tl"
   $PY - "$B" <<'PYEOF'
import sys, numpy as np, imageio.v3 as iio
B = sys.argv[1]
a = np.load(f"{B}/cmp_flat/frame_0000_hdr.npy").astype(np.float32); b = np.load(f"{B}/cmp_tl/frame_0000_hdr.npy").astype(np.float32)
pa = iio.imread(f"{B}/cmp_flat/frame_0000.png").astype(int); pb = iio.imread(f"{B}/cmp_tl/frame_0000.png").astype(int)
rel = np.abs(a - b).mean() / max(a.mean(), 1e-9)
print(f"[compare] flat vs two-level (1 plane, all particles, 480x270, 64 spp, OIDN): hdr identical {np.array_equal(a, b)}, "
      f"mean |diff| / mean {rel:.2e}, max |diff| {np.abs(a - b).max():.3e}; png pixels differing {(pa != pb).any(-1).mean() * 100:.2f} %, "
      f"max 8-bit diff {np.abs(pa - pb).max()}")
PYEOF
   run full_tl_p4 --width 1920 --height 1080 --passes 32 --spp 4 --ground-planes 4 --structure two_level ;;
narrow)   # build only: 4 planes at ~10 / 15 / 20 M grains; all particles with 2 planes, 4 small planes, 4 far planes
   CONTINUE=1
   for n in 370000 556000 741000; do run build_p4_n$n --timing --no-trace $LOW --ground-planes 4 --max-particles $n; done
   run build_p2_n0 --timing --no-trace $LOW --ground-planes 2
   run build_p4_e6_n0 --timing --no-trace $LOW --ground-planes 4 --plane-extent 6
   run build_p4_far_n0 --timing --no-trace $LOW --ground-planes 4 --plane-shift 200 ;;
slow)   # the stalled configuration alone, 15 min watchdog, heartbeat (elapsed, utilisation) every 10 s
   health "before slow build" || { echo "[stage$STAGE] STOP: probe failed"; exit 3; }
   t0=$(date +%s)
   ( while true; do /bin/sleep 10; echo "[heartbeat] $(( $(date +%s) - t0 )) s, utilisation $("$REPO/scripts/gpu_util.sh")%"; done ) &
   HB=$!
   /usr/bin/python3 "$C/scripts/particle_demo/watchdog.py" --timeout 900 --stall 900 --log "$B/stage_slow_build_p4_n0.log" -- \
     "$PY" "$C/scripts/particle_demo/render_anymal.py" --sim "$SIM" --out "$B/out" --grains 27 --frames 0:1 --camera track \
     --timing --no-trace $LOW --ground-planes 4 --max-particles 0 ${EXTRA:-}
   rc=$?; kill $HB 2>/dev/null
   grep -a "\[timing\]\|watchdog\]" "$B/stage_slow_build_p4_n0.log" | sed "s/^/  [slow] /"
   echo "[slow] elapsed $(( $(date +%s) - t0 )) s, exit $rc"
   health "after slow build"
   exit $rc ;;
*) echo "stage 1, 2 or 3"; exit 2 ;;
esac
echo "[stage$STAGE] done"
