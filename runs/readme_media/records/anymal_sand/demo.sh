#!/bin/bash
# the demo at the original settings on origin/main 94b75be (two-level grains, chunks of 2^24 - 1 via ramp.py):
# 240 frames 1920x1080 at 30 fps, 32 passes x 4 spp, OIDN, then the three 256-spp stills; health before and after
L=~/metalsim-fresh/runs_render/sand25m
R=~/metalsim-fresh/MetalSim
export RENDER_SCRIPT=$L/repo_fix/scripts/particle_demo/render_anymal.py PYTHONPATH=$R OIDN_LIB=$HOME/metalsim-fresh/render_deps/pyoidn/oidn/lib/libOpenImageDenoise.2.dylib POLYHAVEN=$HOME/metalsim-fresh/assets/polyhaven
U() { echo "[health] $(date +%H:%M:%S) $1: $(ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o "\"[A-Za-z]* Utilization %\"=[0-9]*\|\"recoveryCount\"=[0-9]*" | tr "\n" " ")"; }
U before
SIM=$HOME/metalsim-fresh/runs_render/sim1
$R/.venv/bin/python $L/ramp.py --sim $SIM --out $L/demo/frames --grains 27 --width 1920 --height 1080 --passes 32 --frames 0:240 --camera track || exit 1
U "after frames"
for c in still-a:150 still-b:200 still-c:225; do
  $R/.venv/bin/python $L/ramp.py --sim $SIM --out $L/demo/stills --grains 27 --width 1920 --height 1080 --passes 64 --camera ${c%%:*} --frames ${c##*:}:$((${c##*:}+1)) || exit 2
done
U "after stills"
echo "[demo] done $(date +%H:%M:%S)"
