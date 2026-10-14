#!/bin/bash
# fresh_install_check.sh BASE [REF]: install MetalSim from a fresh clone exactly as README / docs/GUIDE.md say, then run
# the test suites and a short smoke of each advertised demo, logging every step. Proves the install works on a given
# Mac and user; correctness only (no timing claims).
#
#   BASE  a new directory; the clone goes to "BASE/Metal Sim" (a space on purpose: quoting bugs show up), logs to
#         BASE/logs/NN_step.log, one line per step to BASE/results.tsv (step, status, exit code, seconds, note);
#         BASE/DONE when finished, BASE/STOPPED (with the reason) when it stopped on an anomaly
#   REF   branch, tag or commit to check out (default main)
# Env: METALSIM_REPO_URL (default https://github.com/pulipakaa24/MetalSim); GPU_RUN, the GPU queue to run the GPU
#      steps through (default the clone's own scripts/gpu_run.sh; give the machine's shared queue where agents share
#      the GPU; jobs run in the clone either way); ONLY="step step ..." to run a subset; FORCE=1 re-runs steps
#      that already passed.
# Pins are never named here: the pip lines are read from README.md's "Install" block and the fork commits from the
# setup scripts of the checked-out tree; step 15_pins prints what was installed.
# Resumable: a step that passed (BASE/state/<step>.ok) is skipped when the script runs again with the same BASE.
# Stop-on-anomaly: before every GPU step the GPU must not be busy without an owning process (Metal work that outlived
# its process wedges the GPU, see metalsim/render/PARTICLES.md); a queue refusal (exit 75), a watchdog stall (125) or
# timeout (124), or a process stuck exiting stops the run and writes BASE/STOPPED.
# Caches: pip, uv, Warp kernels and the PyTorch extension build go under BASE/cache, so nothing comes from an earlier
# install on the machine (PIP_CACHE_DIR, UV_CACHE_DIR, WARP_CACHE_PATH, TORCH_EXTENSIONS_DIR).
set -u
BASE=${1:?usage: fresh_install_check.sh BASE [REF]}
REF=${2:-main}
URL=${METALSIM_REPO_URL:-https://github.com/pulipakaa24/MetalSim}
mkdir -p "$BASE/logs" "$BASE/state" "$BASE/cache"
BASE="$(cd "$BASE" && pwd)"
CLONE="$BASE/Metal Sim"
LOGS="$BASE/logs"; STATE="$BASE/state"; RES="$BASE/results.tsv"
rm -f "$BASE/DONE" "$BASE/STOPPED"; touch "$BASE/state/.started"
export PIP_CACHE_DIR="$BASE/cache/pip" UV_CACHE_DIR="$BASE/cache/uv" WARP_CACHE_PATH="$BASE/cache/warp" \
       TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
unset PYTHONPATH VIRTUAL_ENV MENAGERIE POLYHAVEN ISAACLAB3_SOURCE METALSIM_NEWTON152_DIR METALSIM_WARP_DIR CONDA_PREFIX
# a non-interactive shell (ssh, launchd) may not have the user's PATH: the usual places of python3.12, uv and git
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
[ -f "$RES" ] || printf "step\tstatus\trc\tseconds\tnote\n" > "$RES"
log() { echo "[fresh $(date +%H:%M:%S)] $*" | tee -a "$LOGS/00_driver.log"; }
log "base $BASE ref $REF url $URL; $(sw_vers -productName) $(sw_vers -productVersion), $(uname -m), user $(id -un)"

want() { [ -z "${ONLY:-}" ] && return 0; case " $ONLY " in *" $1 "*) return 0 ;; esac; return 1; }
passed() { [ -f "$STATE/$1.ok" ] && [ -z "${FORCE:-}" ]; }
record() {   # record STEP STATUS RC SECONDS NOTE
  printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" >> "$RES"
  rm -f "$STATE/$1.ok" "$STATE/$1.fail"; [ "$2" = ok ] && touch "$STATE/$1.ok" || echo "$3 $5" > "$STATE/$1.fail"
}
stop() { log "STOP: $*"; echo "$(date) $*" > "$BASE/STOPPED"; exit 3; }

# step NAME NEEDS -- command...: runs in the clone (bash -c string), logs to logs/NAME.log; NEEDS lists steps that must
# have passed (else the step is recorded as blocked)
step() {
  local name=$1 needs=$2; shift 3
  want "$name" || return 0
  if passed "$name"; then log "$name: passed earlier, skipped"; return 0; fi
  local n; for n in $needs; do
    [ -f "$STATE/$n.ok" ] || { record "$name" blocked - 0 "needs $n"; log "$name: blocked (needs $n)"; return 0; }
  done
  log "$name: start"
  local t0=$(date +%s) rc
  ( [ -d "$CLONE" ] && cd "$CLONE"; [ -f "$CLONE/.venv/bin/activate" ] && source "$CLONE/.venv/bin/activate"; set -o pipefail
    bash -c "$*" ) > "$LOGS/$name.log" 2>&1
  rc=$?
  local dt=$(( $(date +%s) - t0 )) note
  note=$(grep -E "passed|failed|error|Error|refused|skipped" "$LOGS/$name.log" | tail -1 | tr '\t' ' ' | cut -c1-200)
  if [ $rc -eq 0 ]; then record "$name" ok 0 $dt "$note"; log "$name: ok (${dt}s) $note"
  else record "$name" FAIL $rc $dt "$note"; log "$name: FAIL rc=$rc (${dt}s) $note; log $LOGS/$name.log"; fi
  case $rc in 75) stop "$name: GPU queue refused (runaway GPU work)";; 124|125|126) stop "$name: watchdog rc=$rc (timeout / stall / memory)";; esac
  return 0
}

# ---- GPU health -------------------------------------------------------------------------------------------------------
QROOT=""
gpu_health() {   # 0 = fine; stops the run on an ownerless busy GPU or a process stuck exiting
  local top util i stuck
  for i in $(seq 1 24); do
    util=$("$QROOT/scripts/gpu_util.sh" 2>/dev/null); util=${util:-0}
    top=$(python3 "$QROOT/scripts/gpu_top.py" 2>/dev/null | head -1)
    if [ "$util" -ge 90 ] && echo "$top" | grep -q "none >= 1 %"; then sleep 5; continue; fi
    stuck=$(ps -Ao stat=,pid=,comm= | awk '$1 ~ /E/ && $3 ~ /python/' | head -3)
    [ -n "$stuck" ] && stop "process stuck exiting: $stuck"
    [ -f "$QROOT/runs/GPU_RUNAWAY_ALERT" ] && [ "$QROOT/runs/GPU_RUNAWAY_ALERT" -nt "$BASE/state/.started" ] && stop "GPU_RUNAWAY_ALERT in $QROOT/runs"
    return 0
  done
  stop "GPU at ${util}% with no owning process for 2 min ($top)"
}
# gpu NAME NEEDS MINUTES TIMEOUT_S -- command...: through the queue (check class), under the watchdog, in the clone
gpu() {
  local name=$1 needs=$2 min=$3 to=$4; shift 5
  want "$name" || return 0
  passed "$name" && { log "$name: passed earlier, skipped"; return 0; }
  local q=${GPU_RUN:-$CLONE/scripts/gpu_run.sh}; QROOT="$(cd "$(dirname "$q")/.." && pwd)"
  gpu_health
  step "$name" "$needs" -- "$(printf '%q ' "$q" "fresh_$name" "${GKIND:-check}" "$min" -- /usr/bin/python3 "$CLONE/scripts/watchdog.py" \
       --timeout "$to" --stall 900 --rss-gb 48 --cwd "$CLONE" -- /bin/bash -c "$*")"
}

# ---- install, exactly as README "Install" / docs/GUIDE.md section 1 -------------------------------------------------
step 01_clone "" -- "$(printf 'if [ -d %q/.git ]; then git -C %q fetch -q origin; else git clone %q %q; fi && cd %q && git checkout -q %q && git log -1 --format=%%H%%n%%s' "$CLONE" "$CLONE" "$URL" "$CLONE" "$CLONE" "$REF")"
# README: python3.12 -m venv .venv (uv only: "$(uv python find 3.12)" -m venv .venv)
step 02_venv 01_clone -- 'if command -v python3.12 >/dev/null; then echo "python3.12: $(command -v python3.12)"; python3.12 -m venv .venv;
  else echo "no python3.12 on PATH: the README uv form"; "$(uv python find 3.12)" -m venv .venv; fi && .venv/bin/python -V'
# readme_line PATTERN: the first line of README.md's Install block that starts with PATTERN (an extended regex), run
# as written (its trailing # comment is a shell comment)
readme_line() { printf 'L=$(awk %q README.md | grep -m1 -E %q) && [ -n "$L" ] || { echo "README Install block has no line matching:" %q; exit 2; }; echo "+ $L"; eval "$L"' \
  '/^## Install/{f=1} f&&/^```/{n++; if(n==2) exit} f&&n==1' "^$1" "$1"; }
step 03_pip_metalsim 02_venv -- "source .venv/bin/activate && $(readme_line 'pip install -e "\.\[')"
step 04_setup_warp 03_pip_metalsim -- 'source .venv/bin/activate && scripts/setup_warp.sh'
step 05_pip_mujoco_warp 04_setup_warp -- "source .venv/bin/activate && $(readme_line 'pip install -e "git\+')"' &&
  python -c "import warp, mujoco_warp, mujoco, newton; print(\"warp\", warp.__version__, warp.__file__); print(\"mujoco_warp\", mujoco_warp.__file__); print(\"mujoco\", mujoco.__version__, \"newton\", newton.__version__)" &&
  python -c "import warp, os, sys; sys.exit(0 if \"upstream/warp-metalsim\" in warp.__file__ else \"warp is not the fork: \" + warp.__file__)"'
step 06_fetch_isaac 01_clone -- 'scripts/fetch_isaac_assets.sh'
step 07_menagerie 01_clone -- 'git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie upstream/mujoco_menagerie && ls upstream/mujoco_menagerie | wc -l'
step 08_setup_newton152 05_pip_mujoco_warp -- 'scripts/setup_newton152.sh && .venv-newton152/bin/python -c "import objc, Metal, ninja, torch, metalsim.interop; print(\"pyobjc/ninja/torch ok\")"'
step 09_fetch_polyhaven_demo 01_clone -- 'scripts/fetch_polyhaven_assets.sh demo'
step 15_pins 01_clone -- 'echo "MetalSim $(git rev-parse HEAD) ($(git log -1 --format=%cd --date=iso))"
  grep -hE "^PIN=" scripts/setup_warp.sh scripts/setup_newton152.sh | sed "s/^/setup script /"
  for d in upstream/warp-metalsim upstream/newton-metalsim-1.5.2 .venv/src/mujoco-warp; do
    [ -d "$d/.git" ] || [ -f "$d/.git" ] && echo "installed $d $(git -C "$d" rev-parse HEAD)"; done
  for py in .venv/bin/python .venv-newton152/bin/python; do [ -x $py ] && $py -c "import importlib.metadata as m, warp
for p in (\"warp-lang\", \"mujoco\", \"mujoco-warp\", \"newton\", \"torch\", \"pyobjc-framework-Metal\", \"pyoidn\"):
    try: print(\"$py\", p, m.version(p))
    except Exception as e: print(\"$py\", p, \"-\")
print(\"$py\", \"warp from\", warp.__file__)"; done; true'

# ---- tests (GPU queue, check class) --------------------------------------------------------------------------------
RENDER="tests/test_render_*.py tests/test_materialx.py tests/test_newton_bridge_poses.py tests/test_replicator*.py
        tests/test_scene_usd.py tests/test_sensor_transparency.py tests/test_sensors_rt.py tests/test_lidar_extras.py"
# tests/test_batch_deterministic.py (deterministic BatchSim, new GPU code) runs alone as an exclusive job when present
DET=tests/test_batch_deterministic.py
set -f; IGN=""; for f in $RENDER tests/test_mpm.py tests/test_mpm_main.py $DET; do IGN="$IGN --ignore-glob=$f"; done; set +f
P="-p no:cacheprovider -q -rfEs"
gpu 10_metal_check 05_pip_mujoco_warp 5 600 -- 'python -c "import warp as wp; wp.init(); print(\"metal:\", wp.is_metal_available()); assert wp.is_metal_available()"'
gpu 11_rigid_suite "05_pip_mujoco_warp 06_fetch_isaac 07_menagerie" 90 7200 -- "python -m pytest $P $IGN tests"
gpu 12_render_suite "05_pip_mujoco_warp 06_fetch_isaac 07_menagerie" 60 5400 -- "python -m pytest $P $RENDER"
gpu 13_test_mpm 08_setup_newton152 30 3600 -- "$CLONE/.venv-newton152/bin/python -m pytest $P tests/test_mpm.py"
gpu 14_test_mpm_main "05_pip_mujoco_warp 08_setup_newton152" 60 3600 -- "python -m pytest $P tests/test_mpm_main.py"
if [ -f "$CLONE/$DET" ]; then
  GKIND=timing gpu 16_test_batch_deterministic "05_pip_mujoco_warp" 30 3600 -- "python -m pytest $P $DET"
fi

# ---- demo smokes (small scale) -------------------------------------------------------------------------------------
gpu 20_demo_g1 "05_pip_mujoco_warp 06_fetch_isaac" 15 1800 -- 'mkdir -p runs/fresh && python -m metalsim.learn.g1_velocity 64 flat &&
  python -m metalsim.learn.g1_velocity 64 flat train 2 runs/fresh/g1_flat.log runs/fresh/g1_flat.pt 0.0025 --seed 0 && ls -la runs/fresh/g1_flat.pt'
gpu 21_demo_franka_pour "08_setup_newton152 06_fetch_isaac" 30 3600 -- 'mkdir -p runs/fresh && .venv-newton152/bin/python -m metalsim.learn.franka_pour_ppo bench --num-envs 16 --mode step --steps 5 &&
  .venv-newton152/bin/python -m metalsim.learn.franka_pour_ppo train --num-envs 16 --iterations 1 --save-interval 1 --out runs/fresh/franka_pour && ls runs/fresh/franka_pour'
gpu 22_demo_box_on_sand "05_pip_mujoco_warp" 15 1800 -- 'mkdir -p runs/fresh && python metalsim/physics/mpm/probes/box_on_sand.py --engine metalsim --steps 60 --out runs/fresh/box_on_sand.npz && ls -la runs/fresh/box_on_sand.npz'
gpu 23_demo_anymal_sim 08_setup_newton152 20 2400 -- '.venv-newton152/bin/python scripts/particle_demo/sim_anymal.py --out runs/demo/anymal_small --seconds 1.5 --bed -0.3 -0.3 0 0.3 1.0 0.1 && python3 -c "import json; m=json.load(open(\"runs/demo/anymal_small/meta.json\")); print({k: m[k] for k in m if not isinstance(m[k], (list, dict))})"'
# render: ~80 K particles x 8 grains, far under the 6.3 M grains this repo treats as the safe ceiling
gpu 24_demo_anymal_render "23_demo_anymal_sim 07_menagerie 09_fetch_polyhaven_demo" 20 2400 -- 'python scripts/particle_demo/render_anymal.py --sim runs/demo/anymal_small --out runs/demo/anymal_small_frames --frames 0:2 --width 480 --height 270 --passes 4 --grains 8 --decor 0 0 0 0 && ls -la runs/demo/anymal_small_frames && cat runs/demo/anymal_small_frames/stats_*.json | head -30'
gpu 25_demo_transparency_lidar "05_pip_mujoco_warp" 10 1200 -- 'mkdir -p runs/fresh && python scripts/diagnostics/sensor_transparency_cost.py --n 64 --out runs/fresh/sensor_transparency_64.json'

log "done: $(grep -c "	ok	" "$RES") ok, $(grep -c "	FAIL	" "$RES") failed, $(grep -c "	blocked	" "$RES") blocked (results.tsv keeps every attempt)"
date > "$BASE/DONE"
