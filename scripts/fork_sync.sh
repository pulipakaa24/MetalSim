#!/bin/bash
# Fast-forward the shared fork checkouts (upstream/mujoco_warp, upstream/warp-innate) to fork/metalsim, THROUGH the
# GPU queue, so the update never happens while a queued job is importing them (a fast-forward during import produced a
# spurious "Warp kernels cannot return values" failure on 2026-09-26). Timing class: granted ahead of waiting jobs, runs
# for seconds.  Usage: scripts/fork_sync.sh            (optionally FORKS="mujoco_warp warp-innate")
cd "$(dirname "$0")/.."
FORKS=${FORKS:-"mujoco_warp warp-innate"}
scripts/gpu_run.sh fork_sync timing 1 -- bash -c '
for d in '"$FORKS"'; do
  git -C upstream/$d fetch -q fork || exit 1
  old=$(git -C upstream/$d rev-parse HEAD)
  git -C upstream/$d merge -q --ff-only fork/metalsim || { echo "$d: not fast-forwardable"; exit 1; }
  echo "$d -> $(git -C upstream/$d rev-parse --short HEAD)"
  if [ "$d" = warp-innate ] && [ -n "$(git -C upstream/$d diff --name-only $old HEAD -- warp/native build_lib.py)" ]; then
    echo "warp native sources changed: rebuilding libwarp"     # a stale dylib fails at wp.init() (symbol not found), not at import
    (cd upstream/$d && ../../.venv/bin/python build_lib.py --no-cuda > ../../runs/warp_rebuild.log 2>&1) || { echo "warp rebuild FAILED, see runs/warp_rebuild.log"; exit 1; }
  fi
done
.venv/bin/python -c "import mujoco_warp, warp as wp; wp.config.quiet=True; wp.init(); assert wp.is_metal_available(); print(\"runtime ok\", wp.__version__)"'
