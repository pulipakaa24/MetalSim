#!/bin/bash
# Fast-forward the shared fork checkouts to the commits this tree pins, THROUGH the GPU queue, so a checkout never
# changes while a queued job is importing it (a fast-forward during import produced a spurious "Warp kernels cannot
# return values" failure on 2026-09-26). Timing class: granted ahead of waiting jobs, runs for seconds.
#
# Pins, read from this checkout (the same sources tests/test_fork_pins.py reads):
#   MuJoCo Warp  the commit of the `pip install ... mujoco_warp@<sha>` line in README.md (docs/GUIDE.md must agree)
#   Warp         PIN in scripts/setup_warp.sh
# Checkouts: upstream/mujoco_warp, upstream/warp-metalsim and (older layout) upstream/warp-innate of the main checkout;
# run from a git worktree, that is the main checkout of the repository, which also holds .venv and the GPU queue.
# FORK_SYNC_UPSTREAM overrides the directory that holds them. The fork remote is found by URL
# (github.com/pulipakaa24/mujoco_warp or .../warp, https or ssh), whatever it is called (`origin` on a clone made by
# scripts/setup_warp.sh, `fork` on older ones), and fetched (its branches, then the commit by SHA) only when the
# pinned commit is missing. A checkout ahead of or diverged from its pin, or with uncommitted changes to tracked files,
# is reported and left as it is. libwarp is rebuilt when the fast-forward changed Warp's native sources (a stale dylib
# fails at wp.init(), not at import); then MuJoCo Warp and Warp are imported and wp.init() is run once.
#
# Usage: scripts/fork_sync.sh [-n]            -n: dry run: report each checkout against its pin, change nothing, no
#                                             fetch, no GPU queue; exit 0 all at their pins, 2 something to do,
#                                             1 something would be refused
#        FORKS="mujoco_warp" scripts/fork_sync.sh    (a subset of the checkout directory names)
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
MAIN=$HERE
COMMON=$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) &&
  [ "$(basename "$COMMON")" = .git ] && MAIN=$(dirname "$COMMON")
UP=${FORK_SYNC_UPSTREAM:-$MAIN/upstream}

MODE=run
case "${1:-}" in
  -n|--dry-run) MODE=dry ;;
  --in-queue) MODE=queue ;;                      # internal: the body, run by gpu_run.sh
  "") ;;
  *) echo "usage: scripts/fork_sync.sh [-n]" >&2; exit 64 ;;
esac

MJW_PIN=$(grep -oE 'mujoco_warp@[0-9a-f]{40}' "$HERE/README.md" | head -1 | cut -d@ -f2)
GUIDE_PIN=$(grep -oE 'mujoco_warp@[0-9a-f]{40}' "$HERE/docs/GUIDE.md" | head -1 | cut -d@ -f2)
WARP_PIN=$(sed -nE 's/^PIN=([0-9a-f]{40}).*/\1/p' "$HERE/scripts/setup_warp.sh" | head -1)
[ -n "$MJW_PIN" ] && [ -n "$WARP_PIN" ] || { echo "fork_sync: pins not found (README.md pip line, setup_warp.sh PIN)"; exit 1; }
[ "$MJW_PIN" = "$GUIDE_PIN" ] || { echo "fork_sync: README.md pins MuJoCo Warp ${MJW_PIN:0:8}, docs/GUIDE.md ${GUIDE_PIN:0:8}"; exit 1; }

if [ "$MODE" = run ]; then
  exec "$MAIN/scripts/gpu_run.sh" fork_sync timing 1 -- bash "$HERE/scripts/fork_sync.sh" --in-queue
fi

DEFAULT_FORKS=mujoco_warp
for d in warp-metalsim warp-innate; do [ -e "$UP/$d/.git" ] && DEFAULT_FORKS="$DEFAULT_FORKS $d"; done
FORKS=${FORKS:-$DEFAULT_FORKS}
PY=$MAIN/.venv/bin/python

fork_remote() {   # $1 checkout, $2 repo name: the remote whose fetch URL is .../pulipakaa24/$2[.git]
  git -C "$1" remote -v | awk -v r="$2" '$3 == "(fetch)" { u = $2; sub(/\/$/, "", u); sub(/\.git$/, "", u)
    n = split(u, p, /[\/:]/); if (n >= 2 && p[n] == r && p[n-1] == "pulipakaa24") { print $1; exit } }'
}

RC=0
worse() { [ "$1" = 1 ] && RC=1; [ "$1" = 2 ] && [ "$RC" = 0 ] && RC=2; return 0; }

sync_one() {      # $1 checkout directory name under $UP
  local d=$1 dir=$UP/$1 repo pin remote head n native
  case $d in
    mujoco_warp) repo=mujoco_warp pin=$MJW_PIN ;;
    warp-metalsim|warp-innate) repo=warp pin=$WARP_PIN ;;
    *) echo "$d: unknown checkout (known: mujoco_warp, warp-metalsim, warp-innate)"; worse 1; return ;;
  esac
  [ -e "$dir/.git" ] || { echo "$d: no checkout at $dir (skipped)"; return; }
  head=$(git -C "$dir" rev-parse HEAD)
  if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
    echo "$d: uncommitted changes to tracked files at ${head:0:8}; not moved (pin ${pin:0:8})"; worse 1; return
  fi
  [ "$head" = "$pin" ] && { echo "$d: at its pin ${pin:0:8}"; return; }
  if ! git -C "$dir" cat-file -e "$pin^{commit}" 2>/dev/null; then
    remote=$(fork_remote "$dir" "$repo")
    [ -n "$remote" ] || { echo "$d: pin ${pin:0:8} not present and no remote with URL .../pulipakaa24/$repo:"
                          git -C "$dir" remote -v | sed 's/^/  /'; worse 1; return; }
    if [ "$MODE" = dry ]; then echo "$d: at ${head:0:8}; pin ${pin:0:8} not present: would fetch $remote, then fast-forward"; worse 2; return; fi
    git -C "$dir" fetch -q "$remote" || { echo "$d: fetch $remote failed"; worse 1; return; }
    git -C "$dir" cat-file -e "$pin^{commit}" 2>/dev/null ||        # a narrowed refspec (single-branch clone): by SHA
      git -C "$dir" fetch -q "$remote" "$pin" 2>/dev/null
    git -C "$dir" cat-file -e "$pin^{commit}" 2>/dev/null || { echo "$d: $remote has no commit $pin"; worse 1; return; }
  fi
  if git -C "$dir" merge-base --is-ancestor "$pin" "$head"; then
    echo "$d: at ${head:0:8}, ahead of its pin ${pin:0:8} by $(git -C "$dir" rev-list --count "$pin..$head") commit(s); not moved"
    worse 1; return
  fi
  git -C "$dir" merge-base --is-ancestor "$head" "$pin" || { echo "$d: ${head:0:8} has diverged from its pin ${pin:0:8}; not moved"; worse 1; return; }
  n=$(git -C "$dir" rev-list --count "$head..$pin")
  native=
  [ "$repo" = warp ] && native=$(git -C "$dir" diff --name-only "$head" "$pin" -- warp/native build_lib.py | head -1)
  if [ "$MODE" = dry ]; then
    echo "$d: would fast-forward ${head:0:8} -> ${pin:0:8} ($n commit(s))${native:+, then rebuild libwarp (native sources change)}"
    worse 2; return
  fi
  git -C "$dir" merge -q --ff-only "$pin" || { echo "$d: fast-forward to ${pin:0:8} failed"; worse 1; return; }
  echo "$d -> ${pin:0:8} (fast-forward of $n commit(s) from ${head:0:8})"
  if [ -n "$native" ]; then
    echo "$d: Warp native sources changed: rebuilding libwarp"
    (cd "$dir" && "$PY" build_lib.py --no-cuda > "$MAIN/runs/warp_rebuild.log" 2>&1) ||
      { echo "$d: libwarp rebuild FAILED, see runs/warp_rebuild.log"; worse 1; return; }
  fi
}

echo "pins (from $HERE): MuJoCo Warp ${MJW_PIN:0:8} (README.md), Warp ${WARP_PIN:0:8} (scripts/setup_warp.sh); checkouts in $UP"
for d in $FORKS; do sync_one "$d"; done
[ "$MODE" = dry ] && exit $RC
(cd "$MAIN" && "$PY" -c "import warp as wp; wp.config.quiet=True; import mujoco_warp; wp.init(); assert wp.is_metal_available(); print('runtime ok: warp', wp.__version__, wp.__file__, '| mujoco_warp', mujoco_warp.__file__)") || RC=1
exit $RC
