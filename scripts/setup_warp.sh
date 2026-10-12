#!/bin/zsh
# Builds and installs the MetalSim fork of Warp (Metal backend with interop, ICB graph replay, deterministic mode,
# device-side launch bounds, GPU sort / BSR / NanoVDB builders, fast_math and FMA contraction as on CUDA; the list is
# METALSIM_CHANGES.md at the branch root) into the current Python environment. No CUDA, no Xcode.app needed (Command
# Line Tools only). The commit is pinned: every MetalSim environment (.venv, .venv-newtonfork, .venv-newton152) shares
# this one checkout. Usage: scripts/setup_warp.sh [dir]   (default: upstream/warp-metalsim)
# Env: METALSIM_WARP_REPO (default github.com/pulipakaa24/warp), METALSIM_WARP_REF (branch, tag or commit; default the
# pinned commit below, the head of branch metalsim).
set -e
PATCHES="$(cd "$(dirname "$0")/../patches" && pwd)"
DIR=${1:-upstream/warp-metalsim}
REPO=${METALSIM_WARP_REPO:-https://github.com/pulipakaa24/warp}     # fork of innate-inc/warp, branch metalsim
PIN=152808b7e33ce08fcc17c898c5ce11d567d60795                         # metalsim head, 2026-10-11 (MPM, scalable determinism, CUDA-faithful fast_math)
REF=${METALSIM_WARP_REF:-$PIN}
if [ ! -d "$DIR" ]; then
  { git clone "$REPO" "$DIR" && git -C "$DIR" checkout -q "$REF"; } || {     # REF: branch, tag or commit
    # the pinned tree is innate-inc/warp ce15f6bb plus patches/warp/metalsim-152808b7.diff (the whole branch as one
    # diff; patches/warp/0001-0007 are the first seven commits, the September tree f194006a)
    echo "fork not reachable; cloning innate-inc/warp at ce15f6bb and applying patches/warp/metalsim-152808b7.diff"; rm -rf "$DIR"
    git clone https://github.com/innate-inc/warp "$DIR"
    git -C "$DIR" checkout -q ce15f6bb2545e2d6c4a4c1fac8e40bb800db7265
    git -C "$DIR" apply "$PATCHES/warp/metalsim-152808b7.diff"
  }
elif [ -e "$DIR/.git" ] && [ -z "$METALSIM_WARP_REF" ] && [ "$(git -C "$DIR" rev-parse HEAD)" != "$PIN" ]; then
  echo "note: $DIR is at $(git -C "$DIR" rev-parse --short HEAD), not the pinned ${PIN:0:8} (left as is; scripts/fork_sync.sh updates shared checkouts)"
fi
cd "$DIR"
python build_lib.py --no-cuda
pip install -e .
python -c "import warp as wp; wp.config.quiet=True; wp.init(); print('warp', wp.__version__, 'metal:', wp.is_metal_available())"
