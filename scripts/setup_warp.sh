#!/bin/zsh
# Builds and installs the MetalSim fork of Warp (Metal backend with interop, ICB graph replay and the
# per-command-buffer free release) into the current Python environment. No CUDA, no Xcode.app needed
# (Command Line Tools only). Usage: scripts/setup_warp.sh [dir]   (default: upstream/warp-metalsim)
set -e
PATCHES="$(cd "$(dirname "$0")/../patches" && pwd)"
DIR=${1:-upstream/warp-metalsim}
REPO=${METALSIM_WARP_REPO:-https://github.com/pulipakaa24/warp}     # fork of innate-inc/warp, branch metalsim
REF=${METALSIM_WARP_REF:-metalsim}
if [ ! -d "$DIR" ]; then
  { git clone "$REPO" "$DIR" && git -C "$DIR" checkout -q "$REF"; } || {     # REF: branch, tag or commit
    # the fork is innate-inc/warp ce15f6bb plus the five commits exported to patches/warp/ (same tree as fork b9557cb)
    echo "fork not reachable; cloning innate-inc/warp at ce15f6bb and applying patches/warp/"; rm -rf "$DIR"
    git clone https://github.com/innate-inc/warp "$DIR"
    git -C "$DIR" checkout -q ce15f6bb2545e2d6c4a4c1fac8e40bb800db7265
    for p in "$PATCHES"/warp/*.patch; do git -C "$DIR" apply "$p"; done
  }
fi
cd "$DIR"
python build_lib.py --no-cuda
pip install -e .
python -c "import warp as wp; wp.config.quiet=True; wp.init(); print('warp', wp.__version__, 'metal:', wp.is_metal_available())"
