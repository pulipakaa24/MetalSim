#!/bin/zsh
# Builds and installs the MetalSim fork of Warp (Metal backend with interop, ICB graph replay and the
# per-command-buffer free release) into the current Python environment. No CUDA, no Xcode.app needed
# (Command Line Tools only). Usage: scripts/setup_warp.sh [dir]   (default: upstream/warp-metalsim)
set -e
DIR=${1:-upstream/warp-metalsim}
REPO=${METALSIM_WARP_REPO:-https://github.com/pulipakaa24/warp}     # fork of innate-inc/warp, branch metalsim
REF=${METALSIM_WARP_REF:-metalsim}
if [ ! -d "$DIR" ]; then
  git clone --branch "$REF" "$REPO" "$DIR" || {
    echo "fork not reachable; cloning innate-inc/warp and applying patches/"; git clone https://github.com/innate-inc/warp "$DIR"
    (cd "$DIR" && git apply "$(dirname "$0")/../patches/"*.patch)
  }
fi
cd "$DIR"
python build_lib.py --no-cuda
pip install -e .
python -c "import warp as wp; wp.config.quiet=True; wp.init(); print('warp', wp.__version__, 'metal:', wp.is_metal_available())"
