# MetalSim

Isaac Sim / Isaac Lab parity robotics simulation, native on Apple Silicon.

Design principle (from the plan): match Isaac Sim and Isaac Lab in realism, sensor
fidelity, feature coverage, and per-step efficiency. The only performance gap allowed
is the one caused by the compute, memory bandwidth, and ray-tracing hardware of the
Apple chip in use. Every speed claim is labelled measured, reported, or estimated.

Layout: `metalsim/interop` (Metal <-> Warp <-> PyTorch MPS ordering and zero-copy),
`metalsim/physics` (MuJoCo Warp on Metal), `metalsim/render` (native Metal renderer,
tiers 0-2), `metalsim/scene` (USD scene layer, MJCF/URDF importers), `metalsim/sensors`,
`metalsim/replicator`, `metalsim/learn`, `metalsim/bench`, `metalsim/tools`.
Phase logs and measurements live in `docs/`.

## Install

MetalSim runs on its fork of Warp, https://github.com/pulipakaa24/warp branch `metalsim` (innate-inc's Metal
backend plus the changes in `patches/`), and on MuJoCo Warp's `metal` branch. Neither is on PyPI.

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
scripts/setup_warp.sh                      # clones, builds (build_lib.py --no-cuda) and installs the Warp fork
pip install -e upstream/mujoco_warp        # MuJoCo Warp, branch metal
pytest tests -q
```

Set `METALSIM_WARP_REPO` / `METALSIM_WARP_REF` to point the script at another fork or commit. The
Warp change set is also kept as `patches/*.patch` against innate-inc/warp so a fresh clone can be
rebuilt without the fork.

## What is and is not claimed

`docs/PARITY.md` lists, per Isaac Sim / Isaac Lab feature, the test that confirms or disproves
equivalence, with results and verdicts. `docs/STATUS.md` is the state against the plan.
