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

Needs an Apple Silicon Mac with the Xcode Command Line Tools (`xcode-select --install`) and Python 3.12
or newer. MetalSim runs on two forks, neither on PyPI:

- Warp: https://github.com/pulipakaa24/warp branch `metalsim` (innate-inc's Metal backend plus the
  commits in `patches/warp/`).
- MuJoCo Warp: https://github.com/pulipakaa24/mujoco_warp branch `metalsim` (the `metal` device branch
  plus the contact fixes in `patches/mujoco_warp/`).

```
git clone https://github.com/pulipakaa24/MetalSim && cd MetalSim
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
scripts/setup_warp.sh                      # clones, builds (build_lib.py --no-cuda) and installs the Warp fork
pip install -e "git+https://github.com/pulipakaa24/mujoco_warp@metalsim#egg=mujoco-warp"
scripts/fetch_isaac_assets.sh              # NVIDIA's g1_minimal.usd for the G1 parity tests (not redistributed)
pytest tests -q
```

Set `METALSIM_WARP_REPO` / `METALSIM_WARP_REF` (branch, tag or commit) to point the script at another
fork or commit. If the fork is unreachable the script rebuilds it from innate-inc/warp plus
`patches/warp/`. Optional extras (`parity`, `rslrl`, `newton`, `tron1`, `learn`), the exact fork
commits, first runs and bringing your own robot are in [`docs/GUIDE.md`](docs/GUIDE.md).

## What is and is not claimed

`docs/PARITY.md` lists, per Isaac Sim / Isaac Lab feature, the test that confirms or disproves
equivalence, with results and verdicts. `docs/STATUS.md` is the state against the plan.
