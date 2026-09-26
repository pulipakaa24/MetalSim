# MetalSim

MetalSim is a robot simulation and learning stack for the GPU of an Apple silicon Mac, built to match
NVIDIA Isaac Sim and Isaac Lab in behaviour, sensor output and task definitions, and measured against them.
It runs MuJoCo Warp physics on Warp's Metal backend, renders on Metal at three tiers (raster, hybrid ray
tracing, a path tracer), traces lidar, radar and height-scan rays on Metal's ray-tracing hardware, generates
labelled data in the layout of Isaac Sim's Replicator, trains policies with a PPO whose rollout runs in Warp
(or with rsl_rl and a PyTorch MPS learner for cameras), and runs Isaac Lab's tasks from Isaac's own assets and
configuration files.

Everything below is measured on an Apple M4 Max (40-core GPU, 64 GB) unless marked otherwise. Isaac
references were recorded by the author on an NVIDIA L4 (Isaac Sim 5.1 + Isaac Lab 2.3.2, and Isaac Sim 6.1 +
Isaac Lab 3.0.0-EA); NVIDIA's published RTX 4090 figures are marked "published". MetalSim is an independent
project, not affiliated with or endorsed by NVIDIA, Google DeepMind or Apple.

## Measured state

The evidence, test by test, is in [`docs/PARITY.md`](docs/PARITY.md); open and closed gaps in
[`docs/GAPS.md`](docs/GAPS.md).

| what | MetalSim | Isaac reference | details |
|---|---|---|---|
| G1 flat velocity task, training (Isaac's flat config and PPO settings) | return +28.4 / +26.3 / +26.0 at iteration 1000 (three seeds), full-length episodes by the same iteration; +27.1 with the base velocity at Isaac's point (pelvis COM, the default since 2026-09-25 evening) | Isaac Lab 2.3.2 + PhysX: +27.3 (one seed) | [§1.5](docs/PARITY.md#15-learning-on-the-g1-task) |
| G1 rough velocity task, training (Isaac's terrain generator, ported exactly) | +16.4 / +22.6 / +19.1 at iteration 1500 (three seeds), terrain level 6.0–6.1 | Isaac Lab 2.3.2 + PhysX: +14.1, level 5.9 (one seed) | [§1.5](docs/PARITY.md#15-learning-on-the-g1-task) |
| Isaac Lab 3.0-EA G1 tasks (its extra push / mass / reset-velocity events), like-for-like on Isaac's own MuJoCo Warp numerics, both trainer fixes in | **flat, three seeds: +27.0 ± 1.4 at iteration 1499** (26.8 / 28.4 / 25.7), inside one standard deviation of Isaac's Newton run at every milestone from iteration 200 (mean +7.2 / +17.9 / +25.4 at 300 / 500 / 1000 vs +8.4 / +19.6 / +25.9); 0 blow-ups; **rough seed 0: +13.7 at 1499, terrain level 5.99** vs Isaac's +14.5 / 5.80 (was +8.4 with 1,010 blow-ups before the two fixes). Two trainer defects found and fixed on the way (no reset before the first episode; effort clamp placed where a saturated drive loses implicit damping) | flat +27.5 (Newton / MuJoCo Warp, 59.4 K on the L4), +28.9 (PhysX, 47.3 K); rough +14.5, level 5.8 (Newton) at iteration 1499, one seed each | [§1.8](docs/PARITY.md#18-isaac-lab-30-ea-reference-isaac-sim-61-measured-on-the-l4-2026-09-25) |
| G1 flat throughput, 4096 envs | **with the elliptic default (impratio 10, cap 20): full PPO loop 32.3 K env-steps/s** (measured 2026-09-26 before that day's +15 % throughput work; re-measurement queued). With the previous pyramidal default: loop 56.4 K, env step 62.3–62.6 K, physics alone 93.7–94.1 K (same day, after the work). Rough, pyramidal: env step 49.1 K, loop 44.5 K. The elliptic premium is the price of PhysX's cone on this solver: 1.60× on the loop after the incremental Newton path (was 7× before 2026-09-25); switch back with `contact_cfg="recommended_pyramidal"` | Isaac Lab 3.0 on the L4 (`isaaclab benchmark runtime`, env step alone): 72.0 K with its default Newton / MuJoCo Warp, 45.1 K PhysX; rough 51.2 K / 35.7 K. Isaac Lab 2.3.2 + PhysX on the L4: 45.9 K; RTX 4090: 82 K (published) | [§1.4](docs/PARITY.md#14-throughput-on-the-g1-task-isaacs-protocol-4096-envs), [§1.8](docs/PARITY.md#18-isaac-lab-30-ea-reference-isaac-sim-61-measured-on-the-l4-2026-09-25) |
| Rendering vs RTX (G1 scene, identical states) | tier 2 + Open Image Denoise vs RTX path tracer: robot pixels 28.8 dB PSNR / SSIM 0.94, whole frame 45.6 dB | RTX real-time vs RTX path tracer on the robot: 25.9 dB apart | [§1.7](docs/PARITY.md#17-fidelity-protocol-against-isaac-sim-51-on-an-l4-recorded-2026-09-24-physx--rtx) |
| Physics vs PhysX (same asset, open-loop protocol) | joints within 0.03–0.06 rad through hold and 1 m drop; contact impulses within 1–4 % of Isaac's; **default contacts since 2026-09-26: elliptic friction cones (PhysX's exact cone), impratio 10, Newton cap 20**, on top of the 10 ms impact-stiffened preset with hard joint limits: 20 ms impact forces 1.08× / 1.07× of PhysX's (pyramidal 1.70× / 1.23×), peak drop penetration 1.55 cm, feet slide within noise of Isaac's; learning inside seed spread | Isaac Sim 5.1 PhysX recording; Isaac Sim 6.1 / Isaac Lab 3.0 PhysX reproduces it within 0.002 rad, and 3.0's own MuJoCo Warp sits 0.002–0.005 rad from its PhysX | [§1.7](docs/PARITY.md#17-fidelity-protocol-against-isaac-sim-51-on-an-l4-recorded-2026-09-24-physx--rtx), [§1.8](docs/PARITY.md#18-isaac-lab-30-ea-reference-isaac-sim-61-measured-on-the-l4-2026-09-25) |
| Deformables (cloth, rope, soft cube) | vs PhysX 5.1: rope period 1.077 vs 1.026 s, soft-cube bounce 0.140 vs 0.135 m, penetration 2.6 vs 1.6 mm (damping fitted); vs Isaac Lab 3.0: soft cube matches PhysX's FEM with PhysX's own parameters, cloth within 7 mm; Newton VBD run on Metal replays 3.0's Newton recording within 1.2 mm over 5 s (same solver, Metal vs CUDA) | Isaac Sim 5.1 and 6.1 recordings | [§2](docs/PARITY.md#2-platform-capabilities-the-workstreams-with-the-tests-behind-them), [GAPS](docs/GAPS.md#open-2026-09-25-evening) |
| Camera RL (Isaac-Cartpole-RGB-Camera, 1024 envs, Isaac's skrl config) | 12.7 K env-steps/s including training (1 spp); 8.8 K at the physical tier-2 default, return 85.3 vs 86.3 for tier 0 | RTX 4090: 32 K (published, throughput only) | [§2](docs/PARITY.md#2-platform-capabilities-the-workstreams-with-the-tests-behind-them) |

Isaac's numbers come from one seed each; MuJoCo's contact model is not PhysX's, and the differences that remain are
listed in [`docs/PARITY.md` §3](docs/PARITY.md#3-what-is-disproved-missing-or-cannot-be-tested-here) and the gap
ledger. [`docs/STATUS.md`](docs/STATUS.md) is the state against the original plan.

## Default contact model and how to switch it off

Since 2026-09-26 the G1 task's default contacts are **elliptic friction cones** (the exact Coulomb cone, as PhysX
uses) with impratio 10 and a Newton iteration cap of 20, on top of the impact-stiffened 10 ms preset with hard joint
limits (`contact_cfg="recommended"` = `tau10_impact_hardlimits_ellip10`). This is the most PhysX-faithful setting
measured (impact forces within 8 % of PhysX's instead of 70 %), and it costs **1.6× on the full PPO loop** relative to
pyramidal cones. The owner chose fidelity over throughput on the reasoning that the M4 Max's compute deficit against an
L4 is inherent anyway. Known counter-signal: a PhysX-trained checkpoint falls 7–10 % more often under it (within one
seed's standard deviation; cause not established). To run the previous default, pass
`contact_cfg="recommended_pyramidal"` (or `--contact_cfg recommended_pyramidal`); `"default"` gives MuJoCo's own
contacts. Every number in this README names the setting it was measured under. Comparisons against Isaac Lab 3.0's
Newton / MuJoCo Warp backend use `solver_cfg="isaaclab3*"`, which imposes Isaac's own pyramidal cone regardless of the
contact preset.

## What was built where

MetalSim spans this repository and three forks. Each fork carries a `METALSIM_CHANGES.md` at its branch root
listing its commits, and keeps its upstream licence unchanged. "Head" is the branch head, which adds only
`METALSIM_CHANGES.md` to the code commit MetalSim's results were produced with.

### MetalSim (this repository)

[github.com/pulipakaa24/MetalSim](https://github.com/pulipakaa24/MetalSim), branch `main`. MIT licence.

| module | what it does |
|---|---|
| `metalsim/interop` | Warp ↔ Metal ↔ PyTorch MPS: zero-copy tensors over Warp and renderer buffers, cross-queue ordering with shared events, a native bridge (`metalsim/native/torch_metal_bridge.mm`) |
| `metalsim/physics` | `BatchSim`: batched MuJoCo Warp on Metal with graph replay and per-world model fields; contact-tuning and solver presets; deformables (MuJoCo Warp flex, XPBD cloth and cables, a port of PhysX's particle cloth); the archived Newton XPBD backend |
| `metalsim/render` | tier 0 raster, tier 1 hybrid ray tracing (shadows, AO, reflections), tier 2 path tracer with OmniPBR / UsdPreviewSurface BRDFs, USD light units, textured DomeLight HDR environment maps (importance-sampled, MIS), RTX's display transform and Open Image Denoise; RTX-parity presets |
| `metalsim/sensors` | lidar (beam divergence, multi-return, intensity), radar-lite and height scan on Metal ray tracing; Isaac Lab's `ContactSensor` as Warp kernels |
| `metalsim/scene` | USD → MuJoCo loader (lossless and generic UsdPhysics, fixes for NVIDIA asset conventions), MJCF/URDF → USD, MaterialX flattening |
| `metalsim/replicator` | annotators, randomizers, Isaac Lab event terms, and writers including Replicator's `BasicWriter` layout |
| `metalsim/learn` | Isaac Lab tasks (G1 velocity flat / rough, Isaac Lab 3.0 variants, cartpole state and RGB camera), Isaac's terrain generator, PPO with the rollout in Warp, pixel PPO on MPS with custom Metal gradient kernels, rsl_rl adapter, pre-flight checks, anomaly monitor, lidar navigation, SO-101 lift, TRON1 wheel-foot |
| `metalsim/parity` | fidelity protocol against an Isaac recording (replay, metrics, momentum-based contact impulses, side-by-side videos, policy export) and the Isaac-side scripts (below) |
| `metalsim/bench`, `metalsim/tools` | benchmark suite with compute-normalized figures, render fidelity metrics, cross-world race finder |
| `metalsim/tron1` | LimX TRON1 simulator with a hardware realism layer, controllers and the robot bridge ([`docs/TRON1.md`](docs/TRON1.md)) |
| `tests/`, `scripts/` | the test suite; setup scripts for the forks, the GPU job queue (`scripts/gpu_run.sh`), diagnostics and gallery scripts |
| `patches/` | exports of the fork commits, so each fork can be rebuilt from its upstream (Apache-2.0, see the notices) |

### Warp fork

[github.com/pulipakaa24/warp](https://github.com/pulipakaa24/warp), branch
[`metalsim`](https://github.com/pulipakaa24/warp/tree/metalsim), head
[`9abceff9`](https://github.com/pulipakaa24/warp/commit/9abceff9) (2026-09-26; the results before that day's throughput work were
produced at [`4127c48`](https://github.com/pulipakaa24/warp/commit/4127c4818334080de6f3d23aef522e582ff6a2a7), code
[`f194006`](https://github.com/pulipakaa24/warp/commit/f194006a4cf197fda93633252af81bc3f57f5561)). Base: innate-inc/warp
`ce15f6b`, NVIDIA Warp with innate-inc's Metal backend. Apache-2.0.

- Interop entry points: Metal device, queue and buffer handles and foreign event wait / signal (`b39fee1`).
- Captured graphs replayed through Metal indirect command buffers, and runtime counters of host waits, flushes and
  dispatches (`b1ec949`; `WP_METAL_ICB=0` disables).
- Deferred frees released per completed command buffer, with `WP_METAL_INFLIGHT` and `WP_METAL_ICB_BATCH` knobs
  (`7b5c828`; eager loops at 4096 worlds had run out of GPU memory).
- Fixed-size arrays (`wp.zeros` inside kernels) on Metal (`786cdae`).
- `segmented_sort_pairs` inside a graph capture (`9dcb140`, from `metalsim-flex`).
- Register-tile Cholesky bound configurable, `warp.config.metal_register_cholesky_max` (default 40; MetalSim sets 48,
  which puts the 43-dof G1 on the register path) (`b9557cb`).
- `nextafterf` in the Metal kernel runtime, bit-exact (Newton VBD) (`9050cb5`).
- Register triangular solves in `tile_cholesky_solve` (`27e63fd1`; solve 0.52 → 0.34 ms at n = 43, +5.2 % env step,
  summation order only); in-place rank-1 Cholesky updates (`e29950ee`); compact register layout and rolled Cholesky
  variants kept off by default (measured slower).

Branch [`metalsim-flex`](https://github.com/pulipakaa24/warp/tree/metalsim-flex), head
[`01ccf4a`](https://github.com/pulipakaa24/warp/commit/01ccf4ad687c0ebcdee139ee1131fbe2ad97bf51) (code `9dcb140`): the
first four commits plus the graph-capture sort; merged into `metalsim`.

### MuJoCo Warp fork

[github.com/pulipakaa24/mujoco_warp](https://github.com/pulipakaa24/mujoco_warp), branch
[`metalsim`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim), head
[`0a9de8e`](https://github.com/pulipakaa24/mujoco_warp/commit/0a9de8e) (2026-09-26;
the rigid-body results before the elliptic-cone work were produced at
[`07a51a6`](https://github.com/pulipakaa24/mujoco_warp/commit/07a51a63bce527d2d26b363a32c52765df4412fc) / code
[`8fbf965`](https://github.com/pulipakaa24/mujoco_warp/commit/8fbf965acb630aa32573dda802404d519c843a99)). Base: Google
DeepMind's MuJoCo Warp v3.14.0 (`88af9cc`) plus the Metal device patch by David Dobas
([DavidDobas/mujoco_warp#1](https://github.com/DavidDobas/mujoco_warp/pull/1)), branch
[`metal`](https://github.com/pulipakaa24/mujoco_warp/tree/metal) (`8ce5bb0`). Apache-2.0.

- Heightfield contacts against each prism's top plane per triangle (`f2716b4`); upstream gave inverted normals and
  launched robots on rough terrain.
- Plane-to-convex contacts with MuJoCo C's contact set (4 contacts on a tilted foot where upstream kept 2),
  switchable back with `MJW_PLANE_CONVEX=legacy` (`284dcd1`, `a8e6485`).
- Sparse L'DL factor and solve with one world per thread on Metal; no-op gravity-compensation and tendon-damping
  launches skipped (`ad22120`).
- The Newton solver's incremental Hessian update fused into the register Cholesky launch (`ccfaffb`;
  `MJW_METAL_FUSE_H_CHOLESKY=0` restores it), and a `MJW_METAL_DENSE_CHOL_MAX` knob (`1791414`).
- Nine flex (deformable) commits that make flex contacts match MuJoCo C: per-pair contact cap and selection,
  box-triangle and cable-capsule contacts, element ids of later flexes, active-layer volume contacts, mesh normals,
  flex equality rows in C's order, device-side sorts on Metal (`b2e9ea5` … `dfa5d30`, merged in `8fbf965`).
- Elliptic-cone Newton Hessian on Metal without capacity-sized launches or one-lane groups (`c301880`; modes
  `MJW_JTCJ_MODE=world2|world|contact|capacity`, `MJW_JTDAJ_ELLIPTIC_LANES`): Go2 elliptic 240 K → 580 K physics
  steps/s, the G1 task 11 K → 55 K env-steps/s, physics unchanged to float noise (`docs/research/elliptic_cones_2026-09-25.md`).
- Elliptic cones on MuJoCo C's incremental Newton structure (`a33c731`, `b895bff`; `MJW_ELLIPTIC_INCREMENTAL=2` default):
  the G1 task's elliptic premium 1.42× → 1.21× over pyramidal, Go2 / humanoid / SO-101 +4–7 %, physics unchanged.
- The five per-iteration Newton launches fused into one 32-lane kernel (`dd42c23`, +3.9 % env step, arithmetic
  unchanged); lane-parallel and chain-parallel sparse L'DL variants kept off by default (`7245794`, `cdc4fd4`, measured
  slower); the elliptic cone term as rank-1 factor updates kept off (`4ef3da3`, 5–24 % slower).
- Per-model unrolled register L'DL factorization of M (`f4276b0`, `MJW_METAL_LDL_UNROLLED`, default on): factor 0.453 → 0.314 ms
  per 4096 worlds, physics +3.6 %, bitwise the serial kernel (fp contraction off).
- Heightfield plane contacts restricted to meshes; primitives take upstream's GJK/EPA again (`b630530`; the fork's
  plane path missed contacts of primitives on heightfields smaller than the box; `HFIELD_PLANE_CONTACTS_PRIMITIVES=True`
  archives the old form).

Branch [`metalsim-flex`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim-flex), head
[`fc1d5ee`](https://github.com/pulipakaa24/mujoco_warp/commit/fc1d5eef8f9045309087bc822f36c1c62a7214a2) (code
`64a1ea5`): the contact commits and the nine flex commits, plus two not merged into `metalsim`: implicit
(backward-Euler) flex elasticity damping (`c013067`) and the configurable per-pair flex contact cap
`FLEX_MAXCONPAIR` (`64a1ea5`). `scripts/setup_flex.sh` installs it. The flex fixes proposed upstream are in
[`scripts/diagnostics/deformable/UPSTREAM.md`](scripts/diagnostics/deformable/UPSTREAM.md).

### Newton fork

[github.com/pulipakaa24/newton](https://github.com/pulipakaa24/newton), branch
[`metalsim`](https://github.com/pulipakaa24/newton/tree/metalsim), head
[`fb95e09`](https://github.com/pulipakaa24/newton/commit/fb95e095355f15cfb576d3022d44a34de23fa516) (code
[`90e2332`](https://github.com/pulipakaa24/newton/commit/90e23324521fa86710e483e01d02e80d1dac0060)). Base:
newton-physics/newton `4545802`. Apache-2.0.

- XPBD solver fixes, counted as six defects in the gap ledger: revolute angles crossing ±π (`f844a4e`), inconsistent
  joint relaxation and in-solver PD drives whose stiffness does not depend on the iteration count (`fda6658`), joint
  colouring and exact per-body contact forces (`9f626ce`), PD damping on light links (`9b0901d`), and drive-row
  restructuring and fast paths (`02799d8`, `6476ac4`, `fcc1505`, `742a775`, `242eeda`, `90e2332`).
- Filed upstream: issues [#4313](https://github.com/newton-physics/newton/issues/4313),
  [#4314](https://github.com/newton-physics/newton/issues/4314),
  [#4315](https://github.com/newton-physics/newton/issues/4315) and pull requests
  [#4316](https://github.com/newton-physics/newton/pull/4316),
  [#4317](https://github.com/newton-physics/newton/pull/4317),
  [#4318](https://github.com/newton-physics/newton/pull/4318). Status: open, awaiting the contributor licence
  agreement and maintainer approval ([`FILED.md`](scripts/diagnostics/newton_upstream/FILED.md)).

MetalSim's Newton XPBD engine is archived as experimental: it trains about 1.6× faster than MuJoCo Warp on the G1,
but its joints drift and PhysX-trained policies over-travel by 14–23 % in it, so it is not used for parity claims
([GAPS: Newton XPBD archived](docs/GAPS.md#newton-xpbd-archived-as-an-experimental-option-2026-09-25),
[PARITY §2.1](docs/PARITY.md#21-contact-model-what-mujoco-can-do-and-newton-xpbd-on-metal-measured-2026-09-24-newton-xpbd-archived-as-experimental-on-2026-09-25-see-gapsmd)).

### Isaac-side reference scripts

[`metalsim/parity/isaac_side/`](metalsim/parity/isaac_side) runs inside Isaac Sim on an NVIDIA L4 VM and records
what MetalSim is compared against. Reference versions: Isaac Sim 5.1 + Isaac Lab 2.3.2 (tag `v2.3.2`, PhysX), and
Isaac Sim 6.1 + Isaac Lab 3.0.0-EA (tag `v3.0.0-EA`, PhysX and Newton / MuJoCo Warp).

- `record_g1.py`: the fidelity protocol on Isaac-Velocity-Flat-G1 (3 s PD hold, 5 s seeded random joint targets,
  1 m drop): joint states, root pose, torques, contact forces and RTX frames (real-time or path-traced).
- `play_policy.py`: rolls Isaac's and MetalSim's checkpoints in Isaac, flat and at fixed rough-terrain levels, for
  transfer and side-by-side videos.
- `record_deformables.py` (5.1) and `record_deformables_il3.py` (3.0, PhysX and Newton VBD, mesh-resolution
  sweep), with `make_tetmeshes.py`: cloth, rope and soft-cube recordings.
- `stage*.sh`: Isaac's own benchmark and rsl_rl training runs (flat and rough), recordings and checkpoint playback.
- `isaaclab3/`: installation of Isaac Lab 3.0.0-EA next to 2.3.2, smoke test, runtime benchmark, fidelity protocol and
  training on both backends, and tools to summarize and compare the recordings.

Recordings, logs and benchmark JSON are under `runs/parity/`, `runs/parity3/` and `runs/deformable/`.

### Research notes

[`docs/research/`](docs/research), one question each:

- [`pytorch_mps_interop_2026-09-22.md`](docs/research/pytorch_mps_interop_2026-09-22.md): zero-copy and ordering between PyTorch MPS, MLX and foreign Metal buffers.
- [`metal_cnn_update_2026-09-24.md`](docs/research/metal_cnn_update_2026-09-24.md): why convolution backward is slow on MPS for camera RL, and the Metal kernels that fixed it.
- [`mjwarp_throughput_2026-09-25.md`](docs/research/mjwarp_throughput_2026-09-25.md): MuJoCo Warp settings and techniques for the G1 training loop, applied without changing results.
- [`mujoco_contact_vs_physx_2026-09-25.md`](docs/research/mujoco_contact_vs_physx_2026-09-25.md): MuJoCo contact and limit settings that approach PhysX on the G1.
- [`contact_discrepancies_2026-09-25.md`](docs/research/contact_discrepancies_2026-09-25.md): the impact-peak (a reporting artefact) and feet air-time (a policy difference) gaps.
- [`terrain_walls_2026-09-25.md`](docs/research/terrain_walls_2026-09-25.md): collision surfaces for the vertical walls of Isaac's rough terrain.
- [`rendering_vs_rtx_2026-09-25.md`](docs/research/rendering_vs_rtx_2026-09-25.md): what RTX computes on the parity scene and how tier 2 was matched to it.
- [`materialx_2026-09-25.md`](docs/research/materialx_2026-09-25.md): MaterialX in Isaac Sim and what the renderer's material table can carry.
- [`replicator_2026-09-25.md`](docs/research/replicator_2026-09-25.md): Replicator annotators, randomizers and writers against MetalSim's.
- [`deformables_2026-09-25.md`](docs/research/deformables_2026-09-25.md): cloth, cables and soft volumes through MuJoCo Warp's flex path.
- [`physx_deformables_port_2026-09-25.md`](docs/research/physx_deformables_port_2026-09-25.md): PhysX 5's deformable solvers and what a Warp port would take.
- [`newton_vbd_metal_2026-09-25.md`](docs/research/newton_vbd_metal_2026-09-25.md): Newton VBD and the coupled solvers on Metal (Isaac Lab 3.0's deformable path).
- [`isaaclab3_reference_2026-09-25.md`](docs/research/isaaclab3_reference_2026-09-25.md): making Isaac Lab 3.0-EA the reference on both backends.
- [`isaaclab3_like_for_like_2026-09-25.md`](docs/research/isaaclab3_like_for_like_2026-09-25.md): the 3.0 G1 task and Isaac's own MuJoCo Warp settings for a like-for-like comparison.
- [`closed_loops_2026-09-25.md`](docs/research/closed_loops_2026-09-25.md): closed kinematic loops, Newton Kamino on Metal vs MuJoCo Warp's soft closure.

## Install

Apple silicon Mac, macOS with the Xcode Command Line Tools, Python 3.12 or newer:

```
git clone https://github.com/pulipakaa24/MetalSim && cd MetalSim
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
scripts/setup_warp.sh
pip install -e "git+https://github.com/pulipakaa24/mujoco_warp@metalsim#egg=mujoco-warp"
scripts/fetch_isaac_assets.sh              # NVIDIA's g1_minimal.usd, fetched from NVIDIA, not redistributed
pytest tests -q
```

Optional extras, first runs, bringing your own robot, the GPU job queue and the throughput / fidelity knobs are in
the user guide, [`docs/GUIDE.md`](docs/GUIDE.md). What landed when is in [`CHANGELOG.md`](CHANGELOG.md).

## Working rules

- Fidelity first: where an option trades fidelity to Isaac for speed, the faithful one is the default and the faster
  one stays available behind a flag, with both measured.
- Archive, never discard: every rejected option stays in the tree (flag, branch or documented setting), its
  measurements stay in `runs/`, and [`docs/DECISIONS.md`](docs/DECISIONS.md) records the numbers, the reason and how to
  re-enable it.
- Every number is labelled measured, published or estimated; the gap ledger, [`docs/GAPS.md`](docs/GAPS.md), lists
  what is closed and what is still open.

## Licence and notices

MetalSim is released under the [MIT licence](LICENSE), copyright 2026 Aditya Pulipaka. Third-party files included here
(Isaac Lab sources under `assets/isaac/`, MuJoCo Menagerie and Playground models, LimX TRON1 files) keep their own
licences, and the patches in `patches/` modify Apache-2.0 projects and are offered under Apache-2.0.
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists every component, its licence and how it is used. NVIDIA's
Isaac Sim assets are not redistributed.

## Acknowledgements and citation

MetalSim builds on [NVIDIA Warp](https://github.com/NVIDIA/warp) and innate-inc's Metal backend by David Dobas
([innate-inc/warp](https://github.com/innate-inc/warp)), [MuJoCo](https://github.com/google-deepmind/mujoco) and
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), [Newton](https://github.com/newton-physics/newton),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), [NVIDIA PhysX](https://github.com/NVIDIA-Omniverse/PhysX),
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
and [Open Image Denoise](https://github.com/RenderKit/oidn). If you use MetalSim, cite this repository
(Aditya Pulipaka, MetalSim, 2026, https://github.com/pulipakaa24/MetalSim) together with the upstream projects whose
components you use, following their own citation instructions.
