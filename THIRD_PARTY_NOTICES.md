# Third-party notices

MetalSim's own code is released under the MIT licence ([`LICENSE`](LICENSE), copyright 2026 Aditya Pulipaka).
This file lists every third-party component that appears in this repository or that MetalSim depends on, its
licence, where it appears and how it is used. It was produced by an audit of every tracked file and of the full
git history on 2026-09-25 (method at the end). Licence texts that the MIT file does not cover are in
[`licenses/`](licenses) or next to the files they cover.

Three kinds of relationship occur:

- **Included**: third-party files committed here, unchanged, with their licence and headers kept.
- **Derived**: MetalSim code ported or translated from third-party code; the file carries both licences in an
  SPDX header and names its source.
- **Dependency / fork**: installed by pip or by `scripts/setup_*.sh`, not redistributed here. MetalSim's changes to
  forked projects live in those forks under the upstream licence; `patches/` holds exports of them.

## Summary

| component | licence | relationship | where in this repository |
|---|---|---|---|
| NVIDIA Warp, via innate-inc/warp (Metal backend) | Apache-2.0 | fork, dependency; patches | `patches/warp/`, `patches/0001-Metal-fixed-size-arrays-*.patch`, `patches/warp-innate-metal-pending-frees.patch`, `scripts/setup_warp.sh` |
| MuJoCo Warp (Google DeepMind), incl. David Dobas's Metal device patch | Apache-2.0 | fork, dependency; patches | `patches/mujoco_warp/`, `patches/mujoco_warp-*.patch`, `scripts/setup_flex.sh` |
| MuJoCo (Google DeepMind) | Apache-2.0 | dependency (`mujoco`) | imported throughout |
| Newton (Newton project, Linux Foundation) | Apache-2.0 | fork, optional dependency | `metalsim/physics/newton_backend.py`, `scripts/setup_newton*.sh`, `scripts/diagnostics/newton_upstream/` |
| NVIDIA PhysX 5.6.1 | BSD-3-Clause | derived (one module, algorithm port) | `metalsim/physics/physx_cloth.py`; no PhysX source files |
| Isaac Lab (v2.3.2) source files | BSD-3-Clause | included (headers kept); derived (terrain port) | `assets/isaac/**/*.py`, `assets/isaac/*.yaml`, `metalsim/learn/isaac_terrain.py` |
| Isaac Sim assets (`g1_minimal.usd`) | NVIDIA asset terms | fetched, **not redistributed** | `scripts/fetch_isaac_assets.sh`; `.gitignore` |
| Isaac Sim / Isaac Lab (runtime, on the reference VM) | Isaac Sim licence / BSD-3-Clause | used on an NVIDIA machine; recordings of its outputs committed | `metalsim/parity/isaac_side/`, `runs/parity*/`, `docs/gallery/` |
| MuJoCo Menagerie: Unitree G1 | BSD-3-Clause (Unitree) | included | `assets/menagerie/unitree_g1/` (with `LICENSE`) |
| MuJoCo Menagerie: The Robot Studio SO-101 | Apache-2.0 | included | `assets/so101/` (with `LICENSE`) |
| MuJoCo Playground (G1 joystick task) | Apache-2.0 (G1 model: BSD-3-Clause, Unitree) | included | `assets/playground/` |
| LimX Dynamics TRON1 robot description, policies, deploy code, SDK types | Apache-2.0 | included; derived | `assets/tron1/` (with both `LICENSE-*`), `metalsim/tron1/limx_policy.py`, `metalsim/tron1/sdk.py` |
| rsl_rl (`rsl-rl-lib`, ETH Zurich and NVIDIA) | BSD-3-Clause | optional dependency | `metalsim/learn/train_g1_rslrl.py`, `metalsim/learn/rslrl_adapter.py` |
| skrl | MIT | not used; Isaac Lab's skrl agent config reproduced | `assets/isaac/cartpole_skrl_camera_ppo_cfg.yaml`, `metalsim/learn/ppo.py` |
| Intel Open Image Denoise, via `pyoidn` | Apache-2.0 (OIDN), MIT (`pyoidn`) | optional runtime library, loaded with ctypes | `metalsim/render/denoise.py` |
| MaterialX (ASWF) | Apache-2.0 | optional runtime import | `metalsim/scene/materialx.py` |
| OpenUSD (`usd-core`, Pixar) | Tomorrow Open Source Technology License 1.0 | dependency | `metalsim/scene/` |
| PyTorch | BSD-3-Clause (and others, per its wheel) | dependency | `metalsim/interop/`, `metalsim/native/torch_metal_bridge.mm`, learners |
| MLX (Apple) | MIT | not a dependency; one diagnostic script | `scripts/diagnostics/camera_update_profile.py` |
| imageio-ffmpeg | BSD-2-Clause (the ffmpeg binary it downloads has its own licence) | optional dependency (`parity` extra) | video scripts under `scripts/gallery/` |
| Poly Haven HDRIs and textures | CC0 1.0 (public domain) | fetched, **not committed** (`.gitignore`); one rendered frame in the gallery | `scripts/fetch_polyhaven_assets.sh` -> `assets/polyhaven/`; `docs/gallery/g1_hdri_tier2.png` |
| other Python dependencies (numpy, scipy, pyobjc, scikit-image, lpips, flip-evaluator, trimesh, gymnasium, stable-baselines3, onnxruntime, tensordict, pillow, imageio, pytest) | BSD / MIT family | dependencies (`pyproject.toml`) | imported |
| published algorithms (XPBD, a-trous / SVGF, ACES fit, GGX fit, Philox, stable PD, PPO details) | not applicable (no code copied) | reimplemented from the papers | see "Algorithms from publications" |

Nothing in the tree or its history is an NVIDIA binary, a USD file, an Omniverse / Kit component or PhysX source.

Trademarks: NVIDIA, Isaac, Isaac Sim, Isaac Lab, PhysX, Omniverse and RTX are trademarks of NVIDIA Corporation;
MuJoCo is a trademark of Google DeepMind; Apple, Metal and Apple silicon are trademarks of Apple Inc.; Unitree, LimX
and TRON1 belong to their owners. They are used only to name the software and hardware MetalSim is compared with or
runs on. MetalSim is an independent project, not affiliated with or endorsed by any of them.

## Details

### NVIDIA Warp and the innate-inc Metal backend

- Upstream: [NVIDIA/warp](https://github.com/NVIDIA/warp), Apache-2.0, copyright NVIDIA Corporation & Affiliates.
  Metal backend: [innate-inc/warp](https://github.com/innate-inc/warp), where David Dobas develops it (also published
  as [DavidDobas/warp-metal](https://github.com/DavidDobas/warp-metal)), Apache-2.0, base commit `ce15f6bb`; not
  affiliated with NVIDIA.
- MetalSim's fork: [pulipakaa24/warp](https://github.com/pulipakaa24/warp), branches `metalsim` and
  `metalsim-flex`; `LICENSE.md` and `licenses/` are byte-identical to NVIDIA/warp's. The fork's
  `METALSIM_CHANGES.md` lists the changes.
- In this repository: `patches/warp/0001-0007` (the fork's seven code commits), and two older single-commit
  exports, `patches/0001-Metal-fixed-size-arrays-wp.zeros-in-kernels-thread-q.patch` and
  `patches/warp-innate-metal-pending-frees.patch`. **Each patch modifies Apache-2.0 code and is offered under the
  Apache License 2.0** ([`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)), not under MIT.
- Use: installed by `scripts/setup_warp.sh` (clones the fork, or innate-inc/warp plus the patches); `pyproject.toml`
  declares `warp-lang`. No Warp source is copied into `metalsim/`.

### MuJoCo Warp and MuJoCo

- Upstream: [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) v3.14.0 (`88af9cc`),
  Apache-2.0, copyright The Newton Developers (MuJoCo Warp is developed by Google DeepMind and NVIDIA). The Metal device support is David Dobas's patch
  ([DavidDobas/mujoco_warp#1](https://github.com/DavidDobas/mujoco_warp/pull/1), Apache-2.0).
- MetalSim's fork: [pulipakaa24/mujoco_warp](https://github.com/pulipakaa24/mujoco_warp), branches `metal`,
  `metalsim` and `metalsim-flex`; `LICENSE` is byte-identical to upstream's.
- In this repository: `patches/mujoco_warp/0001-0016` (0001 is David Dobas's patch; 0002-0016 are MetalSim's) and
  the older exports `patches/mujoco_warp-hfield-plane-contacts.patch` and
  `patches/mujoco_warp-plane-convex-c-contact-set.patch`. **Each patch modifies Apache-2.0 code and is offered
  under the Apache License 2.0.**
- MuJoCo ([google-deepmind/mujoco](https://github.com/google-deepmind/mujoco), Apache-2.0) is a pip dependency.
  Some comments and research notes cite MuJoCo C function names and file names; no MuJoCo source is copied.

### Newton

- Upstream: [newton-physics/newton](https://github.com/newton-physics/newton), Apache-2.0, copyright The Newton
  Developers.
- MetalSim's fork: [pulipakaa24/newton](https://github.com/pulipakaa24/newton), branch `metalsim`; `LICENSE.md`
  byte-identical to upstream's. Pull-request branches `fix/xpbd-revolute-angle-wrap`, `fix/xpbd-joint-relaxation`,
  `feat/xpbd-pd-joint-drive` (upstream PRs #4316-#4318).
- In this repository: `metalsim/physics/newton_backend.py` calls Newton's API (MetalSim code);
  `scripts/diagnostics/newton_upstream/` holds the issue and pull-request texts filed upstream: MetalSim-written
  reproducers against Newton's public API and permalinks into Newton's source, no copied source. The optional
  `newton` extra installs upstream Newton.

### NVIDIA PhysX

- [NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) 5.6.1, BSD-3-Clause, copyright NVIDIA
  Corporation ([`licenses/BSD-3-Clause-PhysX.txt`](licenses/BSD-3-Clause-PhysX.txt)).
- The PhysX source was read (in a local checkout outside this repository) to document what Isaac Sim computes.
  **No PhysX source file is in the repository or its history** (checked by path and by content search).
- One module is derived: `metalsim/physics/physx_cloth.py` re-expresses the algorithm of PhysX's GPU particle-cloth
  path (pre-integration, spring partitions, TGS iterations, contact and finalisation steps, with PhysX's constants)
  as Warp kernels, citing the PhysX function for each step. It is treated as a derivative work: its header carries
  `SPDX-License-Identifier: MIT AND BSD-3-Clause` and the PhysX copyright notice, and PhysX's licence text is in
  `licenses/`. `docs/research/physx_deformables_port_2026-09-25.md` describes PhysX's formulation in prose with file
  and function references; it contains no code listings.

### Isaac Lab

- [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab), BSD-3-Clause, copyright The Isaac Lab Project
  Developers ([`licenses/BSD-3-Clause-IsaacLab.txt`](licenses/BSD-3-Clause-IsaacLab.txt)).
- **Included, unchanged, headers kept**: `assets/isaac/` (task, reward, PPO and asset configs of the G1 velocity and
  cartpole tasks, the rough-terrain generator package, the contact sensor, `benchmark_non_rl.py`, parts of the
  `isaaclab` API used as references). Every file is byte-identical (by git blob hash) to a file of Isaac Lab
  v2.3.2, some under a shorter name, and keeps its `Copyright (c) ... The Isaac Lab Project Developers ...
  SPDX-License-Identifier: BSD-3-Clause` header. Renamed files and their v2.3.2 paths:
  `g1_asset_cfg.py` = `source/isaaclab_assets/isaaclab_assets/robots/unitree.py`;
  `g1_velocity_env_cfg.py` = `.../manager_based/locomotion/velocity/velocity_env_cfg.py`;
  `g1_flat_env_cfg.py`, `g1_rough_env_cfg.py`, `g1_rsl_rl_ppo_cfg.py` = `.../velocity/config/g1/{flat_env_cfg,rough_env_cfg,agents/rsl_rl_ppo_cfg}.py`;
  `g1_rewards.py` = `.../velocity/mdp/rewards.py`; `cartpole.py` = `source/isaaclab_assets/isaaclab_assets/robots/cartpole.py`;
  `cartpole_env.py`, `cartpole_camera_env.py` = `.../direct/cartpole/`; `cartpole_rsl_rl_ppo_cfg.py`,
  `cartpole_skrl_camera_ppo_cfg.yaml` = `.../direct/cartpole/agents/{rsl_rl_ppo_cfg.py,skrl_camera_ppo_cfg.yaml}`;
  `benchmark_non_rl.py`, `utils.py` = `scripts/benchmarks/`; `api/train.py`, `api/play.py` =
  `scripts/reinforcement_learning/rsl_rl/`; `api/camera_cfg.py`, `api/simulation_cfg.py`, `api/mdp_observations.py` =
  `source/isaaclab/isaaclab/{sensors/camera/camera_cfg,sim/simulation_cfg,envs/mdp/observations}.py`;
  `sensors/contact_sensor/` and `terrains/` keep their v2.3.2 names under `source/isaaclab/isaaclab/`.
  `assets/isaac/api/observations.py` is an empty placeholder.
- Use: the tests read these files as the oracle (`tests/test_g1_parity.py` parses `G1_CFG` with `ast`;
  `tests/isaac_terrain_ref.py` executes the terrain package on the CPU with the `isaaclab` utilities it needs
  stubbed).
- **Derived**: `metalsim/learn/isaac_terrain.py` ports `isaaclab.terrains.TerrainGenerator` and the sub-terrain
  functions to numpy; header `SPDX-License-Identifier: MIT AND BSD-3-Clause` with Isaac Lab's notice. Other modules
  (`metalsim/learn/g1_velocity.py`, `g1_il3.py`, `metalsim/sensors/contact.py`, `metalsim/replicator/events.py`)
  reimplement Isaac Lab's documented term formulas and semantics as Warp kernels and cite the Isaac Lab function they
  match; they are MetalSim code.
- `metalsim/parity/isaac_side/` holds MetalSim scripts that import Isaac Lab and run inside Isaac Sim on the
  reference VM; they are MetalSim code (MIT) and contain no Isaac Lab source.

### Isaac Sim assets and outputs

- `assets/isaac/G1/g1_minimal.usd` (Isaac Lab's G1 asset, from NVIDIA's public asset store) is under NVIDIA's asset
  terms and is **not redistributed**: `scripts/fetch_isaac_assets.sh` downloads it and checks its md5, and
  `.gitignore` excludes `assets/isaac/**/*.usd`. The full history contains no `.usd`, `.usda`, `.usdc`, `.usdz` or
  `.mdl` file.
- `assets/isaac/G1/g1_minimal_from_usd.xml`, an MJCF conversion of that USD (kinematics, inertials and the three
  collision boxes), was committed on 2026-09-22 (`1359a52`) and removed from the tree in this audit because it is a
  derivative of NVIDIA's asset and nothing referenced it. It remains in the history of `1359a52` and later commits up
  to the removal (see "History" below). It can be regenerated locally from the fetched USD with
  `metalsim.scene.usd_to_mjcf.load_usd(...).to_xml()`; `.gitignore` now excludes `assets/isaac/**/*_from_usd.xml`.
- The robot model inside NVIDIA's asset originates from Unitree's G1 description
  ([unitree_ros](https://github.com/unitreerobotics/unitree_ros), BSD-3-Clause).
- Recorded outputs of Isaac Sim runs made by the author on a rented NVIDIA L4 (joint trajectories, contact forces,
  training logs, benchmark JSON, RTX frames and videos) are committed under `runs/parity/`, `runs/parity3/`,
  `runs/il3/`, `runs/deformable/` and `docs/gallery/` (e.g. `g1_stage_isaac_*.mp4`, `fidelity_*.png`). They are
  measurements and renders produced by running the software, not NVIDIA files; the renders depict the G1 asset.
- The OmniPBR and UsdPreviewSurface BRDFs in `metalsim/render/shaders/pathtrace.metal` are reimplemented from their
  published definitions (NVIDIA's MDL documentation, the UsdPreviewSurface specification); no MDL file is included.
  `metalsim/replicator/basic_writer.py` reproduces the on-disk layout of Replicator's `BasicWriter` (file names and
  formats) for interoperability; no `omni.replicator` code is included.

### MuJoCo Menagerie and MuJoCo Playground

- [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (each model under its own
  licence), files identical to Menagerie `822c2d8`:
  - `assets/menagerie/unitree_g1/` (`g1.xml`, `g1_with_hands.xml`, `README.md`, `LICENSE`): Unitree G1, BSD-3-Clause,
    copyright HangZhou YuShu Technology Co., Ltd. (Unitree Robotics). Meshes are not included.
  - `assets/so101/` (`so101.xml`, `scene.xml`, `scene_box.xml`, STL meshes, `README.md`, `CHANGELOG.md`, `LICENSE`):
    The Robot Studio SO-101 ([SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)), Apache-2.0.
    `scene_box_rl.xml` and `base_cam_preview.png` are MetalSim additions.
- Tests optionally read other Menagerie models from a local clone (`upstream/mujoco_menagerie`, not committed).
- [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground), Apache-2.0, copyright
  Google LLC: `assets/playground/base.py`, `g1_constants.py`, `joystick.py` (headers kept) and
  `g1_mjx_feetonly.xml` (Playground's MJX variant of the Menagerie G1, whose model is Unitree's, BSD-3-Clause). Kept
  as reference for the G1 settings MuJoCo's own stack uses; no MetalSim module imports them.

### LimX Dynamics TRON1

- Apache-2.0, copyright LimX Dynamics:
  - `assets/tron1/WF_TRON1A/` (URDF, MJCF, STL meshes) from
    [limxdynamics/robot-description](https://github.com/limxdynamics/robot-description)
    (licence: `assets/tron1/LICENSE-robot-description`);
  - `assets/tron1/limx_policy/WF_TRON1A/` (`params.yaml`, ONNX encoder and policy networks, Isaac Gym and Isaac Lab
    variants) from [limxdynamics/tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python)
    (licence: `assets/tron1/LICENSE-rl-deploy-python`).
- **Derived**: `metalsim/tron1/limx_policy.py` ports that repository's `controllers/WheelfootController.py`
  (commit `035e4c9`); header `SPDX-License-Identifier: MIT AND Apache-2.0`, and the docstring states the changes.
  `metalsim/tron1/sdk.py` mirrors the message field names, units and motor order of
  [limxdynamics/limxsdk-lowlevel](https://github.com/limxdynamics/limxsdk-lowlevel) (Apache-2.0) `datatypes.h`.
  The `limxsdk` binaries are not included; `metalsim/tron1/limx_bridge.py` imports them when installed on the robot.
- `assets/tron1/scenes/spatial_station.npz` is a 562 x 596 heightfield computed by `metalsim/tron1/scene.py` from a
  scanned mesh (`spatial_station.ply`, not committed) of a physical space. The repository does not record who made
  the scan; see "Items for the author to confirm" below.

### Learners and RL libraries

- rsl_rl ([leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl), `rsl-rl-lib` 3.1.2 on PyPI),
  BSD-3-Clause, copyright ETH Zurich and NVIDIA: optional dependency (`rslrl` extra), imported by
  `metalsim/learn/train_g1_rslrl.py`; `rslrl_adapter.py` implements its `VecEnv` interface.
  `metalsim/learn/ppo_warp.py` implements the same PPO formulation (clipped value loss, adaptive KL schedule,
  time-out bootstrapping) independently, with comments naming the rsl_rl behaviour it matches.
- skrl ([Toni-SM/skrl](https://github.com/Toni-SM/skrl), MIT) is not used. `metalsim/learn/ppo.py` reproduces the
  network and hyperparameters of Isaac Lab's skrl configuration (above).
- Stable-Baselines3 (MIT) and gymnasium (MIT): optional `learn` extra.

### Rendering, scene and media libraries

- Intel Open Image Denoise ([RenderKit/oidn](https://github.com/RenderKit/oidn), Apache-2.0), shipped by
  `pyoidn` ([Hyiker/pyoidn](https://github.com/Hyiker/pyoidn), MIT): `metalsim/render/denoise.py` loads
  `libOpenImageDenoise` with ctypes when installed; not a declared dependency.
- MaterialX ([AcademySoftwareFoundation/MaterialX](https://github.com/AcademySoftwareFoundation/MaterialX),
  Apache-2.0): imported by `metalsim/scene/materialx.py` to read `.mtlx` documents when installed.
- OpenUSD `usd-core` (Pixar, Tomorrow Open Source Technology License 1.0): dependency of `metalsim.scene`.
- imageio-ffmpeg (BSD-2-Clause): `parity` extra; it downloads an ffmpeg binary under ffmpeg's licence, which is not
  redistributed here.
- lpips (BSD-2-Clause), flip-evaluator (NVIDIA FLIP, BSD-3-Clause), scikit-image (BSD-3-Clause): `parity` extra.

### Poly Haven assets (environment maps and textures)

- [Poly Haven](https://polyhaven.com) publishes its HDRIs and textures under CC0 1.0 (public domain dedication;
  [polyhaven.com/license](https://polyhaven.com/license)). `scripts/fetch_polyhaven_assets.sh` downloads 13
  equirectangular 2k HDRIs and 5 textures into `assets/polyhaven/`, which is git-ignored: nothing is committed,
  each machine fetches its own copy. They are the environment maps of the tier-2 path tracer's dome-light
  lighting and of the replicator's per-episode environment randomization. `docs/gallery/g1_hdri_tier2.png` is a
  frame rendered under one of them (`lebombo`); CC0 requires no attribution, credit is given here anyway.

### PyTorch and MLX

- PyTorch (BSD-3-Clause and the licences listed in its wheel): dependency. `metalsim/native/torch_metal_bridge.mm`
  uses PyTorch's public C++ extension and MPS APIs. `metalsim/learn/isaac_terrain.py` reproduces the output stream
  of PyTorch's CUDA uniform generator (Philox4x32-10 via curand) in numpy so the terrain matches Isaac's on NVIDIA
  GPUs; it implements the published Philox algorithm and contains no PyTorch or CUDA source.
- MLX (MIT): only `scripts/diagnostics/camera_update_profile.py` imports it, for a measurement recorded in
  `docs/research/metal_cnn_update_2026-09-24.md`. It is not a dependency of the package.

### Algorithms from publications

Implemented from the papers, no code copied: XPBD (Macklin, Müller, Chentanez 2016; small-step XPBD, Macklin et al.
2019) in `metalsim/physics/deformable.py`; edge-avoiding a-trous wavelet filter (Dammertz et al., HPG 2010) and the
spatial pass of SVGF (Schied et al., HPG 2017) in `metalsim/render/shaders/denoise.metal` and `tier2.py`; the ACES
filmic curve fit (Narkowicz 2015) and the split-sum GGX albedo fit (Karis 2013) in `pathtrace.metal`; the Philox
counter-based generator (Salmon et al., SC 2011); stable PD control (Tan, Liu, Turk 2011) in `newton_backend.py`;
PPO implementation details (Huang et al. 2022) cited in `metalsim/learn/monitor.py`.

## History (committed then removed)

`git log --all` over the whole history (253 commits on `main`, no other branches) shows these paths that were
committed and later removed:

| path | status | rewrite needed? |
|---|---|---|
| `assets/isaac/G1/g1_minimal_from_usd.xml` | MJCF conversion of NVIDIA's `g1_minimal.usd` (numbers and three boxes; no meshes, no USD); in history from `1359a52` (2026-09-22), removed in this audit | Recommended if strict: it is the one derived NVIDIA-asset file in the public history. The underlying kinematic and inertial values originate from Unitree's BSD-3-Clause G1 description, so the practical exposure is small. Not rewritten (report only). |
| `orchard/**` | MetalSim's own code before the rename to `metalsim` | no |
| `patches/warp/0005-Metal-configurable-register-Cholesky-bound-*.patch` | own patch, renumbered | no |
| `runs/contact_tuning_bench_4096*.log` | own logs | no |
| `.DS_Store`, `MUJOCO_LOG.TXT` | macOS and MuJoCo droppings | no |

No USD, MDL, Omniverse, PhysX or other NVIDIA binary file appears anywhere in the history.

## Items for the author to confirm

- `assets/tron1/scenes/spatial_station.npz`: confirm the source scan (`spatial_station.ply`, `Spatial-Station.spz`)
  is yours or licensed for redistribution. The heightfield keeps only floor and obstacle heights at 5 cm, no colour
  or texture; if the scan is someone else's, remove the file (and consider it in the history question above).
- `metalsim/replicator/basic_writer.py` and the OmniPBR BRDF were written from NVIDIA's documentation and observed
  outputs; the audit found no copied `omni.*` code, but it could not compare against the closed Kit sources.

## Audit method

1. Every tracked file (`git ls-files`, 1,313 files) classified by directory and type; every binary type
   (`.stl`, `.onnx`, `.npz`, `.png`, `.mp4`, `.pt`) traced to its producer.
2. All tracked text searched for `Copyright`, `SPDX`, `License`, `adapted from`, `based on`, `ported from`,
   `port of`, `verbatim`, `copied from`, `derived from`, `et al`, and for every URL in code and docstrings.
3. Every path ever committed (`git log --all --name-only`) compared with the current tree, and searched for
   USD / MDL / shared-library / PhysX / Omniverse files.
4. Included third-party files compared with their upstream checkouts (Menagerie `822c2d8`, Isaac Lab `v2.3.2`);
   fork `LICENSE` files compared by blob hash with upstream.
5. Licences of dependencies read from their PyPI metadata and GitHub.
