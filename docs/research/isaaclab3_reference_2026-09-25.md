# Isaac Lab 3.0 Early Access as the parity reference (research, 2026-09-25)

Question: what has to change to make Isaac Lab 3.0.0-EA the reference for MetalSim, on both of its
physics backends (Isaac Sim PhysX and Newton / MuJoCo Warp), so that MetalSim (MuJoCo Warp on Metal)
is compared like for like against Isaac's own MuJoCo Warp on NVIDIA hardware and against PhysX?

Sources are cited inline. "Source" means the v3.0.0-EA tag of `isaac-sim/IsaacLab`
(commit `ae37b028ea415c91ea2bc32609efcd759ed2b974`, 2026-09-16, `VERSION` = 3.0.0), read directly;
paths are relative to the repository root.

## 1. Release facts (v3.0.0-EA)

- Release page: https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA (published
  2026-09-16T22:34Z, tag `v3.0.0-EA` on branch `release/3.0.0`). Built for **Isaac Sim 6.1**,
  Python 3.12, PyTorch 2.11, Warp 1.16, Newton 1.5.2. General Availability targeted for the end of
  October 2026; `release/3.0.0` takes only fixes until then.
- Pins in `pyproject.toml` `[tool.isaaclab.versions]` and `[tool.uv].override-dependencies`:
  `isaacsim[all,extscache]==6.1.0.0`, `torch==2.11.0`, `torchvision==0.26.0`, `warp==1.16.0`,
  `newton[sim]==1.5.2`, `mujoco~=3.11.0`, **`mujoco-warp~=3.11.0`**, `rsl-rl-lib==5.4.1`
  (extra `rsl-rl`), `numpy>=2`.
- Three physics backends selected at launch with a Hydra-style token: `physics=isaacsim_physx`
  (Kit PhysX), `physics=ovphysx` (kit-less PhysX wheel), `physics=newton_mjwarp` (Newton with
  MuJoCo Warp); plus `newton_kamino` for tasks that define it. Renderers `renderer=isaacsim_rtx |
  ovrtx | newton_renderer`; visualizers `--viz newton_gl | newton_rtx | viser | rerun | kit | none`.
  Isaac Sim backends cannot be combined with OVPhysX / OVRTX (release notes, "Known limitations").
- **Breaking changes that affect our Isaac-side scripts** (release notes, "Core breaking changes"):
  - `train.py` / `play.py` and the "legacy benchmark scripts" (incl. `benchmark_non_rl.py`) are
    removed; use `isaaclab train --rl_library rsl_rl`, `isaaclab play`, `isaaclab benchmark
    runtime|startup|training|play`.
  - `--headless` is removed; use `--viz none`.
  - Quaternions are **XYZW** everywhere (identity `(0, 0, 0, 1)`); our `record_g1.py` writes
    `root_quat_w` and builds the camera rotation in WXYZ, both must be converted.
  - `.data.*` returns `ProxyArray`; use `.torch` / `.warp`.
  - `write_root_pose_to_sim(data, env_ids)` removed; `write_root_pose_to_sim_index(root_pose=...,
    env_ids=...)` / `_mask(...)`.
  - Actuators under `robot.actuators` (`applied_effort`); `effort_limit_sim` → `joint_effort_limit`.
  - Contact sensor: total / normal / friction forces are now separate properties.
  - `isaaclab.sh` is deprecated (still works in EA); `uv run isaaclab ...` is the recommended path.

## 2. Installing Isaac Sim 6.1 + Isaac Lab 3.0.0-EA (pip path)

Source: `docs/source/setup/installation/index.rst` and the command template in
`docs/_extensions/isaaclab_docs.py` (`_quickstart_isaacsim`), published at
https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/installation/index.html:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git --branch <branch>   # we use --branch v3.0.0-EA
uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate
uv pip install --upgrade pip
uv pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com \
   --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
./isaaclab.sh -i            # or -i 'newton,rl[rsl-rl],visualizer[kit]' (selectors table on the same page)
```

Requirements: Python 3.12, Ubuntu 22.04+ (badge: 24.04), GLIBC ≥ 2.35 for the pip wheels, ≥ 32 GB RAM,
≥ 16 GB VRAM; "Isaac Sim 5.1 and older are not supported". Driver: Isaac Lab recommends the latest
production branch, **580.95.05 or later** on Linux x86_64 (same page). The Isaac Sim requirements page
(https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html, "latest" at the time
of reading) lists Linux driver **595.58.03**, 50 GB SSD minimum, RTX-core GPUs only (A100/H100 not
supported). Our L4 VM runs 580.178.04: above Isaac Lab's floor, below Isaac Sim's listed driver; we
install without touching the driver (the 5.1 environment must stay intact) and record whether anything
fails. Non-interactive EULA: `OMNI_KIT_ACCEPT_EULA=yes`. First Kit launch can take > 10 min
(extension download).

## 3. How a task selects its backend

- Physics is a `PresetCfg` on `SimulationCfg.physics`; the CLI token `physics=<name>` picks one of
  its fields (`docs/source/concepts/backends_and_presets.rst`; https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/backends_and_presets.html).
  `PresetCfg` fields may also appear in other configs via `preset(default=..., newton_mjwarp=...)`
  (e.g. the PPO `max_iterations` below).
- For the velocity tasks (`source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`,
  `RoughPhysicsCfg`) the fields are `isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10*2**15)`,
  `ovphysx`, `physx = PhysxAutoCfg(...)`, `newton_mjwarp = NewtonCfg(...)`, `newton_kamino`, and
  **`default = newton_mjwarp`**. The G1 flat config sets `self.sim.physics.default = newton_mjwarp`
  explicitly. So for these tasks **Newton/MuJoCo Warp is the default**, not PhysX; PhysX has to be
  requested with `physics=isaacsim_physx` (Kit) or `physics=ovphysx`.

## 4. G1 velocity tasks in 3.0

Registration: `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/__init__.py`.

| 2.3.2 id | 3.0 id | env cfg | rsl_rl cfg | backends (docs env browser) |
|---|---|---|---|---|
| `Isaac-Velocity-Flat-G1-v0` | **`Isaac-Velocity-Flat-G1`** | `flat_env_cfg:G1FlatEnvCfg` | `G1FlatPPORunnerCfg` | isaacsim_physx, newton_kamino, newton_mjwarp, ovphysx |
| `Isaac-Velocity-Rough-G1-v0` | **`Isaac-Velocity-Rough-G1`** | `rough_env_cfg:G1RoughEnvCfg` | `G1RoughPPORunnerCfg` | same |

There is **one task id per terrain**, not one per backend: the reward, observation, action, command,
termination and event configs are the same Python objects on every backend; only `sim.physics`
(and the rough PPO `max_iterations`) differ. The `-Play` variants are gone (`play_mode()` instead).
The backend list is from `docs/source/_static/css/environment-browser.js` lines 52/57.

**Differences from Isaac Lab 2.3.2's G1 task that matter for comparisons** (diff of the 3.0 files
against our copies in `assets/isaac/`):

- Events. 2.3.2's `G1RoughEnvCfg` set `push_robot = None`, `add_base_mass = None` and zeroed the
  reset velocity range. 3.0 no longer does: **`push_robot` is active** (interval 10–15 s, ±0.5 m/s
  xy), **`add_base_mass` is active on `torso_link`** (now multiplicative, log-uniform ×[0.8, 1.25]
  instead of additive), and `reset_base` keeps the base config's **±0.5 m/s / rad/s velocity range**
  on all six axes. `base_com` stays None. Friction 0.8 / 0.6, `reset_robot_joints` scale (1, 1),
  zero external force are unchanged. Consequence: Isaac Lab 3.0's own G1 training numbers are for a
  **harder, randomized task** than 2.3.2's and than MetalSim's `g1_velocity`; a like-for-like
  learning comparison must either port these three events to MetalSim or disable them on the Isaac
  side (`env.events.push_robot=null` etc.). The fidelity protocol disables them.
- Rewards, observations, commands, actuators: identical values (same 13 reward terms and weights,
  same noise, `UniformNoiseCfg` rename only). Asset config `G1_MINIMAL_CFG` identical apart from the
  `effort_limit_sim` → `joint_effort_limit` rename; the USD now comes from the Isaac Sim **6.1**
  asset root (`isaaclab/utils/assets.py`, `_ISAAC_SIM_ASSET_RELEASE = "6.1"`), to be hashed against
  our 5.1 copy.
- Contact sensor prim regex `{ENV_REGEX_NS}/Robot/.*` → `/Robot/[^/]*` (direct children only).
- PPO: rsl_rl 5.4.1 config shape (`actor`/`critic` `RslRlMLPModelCfg`, `obs_groups`), same
  hyper-parameters. Flat: 1500 iterations on every backend. Rough: `max_iterations =
  preset(default=3000, newton_mjwarp=5000)`, with NVIDIA's comment: "Newton needs ~1.7x the PPO
  iterations to match PhysX on G1 ... The gap is sample-efficiency, not a ceiling — no physics or
  reward tuning closes it" (`config/g1/agents/rsl_rl_ppo_cfg.py`).
- Physics step: `sim.dt = 0.005`, decimation 4 (unchanged). On Newton, `num_substeps = 2`, so
  **MuJoCo Warp integrates at 2.5 ms**, which is the step MetalSim already trains at (PARITY §1.5).

## 5. Newton / MuJoCo Warp solver settings as exposed and as the G1 tasks set them

Config classes: `isaaclab_newton.physics.NewtonCfg` (`physics/newton_manager_cfg.py`) and
`MJWarpSolverCfg` (`physics/mjwarp_manager_cfg.py`).

| field | `MJWarpSolverCfg` default | G1 rough (`RoughPhysicsCfg.newton_mjwarp`) | G1 flat |
|---|---|---|---|
| solver | `"newton"` | – | – |
| iterations | 100 | – (100) | – (100) |
| ls_iterations | 50 | – (50) | – (50) |
| tolerance | 1e-6 (early exit on residual) | – | – |
| integrator | `"euler"` | `"implicitfast"` | inherited |
| cone | `"pyramidal"` | `"pyramidal"` | inherited |
| impratio | 1.0 | 1.0 | inherited |
| njmax (constraint rows / world) | 300 | 1000 in base, **300** in `G1RoughEnvCfg` | **95** |
| nconmax (contacts / world) | None | 300 | **10** |
| use_mujoco_contacts | True | **False** → Newton `CollisionPipeline` (`max_triangle_pairs=2_500_000`, broad phase "explicit", contact reduction on) | inherited |
| ccd_iterations | 35 | – | – |
| update_data_interval | 1 | – | – |
| `NewtonCfg.num_substeps` | 1 | **2** (2.5 ms solver step) | inherited |
| `NewtonCfg.collision_decimation` | 0 = collide **once per 5 ms physics tick**, reused by both substeps | – | – |
| `NewtonCfg.default_shape_cfg` | margin 0, gap 0.01, ke 2.5e3, kd 100, mu 1.0 | **margin 0, ke 1.6e5, kd 1100** | inherited |
| `NewtonCfg.use_cuda_graph` | True | – | – |

Notes:
- MuJoCo Warp on CUDA exits the Newton iterations early at `tolerance` (100 / 50 are caps); MetalSim
  runs a fixed 10 / 20 budget because Metal has no conditional graph node (PARITY §1.4). Matching
  Isaac's convergence rather than its caps is the relevant target.
- Contact softness comes from Newton's shape `ke`/`kd`, converted to a MuJoCo `solref` by Newton's
  `SolverMuJoCo` (`newton/_src/solvers/mujoco/kernels.py::convert_solref` in Newton 1.5.2: with
  d(width) = d(r) = 1, timeconst = 2 / kd = 1.82 ms and dampratio = kd / 2 · sqrt(1 / ke) = 1.375
  for the G1 values, versus MetalSim's solref 0.02 / 1; which conversion mode 1.5.2 applies is to be
  read from the live `mjw_model.geom_solref` on the VM and recorded in `meta.json`).
- `save_to_mjcf` writes the generated MJCF; with the live `mjw_model.opt` it gives the exact
  settings to replicate. `debug_mode=True` reports per-env iteration counts and cap hits.
- The "solver transitioning" page on `main`
  (https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/solver-transitioning.html)
  still documents the beta API (`nefc_per_env`, `ls_parallel=True`, `isaaclab.sim._impl...`); in
  3.0.0-EA `nefc_per_env` is `njmax`, and `ls_parallel` is deprecated and ignored ("Isaac Lab uses
  iterative line search"). The Newton integration index page on `main`
  (https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html)
  still describes Newton as beta with "a limited set of classic RL and flat terrain locomotion"
  examples; the 3.0 docs replace it with `concepts/physics_backends.rst`, `solver_differences.rst`
  and `solver-tuning/tune_mjwarp.rst` (the tuning guide: capacities `nconmax`/`njmax`, substeps,
  `iterations`/`ls_iterations`/`tolerance`, cone / impratio, contact pipeline, "diagnose-first order").

## 6. Deformables on both backends

Source: `docs/source/concepts/deformables.rst`, `coupled_solvers.rst`, `migration/include/deformables.rst`.

- One asset class `isaaclab.assets.DeformableObject` / `DeformableObjectCfg` for volume
  (`UsdGeom.TetMesh`) and surface (triangle mesh, cloth) deformables; `CableObject` for cables.
  The kind follows from the material cfg (surface material → cloth).
- PhysX (Isaac Sim 6.x, OmniPhysics deformable schema, the same system Isaac Sim exposes as
  `isaacsim.core.experimental.prims.DeformablePrim`): `PhysxDeformableBodyPropertiesCfg`,
  `PhysxDeformableBodyMaterialCfg(youngs_modulus, poissons_ratio, density)`,
  `PhysxSurfaceDeformableBodyMaterialCfg`. Volume and surface: yes; cables: no.
- Newton: **VBD solver only** for every deformable kind; `NewtonDeformableBodyPropertiesCfg`,
  `NewtonDeformableBodyMaterialCfg(k_mu, k_lambda, density)` (Lamé parameters; the docs give the
  conversion from E, ν), `NewtonSurfaceDeformableBodyMaterialCfg(density, particle_radius, tri_ke,
  tri_ka, edge_ke)`; cloth self-contact is off by default. Robot + deformable in one scene uses the
  experimental `isaaclab_contrib.coupling` (MJWarp entry for the rigid robot, VBD entry for the
  soft body, proxy (`mode="lagged"` recommended) or ADMM coupling). Registered tasks:
  `Isaac-Lift-Soft-Franka`, `Isaac-Lift-Cloth-Franka` (+ `-Camera`).
- OvPhysX deformables: experimental, CUDA only.
- Consequence for MetalSim: our deformables run on MuJoCo Warp flex (docs/research/deformables_2026-09-25.md);
  Isaac Lab 3.0's Newton path is VBD, not MuJoCo flex, so "Isaac's own MuJoCo Warp" is **not** a
  like-for-like deformable reference; PhysX FEM and Newton VBD are two different references.

## 7. NVIDIA's published G1 throughput in 3.0 (reported, not re-measured)

`docs/source/_static/benchmarks/environment-performance-release.csv` (2026-09-09, RTX PRO 6000
Blackwell Server, rsl_rl training, 8192 envs, total env-steps/s): flat G1 PhysX 136,950, **Newton
MJWarp 264,026**, OVPhysX 176,802; rough G1 PhysX 112,289, Newton MJWarp 172,780, OVPhysX 150,628.
`environment-performance.csv` (2026-08-18, RTX PRO 6000 Workstation, 4096 envs, rough G1): runtime
(random actions) Newton 153,865, OVPhysX 120,116, PhysX 82,573; training Newton 148,556, PhysX 74,529.
So on NVIDIA's hardware Newton/MuJoCo Warp is 1.5–1.9× Isaac Sim PhysX on this task.

## 8. What changes in our Isaac-side protocol

| 2.3.2 script | 3.0 replacement |
|---|---|
| `scripts/benchmarks/benchmark_non_rl.py --task ... --num_frames 100 --headless` | `isaaclab benchmark runtime --task Isaac-Velocity-{Flat,Rough}-G1 --num_envs 4096 --num_steps N --warmup_steps W physics=<backend> --viz none` (schema JSON output; host-return timing by default, `--measure_sync_step` for a synchronized breakdown) |
| `scripts/reinforcement_learning/rsl_rl/train.py --task ... --headless --max_iterations 1500 --seed 0` | `isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 --num_envs 4096 --max_iterations 1500 --seed 0 physics=<backend> --viz none` |
| `record_g1.py` (Isaac-Velocity-Flat-G1-v0, 2.x APIs) | port: task id, `physics=` selection, XYZW quaternions, `ProxyArray.torch`, `write_*_index`, `robot.actuators.applied_effort`, disable `push_robot` / `add_base_mass` / reset velocity to keep the protocol deterministic; RTX frames need Kit (`renderer=isaacsim_rtx`) on both backends |
