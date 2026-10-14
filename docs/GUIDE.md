# MetalSim user guide

MetalSim runs Isaac Lab-style robot learning (batched physics, rendering, sensors, PPO) on the GPU of
an Apple Silicon Mac, through Metal. This guide covers installing it, running the shipped tasks,
bringing your own robot, sharing the GPU between jobs, choosing an engine and trading throughput for
fidelity. What has been measured against Isaac Sim / Isaac Lab is in [`PARITY.md`](PARITY.md); this
guide links to it and does not repeat its numbers.

Requirements: an Apple Silicon Mac (developed and measured on an M4 Max), macOS with the Xcode
Command Line Tools (`xcode-select --install`; the full Xcode app is not needed), Python 3.12 or newer
(developed on 3.12, the install is also tested on 3.14; macOS's own `/usr/bin/python3` is 3.9, so install one from
python.org, Homebrew or `uv python install 3.12`), git, [uv](https://docs.astral.sh/uv/) for the optional environments
built by `scripts/setup_newton152.sh` and `scripts/setup_newton.sh`, and about 5 GB of disk.

## 1. Install

```
git clone https://github.com/pulipakaa24/MetalSim && cd MetalSim
python3.12 -m venv .venv && source .venv/bin/activate     # uv only: "$(uv python find 3.12)" -m venv .venv
pip install -e ".[test,newton]"
scripts/setup_warp.sh
pip install -e "git+https://github.com/pulipakaa24/mujoco_warp@f1b35e2b3ddc80b197b0c665bf6d71523d175daf#egg=mujoco-warp" "mujoco>=3.14,<3.15"      # pinned: README "Install"
scripts/fetch_isaac_assets.sh
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie upstream/mujoco_menagerie
pytest tests -q
scripts/setup_newton152.sh                 # MPM and Franka-Pour: .venv-newton152 (below)
scripts/fetch_polyhaven_assets.sh demo     # the ANYmal-on-sand render demo's HDRI and sand texture
```

Run every command from the repository root; the clone can live anywhere, under any user and in a directory whose
path has spaces. `newton` in the first `pip install` is Newton 1.7 (the `newton` extra), which MPM in `BatchSim`
scenes (section 2, `tests/test_mpm_main.py`) and the Newton engine import; install it before `scripts/setup_warp.sh`,
so the fork replaces the PyPI Warp it pulls in. `scripts/fetch_isaac_assets.sh` fetches NVIDIA's `g1_minimal.usd`
(the G1 tasks) and the Franka-Pour scene (15 files, SHA-256 checked) from NVIDIA's public asset store.
MuJoCo Menagerie supplies the robot models some tests, the ANYmal demo render and the Franka probes load
(`MENAGERIE` points elsewhere); without it those tests are skipped.

On a fresh clone (2026-09-25, M4 Max, fast network, pip's download cache and Warp's kernel cache
already filled) the install took one to one and a half minutes and the tests two and a half minutes (113
tests then; the suite has grown since: about 6 minutes for `pytest tests` on 2026-10-13, `runs/eqfix/rigid_suite.log`);
with empty caches expect several more minutes for the PyTorch download and the first Metal kernel
compilations. The tests use the GPU; a few compare GPU results bit for bit and have failed
intermittently while other GPU jobs were running, so run them on an otherwise idle GPU. `pip install -e .` without `[test]` is enough
to run everything except the tests. Optional extras:

| extra | for |
|---|---|
| `test` | the test suite (pytest, pillow, imageio, scipy, trimesh) |
| `parity` | the fidelity protocol and render-fidelity metrics (imageio-ffmpeg, scikit-image, lpips, flip-evaluator) |
| `rslrl` | `metalsim.learn.train_g1_rslrl` (rsl-rl-lib 3.1.2, the reference learner) |
| `newton` | the Newton XPBD engine (experimental, see section 5) |
| `tron1` | the LimX TRON1 tools (onnxruntime) |
| `learn` | the Stable-Baselines3 baseline (gymnasium, stable-baselines3) |

Install several at once with `pip install -e ".[test,parity]"`. Tests that need a missing optional
package are skipped, not failed, as are tests whose data is absent (MuJoCo Menagerie, the Isaac assets, Isaac
Lab's source trees under `upstream/`).

`pip install -e .` first installs `warp-lang` from PyPI (a dependency declared in `pyproject.toml`);
`scripts/setup_warp.sh` then replaces it with the fork below. Check with
`python -c "import warp; print(warp.__file__)"`, which should point into `upstream/warp-metalsim`.

MPM (Isaac Lab 3.0-EA's Newton 1.5.2 implicit MPM, `metalsim/physics/mpm/MPM.md`) runs in a second environment,
`.venv-newton152`, built after the steps above by `scripts/setup_newton152.sh` (Newton fork branch `metalsim-1.5.2`
at a pinned commit, the same Warp checkout, mujoco-warp 3.11.0, PyTorch, pyobjc and ninja at the pinned versions of
`scripts/requirements-newton152.txt`; it needs `uv`). Re-running it brings an existing environment up to that list.
Its tests: `.venv-newton152/bin/python -m pytest tests/test_mpm.py`. Every setup script pins its fork commit and takes
`METALSIM_*_REPO` / `METALSIM_*_REF` overrides; if a fork cannot be cloned it rebuilds the same tree from upstream
plus the diff in `patches/`.

### The three forks

MetalSim depends on three forked projects, none of them on PyPI in this form. The commits below are
what the current results were produced with. Each fork's branch head adds only a `METALSIM_CHANGES.md` (the list of MetalSim
commits on that branch) on top of the commit named here, so installing the branch gives the same code.

**Warp** ([pulipakaa24/warp](https://github.com/pulipakaa24/warp), branch `metalsim`, pinned commit `31cf79d7`;
the rigid-body results were produced at `f194006` and `c200d46b`, see the README). NVIDIA Warp with innate-inc's
Metal backend, plus MetalSim commits; the first seven: interop entry
points (Metal device, queue and buffer handles, cross-queue events, used for zero-copy PyTorch MPS
tensors), graph replay through Metal indirect command buffers, release of deferred frees per completed
command buffer (without it, long loops at 4096 envs ran out of GPU memory) and fixed-size arrays
(`wp.zeros` inside kernels) in the Metal code generator, a configurable size bound for the register
tile Cholesky (`wp.config.metal_register_cholesky_max`), `segmented_sort_pairs` inside a captured Metal graph
(MuJoCo Warp's flex collision in graphs), and `nextafterf` in the Metal kernel runtime (Newton VBD). `scripts/setup_warp.sh [dir]` clones it
(default `upstream/warp-metalsim`), builds the native library with `build_lib.py --no-cuda` and
installs it editable. `METALSIM_WARP_REPO` and `METALSIM_WARP_REF` (branch, tag or commit) point it
elsewhere. If the fork cannot be cloned, the script clones innate-inc/warp at `ce15f6b` and applies
`patches/warp/metalsim-31cf79d7.diff`, which gives the same source tree as `31cf79d7` (`patches/warp/0001-0007` are
the first seven commits, the tree of `f194006`). The commits since (MPM, scalable deterministic mode, FMA contraction
from `fuse_fp`, GPU sort / BSR / NanoVDB builders, `fast_math` forms with `warp.config.metal_fast_math = "off"` by
default) are listed in the branch's `METALSIM_CHANGES.md` and in `metalsim/physics/mpm/MPM.md`.

**MuJoCo Warp** ([pulipakaa24/mujoco_warp](https://github.com/pulipakaa24/mujoco_warp), branch
`metalsim`, pinned commit `f1b35e2`, installed by the `pip install` line above; the commits since `8fbf965`, which
this paragraph describes, are listed in the README's "MuJoCo Warp fork" section: elliptic cones on Metal, the fused and
early-exiting Newton iterations, scratch reuse, the equality block at `njmax`). Google DeepMind's MuJoCo Warp v3.14.0
(`88af9cc`) with, up to `8fbf965`, sixteen commits:
the Metal device patch, heightfield contacts computed against each prism's top plane (upstream
produced inverted normals and launched worlds on rough terrain), and the plane-to-convex contact set of
MuJoCo C (upstream kept 2 contacts where MuJoCo C keeps 4 on a tilted foot), with an environment switch
`MJW_PLANE_CONVEX=legacy` that restores the old behaviour for A/B comparisons, plus three Metal throughput
commits (sparse L'DL one world per thread, the Hessian update fused into its Cholesky, no-op launches
skipped; `MJW_METAL_FUSE_H_CHOLESKY=0` and `MJW_METAL_DENSE_CHOL_MAX` for A/B), plus nine flex (deformable) commits that make MuJoCo Warp's flex contacts
match MuJoCo C (per-pair contact cap and selection, box-triangle and cable-capsule contacts, correct element ids for
every flex, active-layer volume contacts, mesh normals), order flex equality rows deterministically (the Metal
`FlexConstraintTest` failures) and sort on the device on Metal; each behind a `collision_flex` flag
(`scripts/diagnostics/deformable/UPSTREAM.md`). The same sixteen commits (flex ones rebased linearly after the
throughput ones; identical tree to the merge `8fbf965`) are in `patches/mujoco_warp/0001-0016`. To build the pinned
tree without the fork, clone google-deepmind/mujoco_warp at `88af9cc`, `git apply`
`patches/mujoco_warp/metalsim-f1b35e2.diff` (the whole branch as one diff) and `pip install -e` the checkout (README
"Install"). Superseded: "`git am` those patches" gives the `8fbf965` tree, not the pinned one.

**Newton** ([pulipakaa24/newton](https://github.com/pulipakaa24/newton), branch `metalsim`, pinned commit
`9f31739` = XPBD code `90e2332` plus the Metal fixes: HashTable, 64-bit atomics, mesh-SDF barrier, collision launches). Only needed for the experimental Newton XPBD engine. The fork fixes XPBD solver defects
found while validating it (revolute angle wrap at ±π, inconsistent joint relaxation,
iteration-dependent drive stiffness, and others; the upstream issues and pull requests are listed in
`scripts/diagnostics/newton_upstream/FILED.md`). `scripts/setup_newton.sh` installs it into a separate
environment, `.venv-newtonfork`, and shares the Warp checkout with `.venv`. The `newton` extra instead
installs upstream Newton at `45458023`, the commit the validation runs used, without the XPBD fixes but with the
Metal fixes the current Warp fork needs (branch `metalsim-hashtable-metal`, `a4be360`). Branch
`metalsim-1.5.2` (Newton 1.5.2 for MPM) is installed by `scripts/setup_newton152.sh`.

The older single-file patches in `patches/` (`warp-innate-metal-pending-frees.patch`,
`0001-Metal-fixed-size-arrays-...patch`, `mujoco_warp-*.patch`) are individual commits of the same
series, kept for reference.

## 2. First runs

Every command runs from the repository root with `.venv` active. The first run of each task compiles
Warp kernels for Metal (tens of seconds); later runs load them from the kernel cache in
`~/Library/Caches/warp`. For a quick check that a command works, run it with 64 environments and 2
iterations first.

### G1 flat benchmark

Isaac Lab's `Isaac-Velocity-Flat-G1-v0` on Isaac's own `g1_minimal.usd`, with Isaac's benchmark
protocol (100 steps after 10 warm-up steps, synchronized timing):

```
python -m metalsim.learn.g1_velocity 4096 flat            # Isaac's 5 ms physics step
python -m metalsim.learn.g1_velocity 4096 flat 0.0025     # the 2.5 ms step used for training
python -m metalsim.learn.g1_velocity 4096 rough 0.0025    # rough terrain (Isaac's terrain generator)
```

Arguments are positional: `num_envs terrain [physics_dt]`. The printed "synchronized" rate is the one
comparable to Isaac's; the per-call rate is printed for reference and is not comparable. For a full
suite with compute-normalized figures, use `python -m metalsim.bench.suite [--quick] [--json out.json]`.
Benchmarks are only meaningful on an idle GPU: run them through the GPU queue with class `timing`
(section 4).

### G1 flat training with the fixed PPO and Isaac's flat config

```
python -m metalsim.learn.g1_preflight --envs 1024 --physics_dt 0.0025 --probe_iters 20
python -m metalsim.learn.g1_velocity 4096 flat train 1000 runs/g1_flat.log runs/policies/g1_flat.pt 0.0025 --seed 0
```

The training form is `num_envs terrain train iterations [log] [checkpoint.pt|-] [physics_dt]`, plus
`--seed N` anywhere on the line. On flat terrain the task uses Isaac's `G1FlatEnvCfg` reward set by
default, and the learner is `metalsim.learn.ppo_warp.PPOWarp` with Isaac's `G1FlatPPORunnerCfg`
settings (the "fixed PPO": three rollout bugs were fixed on 2026-09-24, see `PARITY.md` §1.5). Use a
physics step of 2.5 ms: at Isaac's 5 ms, Isaac's stiff explicit drives blow up on MuJoCo. The run
writes `runs/g1_flat.log`, an anomaly trail `runs/g1_flat.log.anomalies.jsonl` (section 3),
checkpoints every 100 iterations (`runs/policies/g1_flat_it100.pt`, ...) and the final policy. 1000
iterations at 4096 envs take a little over an hour on an M4 Max.

The same task with the actual `rsl_rl` library as learner (the reference, needs the `rslrl` extra):

```
python -m metalsim.learn.train_g1_rslrl --envs 4096 --iters 1500 --seed 0 --log runs/g1_flat_rslrl.log
```

### Camera cartpole

Isaac Lab's `Isaac-Cartpole-RGB-Camera-Direct-v0` (100×100 RGB, image-only) with the skrl agent
configuration Isaac Lab ships for it, learning from rendered pixels:

```
python -m metalsim.learn.train_cartpole_rgb --envs 1024 --tier 0 --steps 4000000 --log runs/cartpole_rgb.log
python -m metalsim.learn.cartpole_rgb 1024      # rollout throughput per render tier, no learning
```

`--tier` selects the renderer (0 raster, 1 hybrid ray tracing, 2 path tracing; `--spp` and
`--bounces` for tier 2), `--checkpoint out.pt` saves the policy, `--seed` sets the seed.

### Cameras on cloth, cables, tendons, skins and sites

Every renderer and ray-traced sensor draws what MuJoCo's own renderer draws besides geoms, under MuJoCo's default
`mjvOption`: sites in the visible groups, flexes (cloth and soft volumes as their skin offset by the flex radius,
cables as edge cylinders), skins and spatial tendons (`metalsim/render/extras.py`). Their vertices are generated on
the GPU each frame from the simulation state, per env (`metalsim/render/dynamic_mesh.py`), so nothing extra is needed
for a MuJoCo Warp flex scene (`BatchSim`, `DeformableSim`). Other simulators pass their vertex arrays:

```python
from metalsim.render.extras import DeformableMesh
r = Tier0Renderer(flex_model, n, camera="cam", outputs=("rgb", "depth", "seg"))
r.render(xpbd, sources={"flexvert_xpos": xpbd.x})            # XPBDSim: the model's flexes, XPBD's vertices
cloth = DeformableMesh("cloth", mesh.tris, len(mesh.x), rgba=(0.2, 0.5, 0.9, 1))
r = Tier0Renderer(model, n, camera="cam", deformables=[cloth])
r.render(sim, sources={"cloth": physx_cloth.x})              # PhysXClothSim (or any (N, V, 3) Warp array)
```

A simulator without an event (`DeformableSim`, `XPBDSim`, `PhysXClothSim`) is synchronized by the caller before the
render. `visual=` takes an `mujoco.MjvOption` (groups and flags as in MuJoCo's viewer); `visual=False` draws geoms
only, as before. In segmentation (geom mode) the j-th extra object has id `ngeom + 1 + j` (`r.tables.extras`); the
Isaac-style annotators report them as unlabelled instances with their own `instance_id`. Isaac does not draw MuJoCo
sites, tendons or skins (docs/RENDER_COVERAGE.md §5).

### Lidar navigation

A MetalSim task (Isaac Lab has no lidar RL benchmark): a wheeled base reaches a goal among random
boxes, observing a 64-beam planar lidar traced with Metal ray queries.

```
python -m metalsim.learn.lidar_nav 1024 300 runs/lidar_nav.log runs/policies/lidar_nav.pt
```

Arguments are positional: `[num_envs] [iterations] [log] [checkpoint.pt]`.

### Franka-Pour (granular MPM manipulation)

Isaac Lab 3.0-EA's `IsaacContrib-Franka-Pour`: a Franka grasps a cup of 735 granular MPM particles and pours them
into a second cup, on Newton 1.5.2's coupled MuJoCo Warp + implicit MPM solver with Isaac's settings and Isaac's
reset-dataset curriculum (`PARITY.md` §1b). It runs in `.venv-newton152` (Newton 1.5.2 fork; `scripts/setup_newton152.sh`
installs PyTorch there too) and needs Isaac's assets (`scripts/fetch_isaac_assets.sh`, SHA-256 checked):

```
.venv-newton152/bin/python -m metalsim.learn.franka_pour_ppo train --num-envs 2048 --iterations 3000 \
    --out runs/franka_pour/run0 [--resume runs/franka_pour/run0/model_last.pt]
.venv-newton152/bin/python -m metalsim.learn.franka_pour_ppo bench --num-envs 2048 --mode step|rollout|ppo
.venv-newton152/bin/python -m metalsim.learn.franka_pour_ppo bench --num-envs 16 --mode step --steps 5   # first check
metalsim/learn/franka_pour_bench.sh scripts/gpu_run.sh .venv-newton152/bin/python 2048 mac   # all three, timing class
scripts/gpu_run.sh test_pour check 20 -- .venv-newton152/bin/python -m pytest tests/test_franka_pour_terms.py
```

The run writes `log.jsonl` (one record per iteration: rsl_rl's mean reward / episode length over the last 100
episodes, the success rate of the episodes finished that iteration, Isaac's `Episode_Termination/*` and sampler
metrics, losses) and checkpoints every 25 iterations (`--save-interval`; `model_<it>.pt` and `model_last.pt`;
`touch <out>/STOP` pauses cleanly at the end of an iteration). One iteration (96 steps x 2,048 envs) takes about 3.3
minutes on an M4 Max (979 env-steps/s, `PARITY.md` §1b.4), so Isaac's 3,000 iterations take about 7 days. Large-world or new-kernel jobs run under
`scripts/watchdog.py` (kills a job on a CPU stall, memory ceiling or timeout).

### MPM material in a BatchSim scene (main environment)

Sand, snow, mud or viscous media in a scene whose robot runs on `BatchSim` (`.venv`, no second environment): the
solver is Isaac Lab 3.0-EA's (Newton 1.5.2's implicit MPM, MetalSim's vendored copy of the fork), two-way coupled the
way Isaac's `SolverCoupledProxy` couples it to MuJoCo Warp (`metalsim/physics/mpm/MPM.md` section 11):

```python
from metalsim.physics.mpm.coupled import MPMCoupling, MPMConfig, MPMMaterial, GridEmitter, SAND

sim = BatchSim(model, num_envs=64, options=BatchSimOptions(substeps=3))
mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.02))          # Isaac's MPMSolverCfg fields + coupling settings
mpm.add_emitter(GridEmitter(lower=(-.1, -.1, 0), upper=(.1, .1, .05), material=SAND))   # same material in every world
mpm.finalize()
mpm.step()                    # replaces sim.step(): MuJoCo substeps with the sand wrench, then the MPM step
mpm.attach_renderer(renderer) # grains drawn straight from the solver's particle buffer
```

MuJoCo geoms (box, sphere, capsule, cylinder, plane, mesh, ellipsoid) are the colliders; `mpm.set_colliders(bodies=...)`
limits the coupled bodies, `MPMConfig(proxy_mass="effective" | "body" | "kinematic", mode="staggered" | "lagged")`
selects Isaac's exchange variant, `mpm.wrenches()` returns the sand wrench per coupled body, `mpm.reset(mask)` resets
worlds. The material is bitwise repeatable run to run; MuJoCo Warp's own contact solve is not (docs/GAPS.md), so a
scene repeats exactly only while its bodies touch nothing but sand. Tests: `scripts/gpu_run.sh test_mpm_main check 60 -- .venv/bin/python -m pytest
tests/test_mpm_main.py`.
A runnable example, a 2 kg box dropped on a sand bed in 2 worlds (needs the `newton` extra in `.venv`):

```
python metalsim/physics/mpm/probes/box_on_sand.py --engine metalsim --steps 60 --out runs/box_on_sand.npz
```

### Rendering particles (MPM sand)

Both renderers draw particles as render grains (small irregular grain meshes, several per particle, per-grain
colour), in the path tracer's acceleration structure and in tier 0's raster pass:

```python
renderer.set_particles(state.particle_q, radius=0.005, grains_per_particle=27)   # numpy or a Warp array on Metal
```

Details, options and limits: `metalsim/render/PARTICLES.md`. A path-traced demo of Newton's ANYmal-on-sand MPM
example: `scripts/particle_demo/` (`sim_anymal.py` records it in `.venv-newton152`, `render_anymal.py` renders it
in `.venv` with MuJoCo Menagerie's ANYmal C and `scripts/fetch_polyhaven_assets.sh demo`). At small scale (about
80 K particles x 8 grains, a few minutes):

```
.venv-newton152/bin/python scripts/particle_demo/sim_anymal.py --out runs/demo/anymal_small --seconds 1.5 \
    --bed -0.3 -0.3 0 0.3 1.0 0.1
python scripts/particle_demo/render_anymal.py --sim runs/demo/anymal_small --out runs/demo/anymal_small_frames \
    --frames 0:2 --width 480 --height 270 --passes 4 --grains 8 --decor 0 0 0 0
```

The full demo (25 M grains) builds acceleration structures that are heavy for the GPU: run it alone, through the
GPU queue under `scripts/watchdog.py`, on a machine you can reboot (`PARTICLES.md`).

### Lidar and depth through glass (transparent geoms)

The ray-traced lidar and depth camera see transparent geoms (MuJoCo `rgba` alpha < 1) the way Isaac's RTX sensors
are documented to (`docs/PARITY.md`, `docs/RENDER_COVERAGE.md` section 3e):

```python
from metalsim.sensors.raytrace import RayTracer
rt = RayTracer(model, num_envs, max_range=8.0)
lidar = rt.make_lidar("lidar_site", azimuths, elevations,           # radians, one entry per beam
                      transparency="transmit",                       # or "surface" / "behind" / "opaque"
                      max_returns=2, return_mode="strongest_last")   # dual returns: the pane and what is behind
rt.trace_host(datas, [lidar]); out = lidar.numpy()                  # out["range"], intensities, ...
```

A scene with three glass panes, every mode, with the time of each: `python
scripts/diagnostics/sensor_transparency_cost.py --n 64`; the tests: `tests/test_sensor_transparency.py`.

## 3. Bringing your own robot

### Loading USD or MJCF

MuJoCo Warp is the physics engine, so every robot becomes a MuJoCo model (`mujoco.MjSpec`).

- **MJCF / URDF**: load directly with `mujoco.MjModel.from_xml_path` (MuJoCo reads URDF too).
- **USD**: `metalsim.scene.usd_to_mjcf.load_usd(path)` returns an `MjSpec`. A stage written by
  `mjcf_to_usd` carries the original MJCF and is loaded losslessly; any other stage that uses
  UsdPhysics (Isaac-authored assets included) goes through the generic converter.

```
python -m metalsim.scene.usd_to_mjcf robot.usd --generic      # convert and report body/joint/geom counts
python -m metalsim.scene.mjcf_to_usd robot.xml robot.usda      # MJCF -> USD (Isaac importer field mapping)
```

The generic converter handles: rigid bodies with mass, centre of mass, diagonal inertia and principal
axes; revolute, prismatic, spherical, fixed and free (unlocked D6) joints with limits; joint drives as
position actuators (stiffness, damping, max force); `physxJoint:armature` and
`physxJoint:jointFriction`; box, sphere, cylinder, capsule, plane and mesh colliders (meshes as convex
hulls); visual geoms; UsdPreviewSurface and MDL OmniPBR materials; cameras and lights; `metersPerUnit`.
It does not handle tendons, deformables, articulation force sensors or other PhysX-only schemas.

USD conventions it fixes, found on NVIDIA's assets:

- Gravity magnitude 0 is read as unauthored (PhysX's behaviour) and becomes 9.81 m/s²; a zero
  direction means the stage's up axis.
- A centre of mass of (-inf, -inf, -inf), USD's "not authored" value, becomes the body origin; a
  principal-axes quaternion of (0, 0, 0, 0) becomes identity.
- Articulations exported from URDF keep all links as siblings with world-space transforms; body poses
  are computed relative to the joint parent, not the USD parent, so the authored pose is the zero
  joint configuration.
- Angular drive gains are per degree in USD and are converted to per radian; angles are degrees.
- Visual meshes referenced as instanceable prims are read through instance proxies, and material
  bindings resolve on them; MDL materials are read from their `info:mdl` inputs.
- Helper links with 1e-6 kg mass are kept (mass and inertia floors instead of compile errors).

Isaac Lab actuator configs (`ImplicitActuatorCfg` stiffness, damping, effort limit, armature per joint
group) are not in the USD. Apply them to the `MjSpec` after loading, as
`metalsim.learn.g1_velocity.build_g1_model` does for the G1 with `ACTUATORS`.

### Writing a task

`metalsim.physics.batch.BatchSim(model, num_envs, options=BatchSimOptions(...))` runs the batched
physics. A task for the PPO in `metalsim.learn.ppo_warp` is any object with `n`, `obs_dim`, `act_dim`,
`ctrl_lo`/`ctrl_hi`, an `obs` array, a `sim`, `launch_obs()`, `launch_reward_done_reset(pol, bufs)` and
`episode_stats()`; see the module docstring for the optional hooks. `CartpoleTask` in that module is
the minimal example (about 40 lines), `G1VelocityTask` the full one.

### Pre-flight

`metalsim.learn.g1_preflight` runs cheap checks before a long training run. It is written for the G1
task; for another robot, copy it and change the task and the expected numbers.

```
python -m metalsim.learn.g1_preflight --envs 1024 --physics_dt 0.0025 --probe_iters 20
```

It checks that a random policy falls quickly with a return of a few units, that the termination
penalty has the expected size per fall (Isaac scales every reward term by dt), that the terminal
penalty does not dominate the episode, that no world blows up and joint speeds stay bounded under
3-sigma random targets for 400 steps, and that over a short PPO probe the return does not fall while
episodes get longer. It prints `[PASS]` / `[FAIL]` per check and exits non-zero on any failure.
`--probe_iters 0` skips the PPO probe; `--engine newton` runs it on Newton.

### The anomaly monitor

`metalsim.learn.monitor.AnomalyMonitor(task, ppo_config, log=..., path=...)` is task-agnostic and is
attached to every G1 training run. At each logging interval it writes `[anomaly] it N: ...` lines to
the log and a JSON line per iteration to `<log>.anomalies.jsonl`. The flags and what they mean:

| flag | meaning |
|---|---|
| `KL ... > 10x target` | updates too large: stale data or learning rate too high |
| `KL ... < target/10` | the policy has stopped moving |
| `explained variance ... < 0 after warm-up` | the value function does not predict returns |
| `action std ... collapsed below 20% of initial` / `exploded above 3x initial` | exploration has collapsed or diverged |
| `learning rate pinned at bound` | the adaptive schedule is stuck at its minimum or maximum |
| `non-finite observations or rewards` / `N non-finite rows dropped` | NaN or inf reached the learner |
| `terminal-step reward dominates the episode` | falling is cheaper than surviving: check reward scaling |
| `returns fall as episodes lengthen` | the reward punishes survival |
| `joint speed ... > 3x actuator velocity limit` | physics is unstable (step too large, drives too stiff) |
| `joint limit violated by X rad` | soft limits are being pushed through (MuJoCo limits are soft) |
| `contact penetration X cm` | contacts deeper than 2 cm: contact stiffness or step size |
| `contact force ... > 50x robot weight` | impulsive contacts, usually a blow-up in progress |
| `MuJoCo Warp <kind> capacity overflow` | raise `njmax` / `nconmax` in `BatchSimOptions` (an `LS_ITERATIONS` overflow is only the line-search cap being hit) |
| `mechanical energy rose by X J` | energy is being injected: instability |

A few `joint limit` and `penetration` flags during falls early in training are normal with MuJoCo's
default soft limits and contacts; persistent or growing ones are not.

### Fidelity against an Isaac recording

To compare your MetalSim model against PhysX, record a fixed open-loop protocol in Isaac Sim and replay
it here. The shipped scripts do this for the G1 and serve as the template:

1. In Isaac Sim / Isaac Lab (on an NVIDIA machine): `python metalsim/parity/isaac_side/record_g1.py
   --headless --out parity_out/rt --render rt`. It records a 3 s PD hold, 5 s of seeded random joint
   targets and a 3 s drop from 1 m: joint states, root pose, torques, contact forces and camera frames.
2. Copy the output directory here and replay it on MuJoCo Warp:
   `python -m metalsim.parity.record_g1 --isaac parity_out/rt --out runs/parity/metalsim --physics_dt 0.0025`
   (`--no_render` for physics only; `--contact_tuning <preset>` to try a contact preset from
   `metalsim.physics.contact_tuning`).
3. Compare: `python -m metalsim.parity.compare --isaac parity_out/rt --metalsim runs/parity/metalsim --out runs/parity/report`
   (needs the `parity` extra). It writes a JSON report with per-joint RMSE, root height and orientation
   error, time to divergence, contact-force statistics and image metrics, plus side-by-side frames.

`metalsim.parity.side_by_side` renders the same policy checkpoint in both simulators as a video, and
`metalsim.parity.export_policy` exports a MetalSim checkpoint for playback in Isaac
(`isaac_side/play_policy.py`).

## 3b. The default contact model (and switching it off)

`G1VelocityTask(contact_cfg="recommended")` is elliptic friction cones with impratio 10 and a Newton cap of 20 on the
impact-stiffened preset (2026-09-26). It is the PhysX-faithful setting and costs 1.5× on the PPO loop against pyramidal
cones (1.68× before the fused elliptic Newton iteration and the graph-replay early exit of 2026-09-26 afternoon,
`docs/research/elliptic_warmstart_2026-09-26.md`). `contact_cfg="recommended_pyramidal"` restores the previous default, `"default"` MuJoCo's contacts, and every
`metalsim.physics.contact_tuning` preset is selectable by name (`--contact_cfg` on the CLI). A `solver_cfg` such as
`isaaclab3` overrides the cone with Isaac's own settings.

## 4. The GPU queue

The GPU is one shared resource: a benchmark run next to a training run measures neither. Long or
timing-sensitive jobs go through a priority queue:

```
scripts/gpu_run.sh NAME KIND MINUTES -- command args...
python3 scripts/gpu_lock.py status
```

`gpu_run.sh` waits for the GPU, runs the command, and releases the GPU when the command exits for any
reason (it also kills the command if the wrapper is killed). `MINUTES` is an estimate shown to others.
The command runs in the caller's directory (since 2026-10-13; before, it ran in the repo root, so `python -m ...`
from a worktree imported the main checkout's packages); the queue's own files stay under the repo root.
Classes, granted in this order when the GPU frees up (first come, first served within a class):

| class | for |
|---|---|
| `timing` | benchmarks and profiles that need an idle GPU (at most 15 minutes) |
| `render` | renders and short rollouts |
| `check` | correctness work that times nothing (tests, bitwise recordings, parity runs): up to three check jobs share the GPU; they start only when no exclusive job holds the GPU or waits ahead of them, and a check job waiting more than 10 minutes goes ahead of timing jobs |
| `train` | training runs |
| `low` | background work, granted only when nothing else waits |

A job refuses to start while the GPU shows runaway work (utilisation of 90 % or more with no owning process for 10
minutes): the queue pauses and writes `runs/GPU_RUNAWAY_ALERT` (`scripts/gpu_run.sh`).

**Updating the shared fork checkouts.** `scripts/fork_sync.sh` fast-forwards `upstream/mujoco_warp` and
`upstream/warp-innate` to their `fork/metalsim` heads through the queue (timing class), so the working tree never
changes under a job that is importing it. Agents work in worktrees and never touch the shared checkouts.

**Live dashboard.** `scripts/gpu_dashboard.sh` starts a local page at http://localhost:8765 (idempotent;
`--restart` to restart; `python3 scripts/gpu_dashboard.py --watch` is the terminal form, `/json` the
machine form). It shows the running job with a progress bar, ETA, its log and last lines, the waiting
jobs in grant order, GPU time per process, and the finished jobs with exit codes
(`runs/gpu_history.jsonl`). It is not configured per job: every job that goes through the queue
appears by itself, its log is found from the files its process tree has open for writing, and progress
is parsed from generic counters ("it 898", "12/500", "37 %") with the total taken from the log line or
the command line; when nothing is recognisable it shows elapsed time against the requested minutes and
the last log line, so a new kind of job is still readable without touching the dashboard.

A holder whose process has died is released automatically. `status` shows the holder and the waiting
jobs. Example:

```
scripts/gpu_run.sh g1_flat train 70 -- python -m metalsim.learn.g1_velocity 4096 flat train 1000 runs/g1_flat.log runs/policies/g1_flat.pt 0.0025 --seed 0
```

The queue is advisory: commands started without it still run. Short tests and one-minute checks do
not need it.

## 5. Engines

**MuJoCo Warp** is the default and the engine all parity results are measured with
(`G1VelocityTask(engine="mjwarp")`).

**Newton XPBD** is an experimental option and its track is archived: it trains about 1.6 times faster
than MuJoCo Warp on the G1 but moves loosely enough (joint drift, 14-23 % over-travel of PhysX-trained
policies, harder impacts) that it is not used for parity claims. It stays in the tree with its tests.
To try it, install the `newton` extra (upstream) or run `scripts/setup_newton.sh` (the fork, in
`.venv-newtonfork`), then:

```
python -m metalsim.learn.g1_velocity 4096 flat --engine newton --newton_it 4 --newton_dt 0.00125
python -m metalsim.learn.g1_preflight --envs 1024 --engine newton --newton_it 4 --newton_dt 0.00125
```

The decision and the measurements behind it are in `GAPS.md` ("Newton XPBD: archived") and
[`PARITY.md` §2.1](PARITY.md#21-contact-model-what-mujoco-can-do-and-newton-xpbd-on-metal-measured-2026-09-24-newton-xpbd-archived-as-experimental-on-2026-09-25-see-gapsmd).

**Cameras on Newton scenes.** `metalsim.render.newton_bridge.NewtonRenderSource(model)` turns any Newton model into
the `sim` argument of the renderers (tiers 0/1/2), the ray-traced sensors and the annotators:

- It compiles a render twin from the Newton shapes, or uses the source MJCF with `twin=mj_model` for the
  MuJoCo-via-Newton path.
- Each frame, `update(state)` writes the poses from `State.body_q` on the GPU. Each env sees its own world plus the
  global shapes.
- Cloth, soft-body surfaces and particles go through the dynamic-mesh and particle paths:
  `renderer_kwargs()`, `sources`, `attach_particles`.

Design, coverage against the MuJoCo path, cost and the differences from Isaac Lab 3.0 are in
[`NEWTON_RENDER.md`](NEWTON_RENDER.md).

## 6. Throughput vs fidelity

The knobs below trade speed for accuracy. The measured cost of each is in the linked place; where a
cost has not been measured, the table says so.

| knob | where | effect | measured cost |
|---|---|---|---|
| physics step | `physics_dt` (G1 CLI's last positional argument, `--physics_dt`) | 2.5 ms instead of Isaac's 5 ms keeps Isaac's stiff explicit drives stable (blow-ups at 5 ms) at twice the substeps | [§1.4](PARITY.md#14-throughput-on-the-g1-task-isaacs-protocol-4096-envs) (5 ms vs 2.5 ms rows and the L4 comparison at both), [§1.5](PARITY.md#15-learning-on-the-g1-task) (blow-ups at 5 ms) |
| solver iterations | `BatchSimOptions(solver_iterations=, ls_iterations=)` | the G1 uses a Newton cap of 20 under the elliptic default (10 pyramidal) and 20 line-search iterations; since 2026-09-26 captured Newton loops exit early per world on Metal (MuJoCo Warp fork `21be81b`, `MJW_METAL_ICB_EARLY_EXIT=0` disables; before, Metal always ran the full budget) | cap 10 vs 20 under elliptic cones and the early exit: [§1.4](PARITY.md#14-throughput-on-the-g1-task-isaacs-protocol-4096-envs), `docs/research/elliptic_warmstart_2026-09-26.md`; a full sweep has not been measured |
| contact and constraint capacity | `BatchSimOptions(nconmax=, njmax=)` | too small overflows (flagged by the monitor, NaN on constraints); too large costs memory and launch size | [§1.4](PARITY.md#14-throughput-on-the-g1-task-isaacs-protocol-4096-envs) notes (`njmax` 256, 32 contact slots per world and why), [§1.6](PARITY.md#16-rough-terrain-mujoco-warps-heightfield-contacts-found-defective-and-patched) |
| contact model | `contact_cfg` (`--contact_cfg`), `metalsim.physics.contact_tuning` presets, `--contact_tuning` | stiffer contacts and hard limits move the replay closer to PhysX in some metrics; elliptic cones (the default) bring impact forces within 8 % of PhysX's | fidelity per preset: [§1.7](PARITY.md#17-fidelity-protocol-against-isaac-sim-51-on-an-l4-recorded-2026-09-24-physx--rtx); throughput: elliptic default 1.5× the pyramidal PPO-loop cost (section 3b, README), the pyramidal preset 11 % of the loop against MuJoCo's default contacts (`docs/research/throughput_regression_2026-09-25.md`) |
| render tier and samples | `--tier`, `--spp`, `--bounces` (camera tasks), `Tier2Renderer` | tier 0 raster, tier 1 adds ray-traced shadows / AO / reflections, tier 2 path tracing is closest to Isaac's RTX frames | throughput per tier and spp: [§2](PARITY.md#2-platform-capabilities-the-workstreams-with-the-tests-behind-them) ("Tier 1 / tier 2 full rollouts"); image fidelity vs RTX: [§1.7](PARITY.md#17-fidelity-protocol-against-isaac-sim-51-on-an-l4-recorded-2026-09-24-physx--rtx) |
| sensor extras | `metalsim.sensors.contact.ContactSensor`, lidar beam divergence and multi-return, radar-lite | Isaac's contact-sensor quantities; lidar echoes closer to RTX lidar | per-sensor cost in [§2](PARITY.md#2-platform-capabilities-the-workstreams-with-the-tests-behind-them) (contact sensor, lidar extras and radar rows) |
| engine | `engine="newton"` | faster, looser joints | [§2.1](PARITY.md#21-contact-model-what-mujoco-can-do-and-newton-xpbd-on-metal-measured-2026-09-24-newton-xpbd-archived-as-experimental-on-2026-09-25-see-gapsmd), [§1.4](PARITY.md#14-throughput-on-the-g1-task-isaacs-protocol-4096-envs) (engine comparison) |
| number of envs | first CLI argument / `--envs` | throughput per env rises with batch size until the GPU is full | [§1.4](PARITY.md#14-throughput-on-the-g1-task-isaacs-protocol-4096-envs) (1024 / 2048 / 4096 rows) |
| GPU queue class | `scripts/gpu_run.sh ... KIND` | does not change a job's speed; decides who waits. Any concurrent GPU job slows every job running, so measure throughput only under `timing` on an idle GPU | – |

## 7. What is measured against Isaac

[`PARITY.md`](PARITY.md) is the reference. For each Isaac Sim / Isaac Lab feature it names the test
that confirms or disproves equivalence, the measured result and a verdict (confirmed, equivalent by
construction, partial, disproved, not testable here). In outline:

- **Asset, physics model and task terms** for the G1 velocity task: same asset, actuators and reward,
  observation and termination terms, checked term by term against Isaac's formulas (§1.1-1.3).
- **Throughput** on Isaac's G1 benchmark protocol, against Isaac's published RTX 4090 figures and
  Isaac Lab measured on an L4 (§1.4). Speed figures are labelled measured, reported or estimated.
- **Learning**: the same PPO configuration on Isaac's flat config reaches Isaac's return at the same
  iteration (three seeds ours, one Isaac's), with rsl_rl as a learner cross-check (§1.5); Isaac Lab 3.0-EA like for
  like (§1.8).
- **Franka-Pour** (Isaac Lab 3.0-EA's MPM manipulation task): asset, MDP terms against Isaac's own code, learner,
  throughput and learning against Isaac on the L4 (§1b); the MPM solver itself in `metalsim/physics/mpm/MPM.md`.
- **Physics and rendering fidelity** against a PhysX / RTX recording of the same open-loop protocol
  (§1.7), and policy transfer in both directions between the simulators.
- **Platform features** (rendering tiers, sensors, replicator, interop) with their tests (§2), the
  contact-model comparison including Newton (§2.1), and what is disproved, missing or cannot be tested
  on a Mac (§3).

[`GAPS.md`](GAPS.md) lists what is closed and still open; [`STATUS.md`](STATUS.md) is the state
against the plan; [`CHANGELOG.md`](../CHANGELOG.md) lists what landed and when.
