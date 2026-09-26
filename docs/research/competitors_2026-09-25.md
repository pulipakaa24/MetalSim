# Competitors: does anything match or beat MetalSim? (research, 2026-09-25)

Question: is there any product or project, on any hardware, that achieves or outperforms MetalSim? The
headline question is Isaac-class robot simulation and learning on an Apple silicon GPU. The wider one is
the same capability, delivered any other way.

Method:
- a web and source survey (READMEs, source at HEAD, GitHub issues, papers, vendor pages);
- benchmarks run on an M4 Max (40-core GPU, 64 GB, on AC power) through `scripts/gpu_run.sh ... timing`
  on an idle GPU.

Every number is labelled **measured here**, **published** (by the project or vendor) or **estimated**.
Scripts: `scripts/diagnostics/competitors/`. Logs: `runs/competitors/`. The elliptic-cone finding that
came out of this is in `docs/HANDOFF_elliptic_cone_perf.md`.

## 1. Bottom line

- **Nothing matches MetalSim as a whole on a Mac.** No other project combines all of these on an Apple
  GPU:
  - Isaac Lab tasks run from Isaac's own assets and configuration;
  - parity measured against Isaac Sim and Isaac Lab;
  - rendering, lidar and radar on Metal's ray-tracing hardware;
  - Replicator-layout data;
  - camera RL.
- **On raw physics speed on a Mac, Genesis is the only serious rival.** Measured here with matched
  cones, it is on par on the Go2 and slower on the G1. Where Genesis leads (elliptic cones, 1.3–1.9×),
  the cause is a Metal launch-size problem in MuJoCo Warp that can be fixed (see the handoff).
- **NVIDIA hardware beats MetalSim on throughput**, whether local (RTX 4090/5090, DGX Spark) or rented
  (Brev L40S, about $1/hr). So does real Isaac on fidelity, by definition. None of it runs on the Mac.
- **The threat to watch** is official Isaac Lab on a Mac GPU through warp-metal plus Isaac Lab 3.0's
  kit-less Newton/MuJoCo Warp path (GA targeted for about the end of October 2026). That would take
  away task parity as a differentiator. The renderer, sensors and Replicator output would remain.

## 2. Measured here

**Head-to-head, physics only** (`metalsim_step.py`, `genesis_step.py`):
- same Menagerie MJCF (Go2, G1) in both engines;
- joint PD to random targets around `home`, resampled every 20 steps;
- 4 ms step, 4096 envs, Newton solver at 10 / 20 iterations;
- synchronized, 1000 steps after 100 warm-up steps.

Genesis is v1.4.2 (`a4b45ff`) on `gs.metal` with torch on MPS.

| model, friction cone | MetalSim steps/s | Genesis steps/s | ratio (MetalSim / Genesis) |
|---|---|---|---|
| Go2, pyramidal | 1,023,772 | 1,133,984 | 0.90 |
| G1, pyramidal (the model's own) | 229,447 | 186,542 | **1.23** |
| Go2, elliptic (the model's own) | 262,371 | 507,474 | 0.52 |
| G1, elliptic | 80,024 | 108,205 | 0.74 |

Log: `runs/competitors/cone.log`.

**Out of the box, the Go2 comparison is not like for like.** Genesis ignores an MJCF's
`cone="elliptic"` unless `RigidOptions(friction_cone=gs.friction_cone.elliptic)` is set (it only warns,
`genesis/utils/mjcf.py:267`). The first matrix therefore showed Genesis 4.3× ahead on the Go2
(1,118,506 vs 262,226; `results_*.jsonl`). That run was Genesis-pyramidal against MetalSim-elliptic.

**Contact load**, contacts per world after 200 steps at 256 envs (`*_contacts.py`,
`runs/competitors/contacts_per_world.log`):

| | MetalSim | Genesis |
|---|---|---|
| Go2 | 4.96 | 4.84 (0 self-contacts) |
| G1 | 13.79 (13.55 in an earlier run) | 6.69 (1.09 self-contacts) |

- **Go2:** the load is equal.
- **G1:** MetalSim solves about 2× the contacts. That is likely MuJoCo C's plane-to-convex contact set
  (4 per pair, MuJoCo Warp fork `284dcd1`) against Genesis's smaller set. So the pyramidal G1 lead
  (1.23×) comes with twice the contacts.

The physics agrees across engines: the final mean base height is Go2 0.189 vs 0.188, and G1 0.084 vs
0.092. The G1 falls under these random targets in both engines.

**End-to-end RL:**
- **Genesis's own `examples/locomotion/go2_train.py`** at 4096 envs on Metal/MPS, rsl-rl 5.5.1:
  170–171K env-steps/s for the full PPO loop (collection 0.30 s + learning 0.27 s per iteration),
  steady over 30 iterations, reward rising. Log: `runs/competitors/genesis_go2_4096.log`.
  - This is the first public evidence that Genesis locomotion trains end to end on a Mac GPU. I found
    no published M-series figure.
  - It is not comparable to the G1 task: 12 actuators, 2 substeps of 10 ms, no self-collision,
    pyramidal cones and a short reward set, against the G1's 43 DoF, 8 substeps of 2.5 ms and Isaac's
    full task.
- **MetalSim G1 flat** at `main` `fbaab25` on this machine (`g1_tp_variants.py 4096`), with the README
  figures in parentheses:
  - physics only 84.1K (81.0K);
  - full env step 55.2K (67.6K);
  - full PPO loop 50.2K (56.6K).
  - The difference is in `solver.solve`: 5.50 ms per substep against 4.23 ms. It is open; see the
    handoff doc.

**MuJoCo-MLX-Cpp** (`20niship/MuJoCo-MLX-Cpp` `3ae0e4a`):
- Built against MuJoCo 3.5.0 and MLX 0.32.2 (it pins 0.31.2).
- Its own `bench_baseline`, one process per benchmark, a 120 s cap, 3 repeats.
- MetalSim ran under the same protocol (`metalsim_mlxprotocol.py`): model file as is, Euler, zero
  control, reset included, solver budget from the XML (100 / 50).

| steps/s | 256 envs | 2048 envs | 4096 envs |
|---|---|---|---|
| DeepMind humanoid, MetalSim | 45,725 | 253,660 | **395,734** |
| DeepMind humanoid, MuJoCo-MLX-Cpp | 23,427 / 40,096 / 25,722 | hung ×3 | 103,444 / 102,311 / 102,850 |
| Go2, MetalSim | 21,788 | 40,178 | 42,917 |
| Go2, MuJoCo-MLX-Cpp | 40,826 / 41,446 / 34,140 | 127,806 / 125,971 / 138,623 | hung, 5,690, 116,527 |

Logs: `runs/competitors/mlx_only.log`, `metalsim_mlxprotocol.log`.

- Its README's 331K steps/s (8192 envs) and 73K training SPS are **published**, on Gymnasium
  Humanoid-v5, and were not reproduced here. Its training benchmarks need `arghyasur1991/MuJoCo-MLX`,
  which returned 404 during the survey.
- It hung on 1 in 3 of the larger runs.
- Its Go2 lead is the elliptic-cone issue again: nothing in its source uses the `ELLIPTIC` enum it
  defines, so it solves the Go2 with pyramidal cones.

## 3. Other simulators that run on the Apple GPU

**innate-inc/warp-metal (David Dobas)**
- What it is: a Metal backend for NVIDIA Warp, adding a `metal:0` device next to stock `warp-lang`.
  Needs macOS 15+.
  - [repo](https://github.com/innate-inc/warp-metal)
  - [release 1.18.0.dev20260917](https://github.com/DavidDobas/warp-metal/releases/tag/v1.18.0.dev20260917)
- Coverage: graph capture, tiles, backward passes, mesh/BVH/volume.
- Gaps: no float64, 32-bit atomics only, no fixedarray. MuJoCo Warp and mjlab need patch branches.
- **Published:** mjlab G1 velocity, 1024 envs, about 1.0 s per training iteration on an M5 Pro, against
  0.3 s on an RTX 5090. **Estimated:** about 25K env-steps/s, assuming 24 steps per env per
  iteration.
- It is MetalSim's own base (the Warp fork is on innate-inc/warp `ce15f6b`), not a rival stack. It has
  no rendering, sensors or Isaac parity.

**Genesis World** (Genesis AI: $105M seed in July 2025; Genesis World 1.0, Nyx and Quadrants in May 2026)
- Physics on Metal: Quadrants (Genesis's Taichi fork) compiles the rigid, MPM, SPH, PBD and FEM
  solvers for Metal. Limits:
  - no float64 on Metal;
  - some compiler optimisations forced off on macOS (NaNs);
  - a Metal-only BVH workaround in the raycaster (`genesis/engine/bvh.py`).
- CUDA only:
  - the Madrona batch renderer, which is its fast camera-RL path;
  - Nyx, its photoreal renderer (x86-64 Linux/Windows with an NVIDIA GPU);
  - the IPC solver (uipc).
- On a Mac, rendering is OpenGL rasterization or the LuisaRender path tracer (Metal 3+, denoiser off).
- Correctness: open Quadrants Metal miscompiles
  [#925](https://github.com/Genesis-Embodied-AI/quadrants/issues/925) (atomics in branches, filed
  2026-09-24) and [#835](https://github.com/Genesis-Embodied-AI/quadrants/issues/835) (a local vector
  read returning 0.0). Its CI skips the FEM implicit test and one IK test on Mac GPU.
- USD import: yes (`genesis/utils/usd/`). Isaac Lab task parity: none. Replicator-style output: none.
  Lidar on Metal: yes, but as a compute BVH rather than ray-tracing hardware.
- Speed claims are all on NVIDIA and **published**:
  - "43M FPS" (Franka, RTX 4090). Stone Tao's re-benchmark measured about 0.29M
    ([blog](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is),
    [issue #181](https://github.com/Genesis-Embodied-AI/genesis-world/issues/181)). Genesis's
    follow-up report restated 43M with self-collision on; that is unverified.
  - "Up to 4.6×" for Quadrants.
- Commercially, the simulator is positioned as an evaluation engine for its GENE-26.5 model. Hardware
  (a hand, a glove, the "Eno" robot) is on a waitlist. No paid simulator product was found.
- Sources: [TechCrunch](https://techcrunch.com/2026/05/06/khosla-backed-robotics-startup-genesis-ai-has-gone-full-stack-demo-shows/),
  [MarkTechPost](https://www.marktechpost.com/2026/05/30/genesis-ai-releases-nyx-quadrants-and-genesis-world-1-0-physics-platform-for-scalable-robotics-foundation-model-evaluation/).

**MuJoCo-MLX-Cpp** (see §2): MuJoCo reimplemented in MLX C++. Euler only in batched mode, no sensors,
no rendering, a 2-week-old repo at the time of the survey.

**Smaller projects**, none competitive:
- [UniLab](https://unilabsim.github.io/) ([arXiv 2605.30313](https://arxiv.org/abs/2605.30313)):
  physics on CPU workers, learning on MPS/MLX. **Published:** FastSAC G1 Walk Flat in 18.8 min on an
  M5 Max. No env-steps/s figure.
- [RobotFlow-Labs/Mujoco-mlx](https://github.com/RobotFlow-Labs/Mujoco-mlx): MJX ported to MLX,
  cartpole and pendulum only. **Published:** 26.7 steps/s.
- [metal-rl-envs](https://github.com/Abhinav-Sai-Podugu/metal-rl-envs): toy planar environments.
  Its billion-steps/s claims are for toy bodies.
- [microduck-lab](https://github.com/jonathanhawkins/microduck-lab): CPU MuJoCo, about 16.5K
  env-steps/s.

## 4. Not GPU-accelerated on a Mac

| project | Mac status | source |
|---|---|---|
| NVIDIA Warp (upstream) | "macOS wheels support CPU execution but not Metal acceleration" | [README](https://github.com/NVIDIA/warp) |
| MuJoCo Warp (upstream) | NVIDIA only; Metal only through forks | [docs](https://mujoco.readthedocs.io/en/latest/mjwarp/) |
| Newton 1.x | Warp-based, so Mac GPU only through warp-metal; no port found | [repo](https://github.com/newton-physics/newton) |
| MJX / Brax / MuJoCo Playground | jax-metal unmaintained; jax-mps "very early" | [discussion](https://github.com/jax-ml/jax/discussions/34648), [jax-mps](https://github.com/tillahoffmann/jax-mps) |
| ManiSkill3 | "GPU simulation is currently not yet supported on MacOS" | [docs](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html) |
| Madrona | CUDA GPU backend | [README](https://github.com/shacklettbp/madrona) |
| mjlab | Mac is "evaluation only" without warp-metal | — |
| Isaac Sim / Isaac Lab | no macOS support | — |
| Webots, CoppeliaSim, Drake, Gazebo Jetty, Unity (ML-Agents), RealityKit / SceneKit | native on Mac but CPU physics, no massively parallel RL | [Gazebo Jetty macOS](https://gazebosim.org/docs/latest/install_osx_src/) |
| Apple | nothing on robotics simulation at WWDC26; LeRobot-MLX covers policy training only | — |

## 5. On NVIDIA hardware or in the cloud (beats MetalSim on throughput, not on running on a Mac)

**Isaac Lab on local GPUs.** **Published** by NVIDIA for `Isaac-Velocity-Rough-G1-v0` at 4096 envs
([benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)):

| GPU | env step only | with training |
|---|---|---|
| RTX 4090 | 94K | 82K |
| L40 | 72K | 62K |

The Isaac Lab paper ([arXiv 2511.04831](https://arxiv.org/html/2511.04831v1)) shows G1 rough on an RTX
5090 at about 50–80K FPS with training. That is **estimated**, read off a figure. Tiled cameras reach
100K+ FPS at 64×64 with 1024 envs on an RTX PRO 6000, far above MetalSim's 12.7K camera cartpole.
MetalSim's own L4 measurements are in PARITY §1.4 and §1.8.

**DGX Spark (GB10, $4,699).**
- **Published:** about 65K sim steps/s on `Isaac-Velocity-Rough-H1-v0` at 512 envs
  ([Arm blog](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/rethinking-robotics-reinforcement-learning-a-practical-humanoid-training-workflow-on-dgx-spark));
  it is unclear whether that includes training.
- Missing on aarch64: SkillGen/cuRobo, XR teleop, JAX on GPU and Cosmos Transfer
  ([install docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)).
- It is native Isaac on a desktop box at about the price of an M4 Max machine.

**Newton 1.0 and Isaac Lab 3.0.**
- Newton 1.0 went GA at GTC in March 2026, under the Linux Foundation (NVIDIA, DeepMind, Disney).
  **Published** (NVIDIA): MuJoCo Warp up to 252× MJX for locomotion and 475× for manipulation on an
  RTX PRO 6000 ([Dataconomy](https://dataconomy.com/2026/03/17/nvidia-launches-newton-1-0-physics-engine-for-industrial-robot-training/)).
- Isaac Lab 3.0 is multi-backend and kit-less; see `isaaclab3_reference_2026-09-25.md`.

**MuJoCo Playground / MJX** ([arXiv 2502.08844](https://arxiv.org/html/2502.08844v1)):
- **Published** on an A100: G1 joystick about 106K steps/s, Go1 about 417K, pixel cartpole with Madrona
  about 403K.
- It has MuJoCo contacts, not PhysX-matched, and no Isaac USD, Replicator or path tracing.

**mjlab:** a community report found it about 6–7× slower than Isaac Lab on Go1 on a 4090
([discussion #220](https://github.com/mujocolab/mjlab/discussions/220)). Unverified.

**Others, all NVIDIA only:**
- ManiSkill3 ([arXiv 2410.00425](https://arxiv.org/abs/2410.00425)): manipulation and rendering.
- GS-Playground ([arXiv 2604.25459](https://arxiv.org/abs/2604.25459)): Gaussian-splat rendering, about
  10⁴ FPS at 640×480.
- RoboVerse/MetaSim ([repo](https://github.com/RoboVerseOrg/MetaSim)): a wrapper over other engines, not
  an engine.

**Cloud and streamed to a Mac:**
- NVIDIA Brev Isaac Launchable ([isaac-launchable](https://github.com/isaac-sim/isaac-launchable)):
  browser VS Code plus Kit WebRTC. Works from macOS, with streaming issues
  ([#5172](https://github.com/isaac-sim/IsaacLab/issues/5172)). An L40S costs about $1.06/hr
  ([ComputePrices](https://computeprices.com/gpus/l40s)).
  - It is the strongest "Isaac on a Mac" option: it is Isaac, and faster.
  - It needs a network, costs per hour, and adds latency.
- Omniverse Kit App Streaming on Azure / AWS / GCP: the same idea with more setup.
- Duality FalconCloud ([blog](https://www.duality.ai/blog/falconcloud)): Unreal-based digital twins,
  strong on sensors and synthetic data, but not massively parallel RL.
- Applied Intuition and Foretellix: enterprise AV and off-road validation. Not RL-throughput tools, not
  on Mac.

## 6. Scorecard against MetalSim

| axis | ahead of MetalSim | not ahead |
|---|---|---|
| G1 PPO throughput | Isaac Lab on RTX 4090/5090 and L40 (published), MJX Playground on A100 (published), DGX Spark (published, H1) | warp-metal (estimated ~25K), MuJoCo-MLX-Cpp, UniLab, Genesis on Mac (no G1 task) |
| Physics throughput on a Mac, same model | Genesis with elliptic cones (measured, 1.3–1.9×; a fixable launch-size issue) | Genesis with pyramidal cones (G1: MetalSim 1.23×; Go2: 0.90×); MuJoCo-MLX-Cpp on the humanoid (MetalSim 3.8×) |
| Camera RL | Isaac tiled cameras, Madrona/MJX (NVIDIA only) | everything on Mac |
| Fidelity to Isaac | real Isaac (cloud, Spark, RTX) | MuJoCo-based stacks, Genesis, warp-metal |
| Path tracing, lidar/radar, Replicator | Isaac (NVIDIA), Duality (cloud) | no Mac-GPU project |
| Physics breadth on Mac | Genesis (MPM/SPH/FEM on Metal, differentiable; Metal FEM shaky) | — |
| Runs on a Mac GPU | — | only warp-metal, Genesis and MuJoCo-MLX-Cpp run at all |
| Cost | Brev for light use (~$1/hr) | — |

## 7. Gaps in this survey

- X/Twitter was not searched. Simulately and MuJoCo-based startups were not surveyed in depth.
- None of the NVIDIA, Playground or DGX Spark figures were re-measured.
- MuJoCo Warp's nightly dashboard
  ([link](https://google-deepmind.github.io/mujoco_warp/nightly/)) could not be read as text.
- Genesis rendering and sensors on Metal were read from source, not benchmarked.
- The warp-metal and mjlab stack was not run here.
- A G1 task in Genesis, or a Go2 task in MetalSim, would make end-to-end RL comparable. Neither exists.
