# Status against the plan (2026-09-24)

Machine: Apple M4 Max (40-core GPU, 18.4 TFLOPS FP32 vendor peak), 64 GB, macOS 26.7, Command Line
Tools only. Every figure below is **measured** here unless marked reported. The per-row evidence
(test file::function, assertion, result, verdict) is in `PARITY.md`; the work log with numbers and
incidents is in `PHASES.md`; sources in `RESEARCH.md`; tests in `tests/` (all pass).

## What the parity claim rests on now

Physics against PhysX (Isaac Sim 5.1 on an NVIDIA L4, `PARITY.md` §1.7, measured 2026-09-24): the
same G1 asset under the same open-loop protocols lands in the same state (hold and 1 m drop: joints
within 0.03–0.06 rad throughout, root height RMSE 1.3 / 3.5 cm, torque RMS within 1 %); the contact
model differs in the impact peak (MuJoCo soft contact: 3.4–5.7× PhysX's peak force at equal mean).
Newton XPBD on the Metal Warp fork (`PARITY.md` §2.1) now runs the G1 with working drives, mesh
colliders and Isaac's actuator limits, tracks MuJoCo C within 0.02–0.04 rad, and gives 4.0–4.5× MuJoCo
Warp's physics rate at 4096 envs with 0.16–0.69 cm impact penetration; it is not yet wired into the
task code.

The direct comparison is **Isaac-Velocity-{Flat,Rough}-G1** on Isaac Lab's own `g1_minimal.usd`
(the asset Isaac loads), with Isaac's actuator table, initial state, observation/reward/termination
terms, terrain layout, height scanner and PPO configuration. Tests pose our converted model at the
USD's authored configuration and match every joint frame to 1e-6 m, every link's mass properties
to the USD, and every actuator gain to Isaac's source; the batched Metal physics matches MuJoCo C
on the same asset. What is *not* claimed: PhysX contact-level equality (MuJoCo's solver is used by
design) and any image comparison against Isaac's RTX renderer (Isaac cannot run here).

The earlier cartpole and SO-101 numbers remain in `PHASES.md` as **pipeline** measurements: the
cartpole tasks are MJCF equivalents without contacts, the lift scene is a reconstruction with no
published benchmark. Neither is cited as physics parity evidence.

## Workstreams

| WS | plan item | state |
|---|---|---|
| WS1 scene layer | USD with UsdPhysics/UsdShade/UsdSemantics; MJCF and URDF importers; compile to device tables; round-trip export | MJCF/URDF→USD and USD→MuJoCo (lossless and generic UsdPhysics) importers; proven on a real Isaac asset (G1: joint frames, masses, colliders, gravity/COM sentinels). Not yet: MaterialX graphs, Hydra/usdview. |
| WS2 GPU physics | MuJoCo Warp on Metal, graph capture, collision audit, deformables later | innate-inc Warp fork (branch `metalsim`) with ICB graph replay; MuJoCo Warp suite 1447/1450 on Metal; per-world model fields; parity vs MuJoCo C on SO-101, Panda, Go1, G1. Backend fix this round: frees released per completed command buffer (eager loops at 4096 worlds exhausted GPU memory). |
| WS3 interop | zero-copy MPS tensors, event ordering, conformance test | Done: DLPack `kDLMetal` / `from_blob` paths, MTLSharedEvent ordering across Warp/render/torch queues, runtime counters as the no-host-sync instrument. |
| WS4 renderer | tier 0 raster; tier 1 hybrid RT + denoise + MetalFX; tier 2 path tracer; MaterialX | Tier 0 (parity vs `mujoco.Renderer`: IoU 0.99, depth 0.1 mm, texture corr 0.996); tier 1 (RT soft shadows, AO, reflections); **tier 2 path tracer done**: analytic Lambertian 0.4000 exact, furnace 0.498/0.5, 1/√spp convergence, 43 dB vs tier 0 direct light, full Cartpole-RGB rollouts at tiers 1 and 2. Not yet: denoiser, MetalFX upscale, MaterialX. |
| WS5 sensors | lidar/radar/cameras/IMU/contact on shared BVH | Metal RT acceleration structures refit from physics; lidar vs `mj_ray` < 2 mm; ray depth == raster depth; Isaac-style height scanner on terrain (Warp heightfield kernel, exact vs `mj_ray`); IMU/contact/joint sensors as tensors. Not yet: beam divergence, multi-return, radar-lite. |
| WS6 data generation | randomizers, annotators, COCO/KITTI writers | Done. |
| WS7 learner | zero-copy obs/actions, rollout policy in Warp, PPO tuning, profiler | Warp MLP rollout policy with whole-rollout graph replay; rsl_rl KL schedule, clipped value loss; Isaac's G1 PPO config; pixel PPO on MPS with Isaac's camera-cartpole agent config (learns from pixels, 7.5K env-steps/s incl. training). Not yet: CNN policy in Warp (the MPS update is 76 % of camera-RL time). |
| WS8 tooling | viewer, debugging, profiling, benchmark suite | Benchmark suite, fidelity benchmark, race finder, gallery, Isaac-protocol G1 benchmark (`metalsim.learn.g1_velocity`). Not yet: usdview/Storm viewer, powermetrics. |
| WS9 ROS 2 / HIL | network bridge | Not started. |

## Acceptance tests (plan §2.3)

| test | result |
|---|---|
| Physics: MuJoCo Warp suite green on Metal; trajectory parity vs CPU MuJoCo | 1447/1450 (3 flex deformable failures); one-step parity on 4 robots; G1 (Isaac's asset) 0.5 s incl. landing: max joint diff 1.2e-2 rad, median 3.6e-5 |
| Rendering tier 0: silhouette IoU > 0.95, texture corr > 0.99 | 0.994–0.997; 0.996 |
| Rendering tier 1/2 vs Isaac RTX on a shared USD scene | **measured** against Isaac Sim 5.1 RTX on an L4, same G1 asset/state/camera/lights (`PARITY.md` §1.7): robot-only, brightness-matched, states agreeing: tier 2 13.7 dB PSNR / 0.48 SSIM / FLIP 0.024, tier 0 11.5 dB / 0.35; silhouette IoU 0.83, robot depth RMSE 2.6 cm; whole-frame rows await the grey-ground re-recording (Isaac's plane terrain ignored the material). Tier 2 also validated radiometrically (analytic + furnace) |
| Sensors: lidar vs Isaac RTX lidar | vs MuJoCo `mj_ray`: median < 2 mm; height scan exact on the heightfield |
| Lidar-based RL | `metalsim.learn.lidar_nav` learns goal navigation from a 64-beam scan: 1024 envs × 300 iterations at 21.2 K env-steps/s, time-to-goal 218 → 69 steps (measured) |
| Pipeline: zero host copies per step, one sync per rollout | runtime counters on physics, render, sensor and Warp-rollout loops: 0 syncs, 0 host ops |
| Training: identical PPO config, identical result | **met (one task, one seed)**: MuJoCo Warp + our fixed PPO on Isaac's flat configuration reaches +28.4 return / 1000-step episodes at iteration 1000 vs Isaac's own +27.3 / 991, full episodes at the same iteration (`PARITY.md` §1.5). Earlier: (fixed PPO on MuJoCo Warp: return +32.8 / length 995 at iteration 1000, tracking rsl_rl on the same physics within a few units; Isaac's own run +27.3 under its flat weights; `PARITY.md` §1.5). Earlier: The real rsl_rl on our physics reaches full episodes by iteration 200 and Isaac's tracking by 1000; our PPO had three rollout bugs (repeated exploration noise, phantom boundary action, time-outs as falls), fixed in f0f2115 (`PARITY.md` §1.5). Also found: our task used the rough reward weights where Isaac trained flat. Earlier text: Isaac's own rsl_rl run on the L4 (per-term log) tracks the command at 0.78 of max by iteration 200 and 0.90 by 300 with full 1000-step episodes; ours reaches 0.11 tracking at iteration 1000 with 70–89 % of episodes ending in a fall (`PARITY.md` §1.5). Learner differential (real rsl_rl on our task) and a Newton-backend PPO run are in progress to split learner vs simulation |

## Headline throughput (uncontended)

- **G1 (Isaac's asset), 4096 envs, synchronized, uncontended (2026-09-24)**: flat step only 55.2K
  env-steps/s (Isaac Lab RTX 4090 published 94K); step + Warp-policy inference 55.4K (88K); full PPO
  loop 47.8–51.7K (82K). Rough (patched heightfield kernel): 55.4K / 47.8K / 39.8–42.2K. Raw 0.48–0.63×
  on a chip with 4.5× less peak FP32; 2.2–2.8× per TFLOPS. Physics alone 68.6K flat, 61.3K rough.
- **Newton XPBD backend, same task, 4096 envs (2026-09-24, clean re-run with a control row)**: full PPO loop 110.1K env-steps/s at the
  training setting (4 it. at 1.25 ms) vs MuJoCo Warp's 27.8K at its 2.5 ms training setting (rough terrain: 26.1K vs 24.8K, no advantage); Isaac's
  published 4090 PhysX number is 82K. Training on Newton gives the same learning curve shape as
  MuJoCo Warp, 4–6× sooner in wall-clock (`PARITY.md` §1.4–1.5).
- **Rough terrain** runs on the MetalSim fork of MuJoCo Warp: its heightfield-mesh contacts were
  defective (inverted normals, worlds launched; `PARITY.md` §1.6) and its heightfield kernel launched
  one thread per contact slot, which exhausted memory at 4096 envs until the per-world capacity was
  set to 32.

- Cartpole-RGB 100×100, 1024 envs, full env step: tier 0 47,911, tier 1 43,368, tier 2 (1 spp)
  36,896 env-steps/s (Isaac Lab RTX 4090 published 50,000, rasterized; pipeline comparison only).
- Cartpole state, 4096 envs, full PPO loop with the rollout in Warp: 595K steps/s (Isaac 510K).

## Judgement calls made (to confirm)

1. Product/package name `metalsim`; workspace `~/robosim`.
2. MuJoCo Warp solver budget on the G1 taken from MuJoCo Warp's own G1 benchmark (10/20,
   `implicitfast`, eulerdamp off), `njmax` 256; Metal cannot exit the Newton loop early.
3. Rough terrain re-implemented from Isaac's config (same layout/mix/ranges, not the same random
   heights); terrain curriculum not applied (all rows sampled).
4. Contact terms use MuJoCo touch sensors on sites enclosing the colliders (1 N threshold) instead
   of PhysX contact reporting with a 3-step history.
5. MuJoCo/OpenGL light intensities treated as π× radiance so tiers 0–2 match MuJoCo's brightness.
6. Tier 2 ships without a denoiser; 1 spp rollouts are noisy by design (documented in `PARITY.md`).
