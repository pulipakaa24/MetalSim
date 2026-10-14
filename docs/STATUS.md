# Status against the plan (2026-10-13, main `3670f4b8`)

This file was first written on 2026-09-25 and brought to the measured record on 2026-10-13 (docs audit). Text that
a later measurement replaced is kept and marked **superseded**. What is in flight right now and what is known to be
broken is in `docs/NOW.md`.

**Reference note:** the G1 rigid-body numbers were first measured against Isaac Sim 5.1 + Isaac Lab 2.3.2 (PhysX).
Isaac Lab 3.0-EA (2026-09-16, Isaac Sim 6.1, Newton with MuJoCo-Warp / VBD / MPM / coupled solvers alongside PhysX)
was installed on the L4 and is measured as a reference on both backends since 2026-09-25 (`PARITY.md` §1.8; G1 like
for like, deformables, MPM, Franka-Pour §1b). Each row says which reference it uses.

Machine: Apple M4 Max (40-core GPU, 18.4 TFLOPS FP32 vendor peak), 64 GB, macOS 26.7, Command Line
Tools only. Every figure below is **measured** here unless marked reported. The per-row evidence
(test file::function, assertion, result, verdict) is in `PARITY.md`; MPM in `metalsim/physics/mpm/MPM.md`; rendering
coverage in `RENDER_COVERAGE.md`; the change log in `CHANGELOG.md`; sources in `RESEARCH.md` and `docs/research/`;
tests in `tests/`.

**Test suites on main (zero known failures):** rigid suite (`tests/` without `test_mpm.py`, render tests included; `.venv`) 424 passed, 10 skipped; MPM suite (`tests/test_mpm.py`, `.venv-newton152`) 15 passed, 1 xfailed; re-run on this tree on 2026-10-13, logs with every skip reason in `runs/docs_audit/`. The four long-standing rigid-suite failures
were root-caused and fixed on 2026-10-13 (PR #7: three stale tests and one real guard bug, the overflow flags that
`reset_data` erased; `research_notes/capacity_overflow_guard/`). Superseded: "tests in `tests/` (all pass)"
(2026-09-25, 113 passed / 14 skipped from a fresh clone; GAPS "Install path").

## What the parity claim rests on now

Physics against PhysX (Isaac Sim 5.1 on an NVIDIA L4, `PARITY.md` §1.7, measured 2026-09-24): the
same G1 asset under the same open-loop protocols lands in the same state (hold and 1 m drop: joints
within 0.03–0.06 rad throughout, root height RMSE 1.3 / 3.5 cm, torque RMS within 1 %). The impact-peak gap
(MuJoCo 3.4–5.7× PhysX's sampled peak) was later shown to be a reporting artefact: impulses per event match Isaac's
within 1–4 % (GAPS, "Contact impact peak"). Since 2026-09-26 the G1 default contacts are elliptic friction cones,
impratio 10, Newton cap 20 (20 ms impact forces 1.08× / 1.07× of PhysX's). Open: peak foot penetration at the 1 m
landing, 13 mm against PhysX's 0.5 mm (GAPS). Newton XPBD on the Metal Warp fork (`PARITY.md` §2.1) runs the G1 with
working drives, mesh colliders and Isaac's actuator limits and is wired into the task (`engine="newton"`); it was
archived as experimental on 2026-09-25 and is not used for parity claims. Superseded: "it is not yet wired into the
task code".

The direct comparison is **Isaac-Velocity-{Flat,Rough}-G1** on Isaac Lab's own `g1_minimal.usd`
(the asset Isaac loads), with Isaac's actuator table, initial state, observation/reward/termination
terms, terrain generator (ported verbatim), height scanner and PPO configuration. Tests pose our converted model at
the USD's authored configuration and match every joint frame to 1e-6 m, every link's mass properties
to the USD, and every actuator gain to Isaac's source; the batched Metal physics matches MuJoCo C
on the same asset. What is *not* claimed: PhysX contact-level equality (MuJoCo's solver is used by
design). Images were compared against Isaac's RTX renderer on the L4 (same asset, state, camera and lights;
`PARITY.md` §1.7); superseded: "any image comparison against Isaac's RTX renderer (Isaac cannot run here)".

Since 2026-10-09 the second reference task is Isaac Lab 3.0-EA's MPM: Newton 1.5.2's implicit MPM runs on Metal with
CUDA's semantics and Isaac's Franka-Pour task runs end to end (`PARITY.md` §1b, MPM.md).

The earlier cartpole and SO-101 numbers remain in `PHASES.md` as **pipeline** measurements: the
cartpole tasks are MJCF equivalents without contacts, the lift scene is a reconstruction with no
published benchmark. Neither is cited as physics parity evidence.

## Workstreams

| WS | plan item | state |
|---|---|---|
| WS1 scene layer | USD with UsdPhysics/UsdShade/UsdSemantics; MJCF and URDF importers; compile to device tables; round-trip export | MJCF/URDF→USD and USD→MuJoCo (lossless and generic UsdPhysics) importers; proven on a real Isaac asset (G1: joint frames, masses, colliders, gravity/COM sentinels); MaterialX flattened at load time for four surface types (PARITY §2); USD area lights, GeomSubsets, mesh UVs (RENDER_COVERAGE §1d). Not yet: Hydra/usdview. Superseded: "Not yet: MaterialX graphs". |
| WS2 GPU physics | MuJoCo Warp on Metal, graph capture, collision audit, deformables later | Warp fork (branch `metalsim`, pinned `95f6f121`) with ICB graph replay and exact early exit of captured loops; MuJoCo Warp fork (pinned `f1b35e2`): heightfield and plane-convex contact fixes, elliptic cones on Metal, flex fixes; per-world model fields; parity vs MuJoCo C on SO-101, Panda, Go1, G1; capacity overflow flags latched per substep (2026-10-13). MuJoCo Warp's own suite on Metal: 1447 passed / 3 failed / 39 skipped (2026-09-22, base fork), 1450 passed / 2 failed / 39 skipped on the elliptic branch (2026-09-25, both failures pre-existing; PARITY §1.4); flex suite 237/237 (PARITY §2). Deformables: flex, XPBD cloth / rope, Newton VBD on Metal (PARITY §2). MPM: Newton 1.5.2 `SolverImplicitMPM` on Metal with CUDA's semantics, bitwise deterministic (MPM.md); it runs through Newton in `.venv-newton152`, not yet in MetalSim's own `BatchSim` scenes. |
| WS3 interop | zero-copy MPS tensors, event ordering, conformance test | Done: DLPack `kDLMetal` / `from_blob` paths, MTLSharedEvent ordering across Warp/render/torch queues, runtime counters as the no-host-sync instrument. |
| WS4 renderer | tier 0 raster; tier 1 hybrid RT + denoise + MetalFX; tier 2 path tracer; MaterialX | Tier 0 (parity vs `mujoco.Renderer`: IoU 0.99, depth 0.1 mm, texture corr 0.996); tier 1 (RT soft shadows, AO, reflections); tier 2 path tracer (analytic Lambertian 0.4000 exact, furnace 0.498/0.5, 1/√spp convergence), with OmniPBR / UsdPreviewSurface BRDFs, USD light units, DomeLight HDR maps (tilted domes since 2026-10-13), USD area lights, Open Image Denoise for hero frames; MaterialX flattening. Since 2026-10-12 (RENDER_COVERAGE): appearance fixes against MuJoCo's renderer, deformables / cables / tendons / skins / sites, Newton models through a bridge (NEWTON_RENDER.md), world extent and per-env fields, fisheye lenses, rolling shutter, per-env textures, haze / fog, MPM particles as render grains in tiers 0 and 2 (PARTICLES.md). Not yet: MetalFX upscaling. Superseded: "Not yet: denoiser, MetalFX upscale, MaterialX". |
| WS5 sensors | lidar/radar/cameras/IMU/contact on shared BVH | Metal RT acceleration structures refit from physics; lidar vs `mj_ray` < 2 mm; ray depth == raster depth; Isaac-style height scanner on terrain (Warp heightfield kernel, exact vs `mj_ray`); Isaac Lab's `ContactSensor` as Warp kernels; lidar beam divergence, multi-return, intensity and radar-lite (GAPS "Sensors"); transparent geoms in the ray-traced depth camera and lidar (2026-10-13, RENDER_COVERAGE §3e). Not yet: MPM particles in the ray-traced sensors. Superseded: "Not yet: beam divergence, multi-return, radar-lite". |
| WS6 data generation | randomizers, annotators, COCO/KITTI writers | Done (Replicator annotators, `BasicWriter` layout, Isaac Lab event terms; renderer-side per-env texture pool since 2026-10-13). |
| WS7 learner | zero-copy obs/actions, rollout policy in Warp, PPO tuning, profiler | Warp MLP rollout policy with whole-rollout graph replay; rsl_rl KL schedule, clipped value loss; Isaac's G1 PPO config; checkpoint loading reaches the Warp rollout policy (fixed 2026-10-13, PR #10); pixel PPO on MPS with Isaac's camera-cartpole agent config and Metal gradient kernels (12.7 K env-steps/s incl. training, was 7.5 K; PARITY §2); Franka-Pour's rsl_rl 5.4.1 PPO transcribed (PARITY §1b.3). Not yet: CNN policy in Warp. Superseded: "the MPS update is 76 % of camera-RL time" (before the Metal kernels, update 7.2 → 3.22 s). |
| WS8 tooling | viewer, debugging, profiling, benchmark suite | Benchmark suite, fidelity benchmark, race finder, gallery, Isaac-protocol G1 benchmark (`metalsim.learn.g1_velocity`), GPU job queue with classes and a live dashboard (GUIDE §4), watchdog. Not yet: usdview/Storm viewer, powermetrics. |
| WS9 ROS 2 / HIL | network bridge | Not started (ROS 2). The TRON1 robot bridge is separate (`metalsim/tron1`, `docs/TRON1.md`). |

## Acceptance tests (plan §2.3)

| test | result |
|---|---|
| Physics: MuJoCo Warp suite green on Metal; trajectory parity vs CPU MuJoCo | MuJoCo Warp suite 1447/1450 on 2026-09-22 (3 flex failures), 1450 passed / 2 failed on the elliptic branch on 2026-09-25 (pre-existing); flex suite 237/237 after the fork's flex fixes; one-step parity on 4 robots; G1 (Isaac's asset) 0.5 s incl. landing: max joint diff 1.2e-2 rad, median 3.6e-5 |
| Rendering tier 0: silhouette IoU > 0.95, texture corr > 0.99 | 0.994–0.997; 0.996 |
| Rendering tier 1/2 vs Isaac RTX on a shared USD scene | **measured** against Isaac Sim 5.1 RTX on an L4, same G1 asset/state/camera/lights (`PARITY.md` §1.7): tier 2 + Open Image Denoise vs RTX path tracer, robot pixels 28.8 dB PSNR / SSIM 0.94, whole frame 45.6 dB (RTX's own real-time and path-traced modes are 25.9 dB apart on the robot). Superseded (first comparison, 2026-09-24): robot-only tier 2 13.7 dB / 0.48 SSIM / FLIP 0.024, tier 0 11.5 dB / 0.35; silhouette IoU 0.83, robot depth RMSE 2.6 cm; whole-frame rows then awaited the grey-ground re-recording. Tier 2 also validated radiometrically (analytic + furnace) |
| Sensors: lidar vs Isaac RTX lidar | vs MuJoCo `mj_ray`: median < 2 mm; height scan exact on the heightfield (no RTX lidar recording) |
| Lidar-based RL | `metalsim.learn.lidar_nav` learns goal navigation from a 64-beam scan: 1024 envs × 300 iterations at 21.2 K env-steps/s, time-to-goal 218 → 69 steps (measured) |
| Pipeline: zero host copies per step, one sync per rollout | runtime counters on physics, render, sensor and Warp-rollout loops: 0 syncs, 0 host ops |
| Training: identical PPO config, identical result | **met on the G1 flat task (three seeds ours, one Isaac's)**: under the elliptic default (2026-09-26) +26.8 / +26.9 / +26.6 at iteration 1000 (mean 26.8 ± 0.2) vs Isaac Lab 2.3.2 + PhysX +27.3, full-length episodes by the same iteration; rough +13.7 / terrain level 6.10 at 1500 (one seed) vs Isaac's +14.1 / 5.9 (README, `PARITY.md` §1.5). Isaac Lab 3.0-EA like for like: flat +27.0 ± 1.4 at 1499 (three seeds) vs Isaac Newton +27.5, rough seed 0 +13.7 / 5.99 vs +14.5 / 5.80 (§1.8). Re-checked on main 2026-10-13 (§1.5: rough and flat reproduce the documented curves). Franka-Pour (Isaac Lab 3.0-EA MPM task, 1,024 envs, 60 iterations): **partial**, Isaac's single run inside our 5-seed prediction interval through iteration 49, about 0.06 success ahead at 50-59 (§1b.5). Superseded, in order: pyramidal default +28.4 / 26.3 / 26.0 (origin velocity point) and +27.1 (COM); fixed PPO on MuJoCo Warp +32.8 / 995 under the rough weights; the real rsl_rl on our physics reaching full episodes by iteration 200 after our PPO's three rollout bugs were fixed in f0f2115 (repeated exploration noise, phantom boundary action, time-outs as falls) and the flat reward weights were ported; before that, Isaac's run tracked the command at 0.78 of max by iteration 200 while ours reached 0.11 at iteration 1000 (`PARITY.md` §1.5) |

## Headline throughput (uncontended)

- **G1 (Isaac's asset), 4096 envs, 2.5 ms training setting, elliptic default (measured 2026-09-26 afternoon,
  `runs/ellip_ws/final.wrapper.log`)**: full PPO loop 37.3–37.7 K env-steps/s, env step 41.0 K, physics alone 84.0 K;
  pyramidal (`recommended_pyramidal`) in the same session: loop 56.9–57.5 K, env step 61.3 K, physics 100.5 K
  (README, `PARITY.md` §1.4). References: Isaac Lab 2.3.2 + PhysX on the L4 45.9 K; Isaac Lab 3.0 on the L4 (env step
  alone) 72.0 K Newton / MuJoCo Warp, 45.1 K PhysX; RTX 4090 82 K (published).
- **Superseded (2026-09-25, MuJoCo's default contacts, before the elliptic default)**: physics only 81.0K, full env
  step 67.6K, rollout + inference 61.7K, full PPO loop 56.6K env-steps/s (was 27.9K; rough 41.9K, was 23.7K), physics
  unchanged to float noise (`PARITY.md` §1.4), above Isaac Lab 2.3.2 + PhysX on the L4 (45.9K measured) and 0.69× the
  published 4090 figure.
- **Earlier (5 ms setting), 2026-09-24**: flat step only 55.2K
  env-steps/s (Isaac Lab RTX 4090 published 94K); step + Warp-policy inference 55.4K (88K); full PPO
  loop 47.8–51.7K (82K). Rough (patched heightfield kernel): 55.4K / 47.8K / 39.8–42.2K. Raw 0.48–0.63×
  on a chip with 4.5× less peak FP32; 2.2–2.8× per TFLOPS. Physics alone 68.6K flat, 61.3K rough.
- **Newton XPBD backend (archived as experimental 2026-09-25; MuJoCo Warp is the parity engine)**, same task, 4096 envs (2026-09-24, clean re-run with a control row): full PPO loop 110.1K env-steps/s at the
  training setting (4 it. at 1.25 ms) vs MuJoCo Warp's 27.8K at its 2.5 ms training setting (rough terrain: 26.1K vs 24.8K, no advantage); Isaac's
  published 4090 PhysX number is 82K. Training on Newton gives the same learning curve shape as
  MuJoCo Warp, 4–6× sooner in wall-clock (`PARITY.md` §1.4–1.5).
- **Rough terrain** runs on the MetalSim fork of MuJoCo Warp: its heightfield-mesh contacts were
  defective (inverted normals, worlds launched; `PARITY.md` §1.6) and its heightfield kernel launched
  one thread per contact slot, which exhausted memory at 4096 envs until the per-world capacity was
  set to 32.
- **Franka-Pour** (Isaac Lab 3.0-EA MPM task, `PARITY.md` §1b.4): 1,024 envs step / full PPO loop 1,054 / 992
  env-steps/s vs Isaac on the L4 317 (about 3.1×); 2,048 envs (the task README's) 986 / 979 here, does not fit the
  L4's 24 GB.
- **MPM scenes** (16-inch, ms/frame, Metal / L4; MPM.md 8, 8f, 8j): pure-MPM scenes near or past the L4 with fast_math
  honoured (`"relaxed"`), slower with the parity default `"off"` (granular 40.5 vs 35.8, multi_material 50.6 vs 35.5);
  coupled scenes: ANYmal eager 0.67x, two-way eager 1.09x, ANYmal captured 1.29x, two-way captured 1.76x.

- Cartpole-RGB 100×100, 1024 envs, full env step: tier 0 47,911, tier 1 43,368, tier 2 (1 spp)
  36,896 env-steps/s (Isaac Lab RTX 4090 published 50,000, rasterized; pipeline comparison only).
- Cartpole state, 4096 envs, full PPO loop with the rollout in Warp: 595K steps/s (Isaac 510K).

## Judgement calls made (to confirm)

1. Product/package name `metalsim`; workspace `~/robosim`.
2. MuJoCo Warp solver budget on the G1 taken from MuJoCo Warp's own G1 benchmark (10/20,
   `implicitfast`, eulerdamp off), `njmax` 256. Since 2026-09-26 the elliptic default raises the Newton cap to 20
   (decided by the owner, DECISIONS), and captured Newton loops exit early on Metal (MuJoCo Warp fork `21be81b`, Warp
   fork `6bceb39d`). Superseded: "Metal cannot exit the Newton loop early".
3. Rough terrain: Isaac Lab v2.3.2's generator ported verbatim (grid-exact against Isaac's own code), with its
   curriculum (GAPS "Exact terrain heights", "Task fidelity"). Superseded: "re-implemented from Isaac's config (same
   layout/mix/ranges, not the same random heights); terrain curriculum not applied (all rows sampled)".
4. Contact terms of Isaac's flat and exact rough reward sets come from `metalsim.sensors.contact.ContactSensor`
   (Isaac's ContactSensor semantics: air / contact times, force history; `metalsim/learn/g1_velocity.py`); the first
   rough port keeps MuJoCo touch sensors on sites enclosing the colliders (1 N threshold). Superseded: "instead of
   PhysX contact reporting with a 3-step history" for every set.
5. MuJoCo/OpenGL light intensities treated as π× radiance so tiers 0–2 match MuJoCo's brightness.
6. Tier 2 camera-RL batches run without a denoiser by default (spend on spp; à-trous is the batched fallback,
   DECISIONS 2026-09-25); hero frames use Open Image Denoise (`PARITY.md` §1.7). 1 spp rollouts are noisy by design.

## Deferred work items

- **MPM / particles** (Isaac Lab 3.0-EA's MPM = Newton 1.5.2 `SolverImplicitMPM`): physics parity with the L4 complete (2026-10-12 close-out); three items open, below. `metalsim/physics/mpm/MPM.md`. Newton's solver runs on Metal with CUDA's semantics, bitwise deterministic (coupled scenes too, MPM.md 8b), every acceleration bitwise against Newton's kernels. Against the L4: column-collapse T1/T2 pass, T3 all pass, and all six floor scenes inside the L4's own perturbation ensemble (worst-frame ratio multi_material 0.80, ANYmal 0.94, two-way 0.83, granular 0.66, plate drag 0.97; MPM.md 8e, `runs/mpm/closeout/floors_final.md`; with `metal_cuda_libm`, Warp's default). Speed (16-inch, ms/frame, Metal / L4): pure-MPM scenes near or past the L4 with fast_math honoured (`"relaxed"`, MPM.md 8, 8c), slower with the parity default `"off"` (granular 40.5 vs 35.8, multi_material 50.6 vs 35.5, MPM.md 8f); coupled scenes re-timed 2026-10-13 on the L4's protocol (MPM.md 8j): ANYmal eager 0.67x (faster), two-way eager 1.09x, ANYmal captured 1.29x, two-way captured 1.76x (slower). Franka-Pour, Isaac's MPM task: MPM step at 2,048 worlds 345 -> 78.9 ms captured (MPM.md 8f), full task at 1,024 envs about 3.1x the L4's env-steps/s, and 2,048 envs (the README's) only fit here (PARITY §1b.4); learning: Isaac's run inside our 5-seed prediction interval through iteration 49, just outside it at iterations 50-59, where Isaac's single run leads ours by about 0.06 success (PARITY §1b.5, verdict partial). Pins: Warp `metalsim` 95f6f121, Newton `metalsim-1.5.2` 3fd1f052 (`scripts/setup_newton152.sh`), Newton 1.7 lines a4be3604 / 9f317390 (both carry the mesh-SDF barrier fix), MuJoCo Warp `metalsim` f1b35e2, all on GitHub. The 2026-10-12 02:32 GPU hang once put on Newton's cylinder-on-plane test was most likely the 25 M-instance particle acceleration-structure build over Metal's 2^24 limit, since fixed (strong circumstantial evidence, not proven; MPM.md 8d). **Open:** (1) Isaac seeds for Franka-Pour (`stage_pour_seeds.sh`) to settle the 0.06 learning gap: seeds 1 and 2 were started on the L4 on 2026-10-13 (in flight, `docs/NOW.md`; no result recorded yet); (2) captured coupled-scene speed at one world (two-way 1.76x, ANYmal 1.29x the L4; MPM.md 8j: the two-way excess is mostly one Metal-only rigidity product that grows during the run; candidate fix there, in progress per `docs/NOW.md`); (3) MPM inside MetalSim's own `BatchSim` scenes (today MPM runs through Newton in `.venv-newton152`; in progress per `docs/NOW.md`).
  **Rendering** (2026-10-12, `metalsim/render/PARTICLES.md`): particles are drawn as render grains (instances of
  irregular grain meshes, hashed per particle, per-grain colour palette) in the tier-2 path tracer's instance
  acceleration structure and in tier 0's raster pass (tier 1 draws them with the raster's shading, without
  ray-traced shadows), from numpy or directly from a Warp particle buffer; validated
  analytically (single-grain coverage and irradiance, white furnace in a grain pile), batched, deterministic; 22 M
  grains per frame measured, 25.1 M on the 16-inch (2026-10-13, PARTICLES.md). Not drawn: particles in the ray-traced
  depth camera and lidar. Demo: Newton's ANYmal C on MPM sand path-traced (`scripts/particle_demo/`).
- **MPM in the main simulator** (in progress since 2026-10-13, branch `mpm-main-sim`, `metalsim/physics/mpm/MPM.md`
  section 11): `MPMCoupling` adds Isaac Lab's MPM material to a `BatchSim` scene in `.venv` (vendored copy of the
  Newton 1.5.2 fork's solver, bitwise Newton's own; Isaac's proxy coupling). A box on sand follows Isaac's coupling
  path to rounding. Found and fixed on the fork: Gauss-Seidel with separate worlds was an invalid coloring (data race,
  non-repeatable multi-world runs, Newton's own 1.5.2 included). Measured: bitwise repeatable to 1,024 worlds (GS), 13.7 / 223.6 ms per coupled step at 1 / 1,024 worlds; rigid side repeatable only with the new opt-in `BatchSimOptions(deterministic=True)` (batch 02). Broken: in the `"staggered"` exchange (Isaac's Franka-Pour setting, MetalSim's default until batch 03) a body resting on sand sinks at g dt per step through any bed (coupling lag, not the sand; default now `"lagged"`, being validated). Not yet measured: the angle of repose (batch 02's lift run void). Open: robot
  scenes (ANYmal on sand) in the main simulator.
