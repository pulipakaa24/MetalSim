# Phase log

Machine: Apple M4 Max, 40-core GPU, 64 GB, macOS 26.7, Command Line Tools only (no Xcode.app).
Python 3.12 venv at `.venv`. Labels: **measured** (this machine), **reported**, **estimate**.

## Phase 0: Measure (2026-09-22) — done

Reproduced mjbatch-metal's baselines with its own scripts (measured):

| workload | published (mjbatch-metal) | reproduced here |
|---|---|---|
| CPU MuJoCo + batched render, cartpole-class, 64 px, N=1024 | 52,600 | 49,595 steps/s |
| same, N=4096 | 53,100 | 53,673 steps/s |
| CPU MuJoCo + batched render, SO-101 lift scene, 128 px, N=64 | 16,957 | 18,040 steps/s |

The SO-101 lift scene is a reconstruction (`assets/so101/scene_box_rl.xml`: Menagerie
robotstudio_so101 + box + `base_cam`); the original lives on another machine.

Not done: Metal System Trace and powermetrics profiling (need Xcode Instruments / sudo).

## Phase 1: GPU physics + interop — in progress

### GPU physics (WS2), measured

MuJoCo Warp 3.14 (metal patch) on the innate-inc Warp fork, `mjwarp-testspeed`, 200 steps,
`--overflow_behavior continue` (the SO-101 scene hits `ls_iterations=20` in a few worlds):

| scene | N | realtime factor | steps/s (= factor / dt) |
|---|---|---|---|
| SO-101 lift (dt 0.005, nv 12, 50 geoms) | 64 | 227× | ~45,000 |
| | 256 | 812× | ~162,000 |
| | 1024 | 2,655× | ~531,000 |
| | 4096 | 5,319× | ~1,064,000 |
| mjwarp humanoid benchmark (dt 0.005, nv 27) | 1024 | 2,084× | ~417,000 |
| | 4096 | 4,711× | ~942,000 |

For scale (reported): MJX humanoid on an A100 at batch 8192 is 950K steps/s; MuJoCo Playground
CartpoleBalance on an A100 is 719K. The plan estimated "low hundreds of thousands" for the
Apple GPU; the measurement is several times higher for these scenes.

MuJoCo Warp test suite on `metal:0`: 1447 passed, 3 failed (2 flex-constraint parity tests,
1 known upstream), 39 skipped. The flex failures are deformable-body tests and are logged for
follow-up (deformables are a later workstream).

### Interop (WS3), measured

- Added to the Warp Metal runtime (fork branch `metalsim`): device/queue/buffer
  handles, foreign event wait and signal.
- `metalsim.interop`: zero-copy MPS tensors over Warp arrays (DLPack `kDLMetal` and
  `at::from_blob`), event wait/signal on torch's command buffer.
- `tests/test_interop.py`: 11 tests pass, including proofs that waits hold the GPU and
  double-buffered producer/consumer ordering in both directions with no host sync in the loop.
- Ping-pong round trip (1M-element ops, both directions each step): 0.46 ms/step host time.

Lesson recorded: Warp's `zeros()`/memsets are asynchronous on its queue; any torch write to an
aliased buffer must be ordered after them (event or sync), otherwise the memset lands later.

### Remaining for Phase 1

- Batched environment on MuJoCo Warp with graph-captured substeps, obs/action tensors aliased
  to MPS, event-ordered handoff (`metalsim.physics`).
- Trajectory parity test vs CPU MuJoCo on SO-101 and Menagerie robots at MJWarp tolerances.
- "Zero host copies per step" conformance: without Instruments, verify by host-time scaling
  (host time per step must not grow with GPU work).

### Graph replay through indirect command buffers (WS2), measured

Warp's Metal graph replay re-encoded every recorded dispatch on the CPU (~4 µs each; a MuJoCo
Warp step of the SO-101 scene is 201 dispatches), so small batches were host-bound. The fork now
encodes a captured graph once into an `MTLIndirectCommandBuffer` (barrier between commands) and
replays it with one call. SO-101 lift, 4 substeps per step, `tests/test_physics.py`:

| N | host time / step before | after | wall / step (GPU-bound) | physics steps/s |
|---|---|---|---|---|
| 64 | 3.4 ms | 0.08 ms | 5.0 ms | 51,000 |
| 2048 | 18.6 ms (queue back-pressure) | 0.02 ms | 16.7 ms | 491,000 |

Runtime counters per step: 804 dispatches, 2 command buffers, 0 host ops, 0 host waits.
Remaining floor at small N: ~6 µs of GPU time per kernel dispatch (Metal's per-dispatch cost
with barriers), i.e. ~1.2 ms per physics substep regardless of batch size. That is a per-kernel
launch floor, the Metal analogue of CUDA kernel launch latency; reducing it needs fewer, fused
kernels in MuJoCo Warp itself (upstream work), not this stack.

## Phase 3 (started early): tier-0 renderer on native Metal — in progress

`metalsim.render.tier0`: PyObjC + MSL, one command buffer per batch (env-params kernel, background
quads, one instanced indexed draw per unique mesh, untile kernel into learner buffers). Instance
transforms and cameras are read from MuJoCo Warp's `geom_xpos/geom_xmat/cam_xpos/cam_xmat` (no
host pack step; body-mounted cameras work). Geometry is clipped to its tile with vertex clip
distances (without it, off-tile geometry such as floor planes was rasterized over the whole
atlas and discarded per fragment, which scaled quadratically with N).

Parity vs `mujoco.Renderer` (`tests/test_render_tier0.py`, measured): silhouette IoU 0.994–0.997
(plan threshold 0.95), depth median error ≤ 0.14 mm (threshold 1 cm), per-geom segmentation IoU
≥ 0.987, texture pattern correlation 0.996 (threshold 0.99), SO-101 silhouette IoU 1.0.

Render-only throughput, primitives scene (mjbatch-metal's `bench.py` scene and grid), measured:

| N | res | metalsim tier-0 (env-frames/s) | mjbatch-metal sync / pipelined (published) |
|---|---|---|---|
| 64 | 64 | 582,830 | 21,438 / – |
| 1024 | 64 | 782,976 | 44,259 / 72,731 |
| 64 | 128 | 435,063 | 12,142 / 20,253 |
| 1024 | 128 | 501,248 | 12,109 / – |
| 256 | 256 | 211,188 | 3,936 / – |

For scale (reported): MuJoCo Warp's own Warp-kernel renderer, primitives, N=512, on an RTX 4090:
522,101 steps/s; Isaac Lab Cartpole-RGB tiled camera 100×100, 1024 envs, RTX 4090: 50K env
steps/s including physics.

SO-101 arm scene (348k triangles per env, undecimated): 13.6K env-frames/s at N=64/128 px and
flat across N, i.e. bound by primitive throughput (~5 G triangles/s on this GPU). mjbatch-metal
published 8.4K sync / 19K pipelined for the same scene. Physics+render at N=64/128 px: 10.1K
steps/s (render-bound). Mesh decimation levels of detail are the remedy; see below.

Levels of detail (measured): welded quadric decimation with a vertex-clustering fallback
(`decimate_faces`). SO-101 arm scene, `metalsim.bench.render_bench assets/so101/scene_box_rl.xml 2000`:

| triangles / env | N | res | render-only env-frames/s | physics (1 substep) + render steps/s |
|---|---|---|---|---|
| 325,034 (undecimated) | 64 | 128 | 13,643 | 10,102 |
| 27,172 (2000 faces/mesh) | 64 | 128 | 90,693 | 22,986 |
| 27,172 | 256 | 128 | 106,179 | – |
| 27,172 | 1024 | 64 | 136,807 | 86,359 |

mjbatch-metal published for this scene: 19,000 env-frames/s pipelined, 16,957 steps/s (N=64,
128 px, CPU physics). MuJoCo Playground PandaPickCubeCartesian with Madrona on datacenter
hardware: ~37,000 steps/s (reported).

## Phase 2: learner path — in progress

### Incident: NaN in batched physics under random control (resolved)

Symptom: the GPU SO-101 lift env produced NaN rewards after ~100 steps; pure MuJoCo Warp on
Metal (no torch) diverged too, Warp CPU did not; 64 identical worlds diverged in a different
subset each run. Diagnosis path: solver-variant sweep (Newton+elliptic NaN; CG or pyramidal
clean), block-size sweep (barrier type irrelevant), atomic-add micro-test (correct), then a
cross-world consistency tool (`metalsim.tools.race_finder`: every launch followed by a check
that identical worlds stay identical) which flagged `d.overflow` and `d.efc.id` right after
collision. Overflow flags: **NEFC** — MuJoCo Warp's default of 64 constraint rows per world
overflows with the SO-101 gripper's condim-6 contacts; rows are dropped in atomic order, so
worlds get different partial contacts, and the elliptic-cone Newton path turns a half-contact
into NaN. `njmax=512`: 0 bad worlds in 256 trials. Not a Metal defect. Fixes: `BatchSim`
defaults `njmax=512` and exposes `overflow_flags()`; tests assert no buffer overflow.
Upstream note: MuJoCo Warp should saturate rather than NaN on NEFC overflow (to file).

### Learner path status (measured)

`metalsim.learn.so101_lift.SO101LiftEnv` (GPU task: MuJoCo Warp physics, tier-0 render at
128 px, reward/termination/reset/visual DR in torch on MPS; no tensor leaves the GPU) and
`metalsim.learn.ppo.PPO` (pixels + proprioception, NatureCNN). N=64:

| quantity | value |
|---|---|
| env only (physics 4 substeps + render + torch reward/reset) | ~7,000 env-steps/s |
| PPO training loop (rollout 64, 4 epochs × 4 minibatches of 1024) | ~1,190 env-steps/s |
| time split | env 23%, policy 3.5%, update 73% |
| mjbatch-metal SB3 baseline (published): CPU physics + WebGPU render + SB3 PPO on MPS | ~1,700 env-steps/s |

The baseline used SB3's config (n_steps 64, batch 512, 8 epochs, gamma 0.9, ent 0.005). The
update phase dominates as the plan predicted (Section 3); per-sample update cost is the item to
fix next (WS7: launch count, precision, minibatch size), then the rollout policy in Warp.

Physics DR (box mass/friction per world) is not yet applied: MuJoCo Warp model fields are
shared across worlds unless expanded (`expand_model_fields` in mjlab); logged as a gap.

### Scene layer (WS1) started

`metalsim.scene.mjcf_to_usd`: MJCF → USD with UsdPhysics (rigid bodies, mass, revolute/prismatic/
spherical joints with limits and drives from actuators, articulation roots, collision APIs with
convex-hull mesh approximation), UsdPreviewSurface materials with textures written as PNG,
UsdSemantics labels, cameras with physical intrinsics, lights, and `mjc:` attributes for lossless
round trip. `tests/test_scene_usd.py` checks the SO-101 scene. Not yet: USD → MuJoCo Warp load
path, URDF import, MaterialX graphs.

### Tier-0 lighting: model lights, headlight, shadow maps (measured)

The renderer now lights with the model's light list (directional/point/spot, positions and
directions read per env from MuJoCo Warp `light_xpos/light_xdir`, so body-mounted lights move),
MuJoCo's headlight and ambient, an optional per-env DR directional light, and a tiled shadow
map (orthographic, one caster per env, 2×2 PCF). MuJoCo/OpenGL light intensities are converted
to radiance (×π) so the energy-conserving BRDF reproduces MuJoCo's brightness.

Image-level comparison with `mujoco.Renderer` on the SO-101 scene at 256 px
(`metalsim.bench.render_fidelity`): PSNR 19.1 dB, FLIP 0.318, mean intensity 88.6 vs 88.6
(before the calibration: 13.4 dB, 0.62, 43 vs 89). The residual is shading-model difference by
design (GGX vs Phong, and MuJoCo's planar floor reflection, which tier 1 provides through ray
tracing); Isaac's own image thresholds (PSNR ≥ 25 / SSIM ≥ 0.9) apply to tier 1 vs Isaac RTX,
not to tier 0 vs MuJoCo. Parity tests (silhouette, depth, segmentation, texture) unchanged.

## WS5 sensors on Metal ray tracing — started (measured)

`metalsim.sensors.raytrace.RayTracer`: one `MTLPrimitiveAccelerationStructure` per unique mesh
(shared vertex/index buffers with the rasterizer), an instance descriptor buffer refit each frame
by a compute kernel from `geom_xpos/geom_xmat`, and one instance acceleration structure rebuilt
per frame for all envs, spatially separated by a grid offset larger than scene extent + max range
(so rays never cross worlds and no per-env masking is needed). Sensors are `intersector` compute
kernels writing Warp/MPS tensors; ordering by events like the renderer.

- Lidar (beam table of azimuth/elevation in a site frame; range, point, normal, hit slot,
  reflectance intensity) vs MuJoCo C `mj_ray` on a primitives scene, 1,625 beams: median
  |Δrange| < 2 mm, 95th percentile < 2 cm (tessellation of curved primitives), hit/miss pattern
  agreement > 97%.
- Ray-cast pinhole depth vs the rasterizer's depth on the SO-101 scene (8 envs, 128 px):
  coverage agreement 100%, median |Δz| 0, 99th percentile 0.1 mm. This cross-validates the
  acceleration structures and the raster pipeline against each other.
- GPU path with `BatchSim`: zero host syncs over a 10-step loop.

Not yet: beam divergence (sub-rays), multi-return, rotation across substeps, range noise
(planned on the torch side), radar-lite, Isaac RTX lidar JSON/USD config import, IMU/contact
sensors (available from MuJoCo Warp sensordata; to be exposed as tensors).

## Phase 4 started: tier 1 hybrid ray tracing (measured)

`Tier0Renderer(..., tier=1)`: the same raster G-buffer pass, but the fragment stage issues
hardware ray queries against the sensor layer's instance acceleration structure (refit each
frame from the physics buffers): soft shadows (cone-sampled, `rt_samples` rays), cosine-weighted
ambient occlusion (0.5 m horizon) and one mirror reflection ray for reflective materials (MuJoCo
`reflectance`, metals), shaded at the hit with the same light list; misses see the model's
skybox colour. Fidelity vs `mujoco.Renderer`, SO-101 at 256 px: PSNR 19.7 dB, FLIP 0.305
(tier 0: 19.1 / 0.318) with the floor reflection now present.

Cost at N=64, 128 px, 2000-face LOD, measured while a training run shared the GPU (so
absolute numbers are pessimistic; the ratios hold):

| tier | rays / pixel | env-frames/s | ms / frame |
|---|---|---|---|
| 0 (shadow map) | 0 | 59,190 | 1.08 |
| 1, rt_samples=1 | 3 | 28,827 | 2.22 |
| 1, rt_samples=4 | 9 | 11,948 | 5.36 |

The plan's estimate for tier 1 was 5–20× tier 0 before upscaling; measured 2–5×. Not yet:
MetalFX temporal denoise/upscale, area lights, one-bounce diffuse GI, motion vectors.

### Rollout policy in Warp (WS7), measured

`metalsim.learn.warp_policy.WarpMLPPolicy`: rsl_rl-style actor-critic MLP evaluated by Warp
kernels on the simulator's queue; torch parameters are re-homed onto Warp memory (zero-copy both
ways), so the optimizer updates weights in place. Actions, log-probs and values are stored in
Warp rollout buffers at a device-side step index, which lets one captured graph (observation
gather + policy + store + physics substeps) be replayed T times per rollout. Parity with the torch
module: 1e-5.

Cartpole (state observations, MJCF equivalent of Isaac's asset, 10 Newton iterations), N=1024,
T=64, measured while a PPO training run shared the GPU:

| quantity | value |
|---|---|
| env-steps/s (physics + policy + storage) | 257,696 |
| host time per step | 0.048 ms |
| dispatches per step | 374 |
| torch launches per step | 0 |
| per TFLOPS FP32 (M4 Max 18.4) | ~14,000 |

Isaac Lab Cartpole-Direct step+inference on an RTX 4090, 4096 envs (published): 910K steps/s,
~11,000 per TFLOPS. Uncontended and at N=4096 numbers to follow from `metalsim.bench.suite`.

Lesson: without GPU-side loop conditions, MuJoCo Warp on Metal runs a model's full `iterations`
budget (MuJoCo's default 100 Newton iterations turned a 2-DOF cartpole substep into 1,081
dispatches); `BatchSimOptions.solver_iterations` should be set per model, as Isaac Lab sets PhysX
iteration counts.

### RL reproducibility: Isaac Lab Cartpole config on the Warp rollout path (measured)

`metalsim.learn.ppo_warp` with rsl_rl's Cartpole config (4096 envs, 16 steps/env, 5 epochs × 4
minibatches, MLP 32×32 ELU, lr 1e-3 with the adaptive-KL schedule at desired_kl 0.01, gamma
0.99, lambda 0.95, entropy 0.005) on the MJCF cartpole with Isaac's reward and reset terms:

| iteration | steps | episode return (max 300) | episode length |
|---|---|---|---|
| 15 | 0.98M | 25.0 | 45 |
| 30 | 1.97M | 197.7 | 208 |
| 45 | 2.95M | 293.6 | 298.5 |
| 150 | 9.8M | 295.2 | 299.9 |

Training-loop throughput (rollout in Warp + torch update): **~320K env-steps/s at N=4096**, measured
while a pixel PPO run shared the GPU. Isaac Lab's published Cartpole-Direct step+inference+train
figure on an RTX 4090 is 510K steps/s; per TFLOPS FP32 that is ~6,200 (4090) vs ~17,400 here.

Two lessons recorded: (1) the initial "no learning" was my MJCF (rail capsule overlapping the cart
box, so contact forces pinned the cart: 100 N moved it 2 mm in MuJoCo C as well) — Isaac's
cartpole has no collisions; (2) rsl_rl's adaptive-KL learning-rate schedule is part of the config
and is needed for the published iteration counts.

## Benchmark suite (uncontended, measured 2026-09-22, `metalsim.bench.suite`)

Raw throughput and the plan's compute-normalized figure (per TFLOPS FP32; M4 Max 40-core = 18.4
TFLOPS vendor peak). Tier-0 rendering here includes the shadow-map pass and full light loop.

| workload | N | raw | per TFLOPS |
|---|---|---|---|
| physics SO-101 lift (BatchSim, ICB replay, njmax 512) | 64 | 45,751 steps/s | 2,486 |
| | 256 | 155,880 | 8,472 |
| | 1024 | 372,169 | 20,227 |
| | 4096 | 541,293 | 29,418 |
| render tier-0 primitives 64 px | 64 | 405,673 env-frames/s | 22,047 |
| | 1024 | 705,256 | 38,329 |
| render tier-0 primitives 128 px | 64 | 336,758 | 18,302 |
| | 1024 | 451,156 | 24,519 |
| render tier-0 SO-101 (2000-face LOD) 128 px | 64 | 64,922 | 3,528 |
| render tier-0 SO-101 (2000-face LOD) 64 px | 1024 | 89,963 | 4,889 |

Reference points (reported, other hardware): MuJoCo Playground CartpoleBalance on an A100
(19.5 TFLOPS) 718,626 steps/s = 36,853 per TFLOPS; MuJoCo Warp render benchmark on an RTX 4090
(82.6 TFLOPS) 522,101 steps/s = 6,321 per TFLOPS; Isaac Lab Cartpole-RGB 1024 envs on a 4090
50K steps/s = 605 per TFLOPS (physics + tiled RTX render).

## Headline comparisons against Isaac Lab's published numbers (measured, uncontended)

| task | this stack (M4 Max, 18.4 TFLOPS) | Isaac Lab (RTX 4090, 82.6 TFLOPS), published | per-TFLOPS ratio |
|---|---|---|---|
| Cartpole-RGB 100×100, 1024 envs, physics + tiled render + reward/reset | 48,056 steps/s | 50,000 (step only) | 4.3× |
| Cartpole state, 4096 envs, rollout only (physics + policy in Warp + reward/reset) | 1,236,013 steps/s | 910,000 (step + inference) | 6.1× |
| Cartpole state, 4096 envs, full PPO training loop | 544,402 steps/s | 510,000 (step + inference + train) | 4.8× |
| Cartpole state, 1024 envs, full PPO training loop | 113,089 steps/s | – | – |

Reading under the plan's principle: at matched task, resolution and batch size the stack delivers
Isaac Lab's throughput class on a laptop chip with 4.5× less peak FP32; the normalized figures
exceed Isaac's, which is the expected consequence of a purpose-built batched renderer and a
rollout with no per-step host or framework launches. The Isaac numbers are NVIDIA's published
figures, not re-measured here; the tasks are equivalents (MJCF cartpole, same reward/reset/camera
spec), not the same USD asset.

Torch-driven per-step envs (`SO101LiftEnv`, `CartpoleRGBEnv`) are bounded by the per-step launch
floor (~60K steps/s at N=1024 for the state-only cartpole); the Warp-rollout path removes it.

## SO-101 lift acceptance run (measured)

Baseline PPO config (SB3's: 64 envs, 64 steps, batch 512, 8 epochs, gamma 0.9, ent 0.005), 3M
steps on the GPU stack: 29.3 min at 1,707 env-steps/s; episode return 15.7 → 127 (reach reward
saturating), success rate 0.0 at 3M steps. The published baseline mentions a 25M-step, ~4 h run
and does not report a success rate; an SB3 run of the original pipeline on the reconstructed
scene (`runs/sb3_baseline/`) is queued for an equal-budget comparison.


## Parity on Isaac's own asset: Unitree G1 velocity task (2026-09-23, measured)

Per the user's direction ("run a direct comparison to something that's been actually benchmarked"),
the reference task is now Isaac-Velocity-{Flat,Rough}-G1-v0 on Isaac Lab's `g1_minimal.usd`, with
Isaac's actuator table, initial state, observation/reward/termination terms and PPO config. The
evidence per row (tests, tolerances, results) is in `PARITY.md`; this section records the work.

### Incidents while bringing Isaac's USD up (all resolved, each now a test)

1. **USD gravity 0**: the asset's `physicsScene` authors `gravityMagnitude = 0`, which PhysX treats
   as "use the default"; the loader copied 0 and the robot floated. Fixed in `usd_to_mjcf.py`
   (positive magnitude only; otherwise 9.81 along the stage down axis).
2. **Centre of mass (-inf, -inf, -inf)**: USD's sentinel for "not authored" on the massless helper
   links (IMU, pelvis contour, logo); produced NaN in `mj_comPos`. Fixed (sentinel → link origin).
3. **`qpos0` overwritten with the init pose**: MuJoCo measures hinge angles from `qpos0`, so this
   silently moved every joint's zero; the model then disagreed with the USD joint frames by up to
   0.38 m at the fingers. Fixed (init pose is a keyframe only). The joint-frame test now shows
   anchor errors of 5.6e-7 m over all 37 joints, at the USD's authored (posed) configuration.
4. **Heightfield spacing**: MuJoCo spans `2·radius` over `n` samples, so the radius must be
   `(n−1)·res/2` for the collision surface to coincide with the scanner's mesh; the height-scan vs
   `mj_ray` test caught a 6.5 mm median error, now 1.9e-6 m.
5. **Metal "Insufficient Memory" at 2048–4096 worlds** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`):
   MuJoCo Warp allocates temporaries every eager step (collision driver `wp.empty`, `qacc`), and the
   Metal backend deferred every free until the next host synchronize, so a benchmark loop with no
   sync accumulated thousands of buffers. Graph replays were unaffected (no allocations). Fixed in
   the backend: frees are attached to the next committed command buffer and released when it
   completes (`pending_frees`). Two knobs were added while diagnosing and left at neutral defaults:
   `WP_METAL_INFLIGHT` (command buffers in flight, default 64 = Metal's queue depth) and
   `WP_METAL_ICB_BATCH` (chunked graph replay, default off).

### Solver settings and honesty notes

MuJoCo Warp's own G1 benchmark settings are used (`benchmarks/unitree_g1/unitree_g1_mjlab.xml`:
10 Newton iterations, 20 line-search iterations, `implicitfast`, eulerdamp off), pyramidal cone,
dense Jacobian, `njmax` 256. On Metal the Newton loop cannot exit early, so the budget is spent
every step; the line-search cap is reached in most worlds most steps (`LS_ITERATIONS` overflow
flag), which is a cost cap, not a failure. Under a PD hold at Isaac's default pose the robot pitches
forward onto its torso within 1.5 s in both MuJoCo C and on Metal (Isaac's 20 Nm/rad ankle gains
cannot hold the COM offset); that is the physics of the configuration, not a bug.

### G1 throughput (Isaac's `benchmark_non_rl` protocol; uncontended; 4096 envs)

| measurement (synchronized, uncontended 2026-09-24) | 4096 | 2048 | 1024 | Isaac (4090, reported) |
|---|---|---|---|---|
| flat, physics only (4 substeps) | 68,660 | 60,682 | – | – |
| flat, step only | 45,958 | 38,722 | 29,617 | 94,000 |
| flat, step + inference | 46,917 | 39,815 | – | 88,000 |
| flat, full PPO loop | 41,740 (log: 44,328) | 34,321 | – | 82,000 |
| rough (patched kernel), physics only | 61,679 | – | – | – |
| rough, step only | 46,408 | 38,848 | – | 94,000 |
| rough, step + inference | 41,159 | – | – | 88,000 |
| rough, full PPO loop | 35,071 (log: 37,217) | – | – | 82,000 |

Eager per-kernel launching instead of one graph per step: 32,815 at 4096. The per-call "Isaac-style"
mean is not used (queue-absorbed, reads ~2× high). Rough terrain: MuJoCo Warp's HFIELD-MESH contacts
return inverted normals (z = −1) with 5 cm penetrations and launch worlds, while MuJoCo C on the same
model is stable (`tests/test_terrain.py::test_hfield_mesh_contacts_match_mujoco_c`, xfail); the
height scanner moved to a Warp heightfield-interpolation kernel (exact vs `mj_ray`) so it captures
into rollout graphs (a cross-queue event signal cannot be captured on Metal).

### G1 flat learning run (measured)

`runs/g1_flat_ppo_300b.log`: Isaac's PPO config, 4096 envs, 300 iterations, 42.4K env-steps/s
end to end. Episode length 41 → 343 control steps; return −203 → −233 (penalties still exceed
tracking; termination still paid). One transient blow-up at iteration 296 (return −1e26 for one
rollout, recovered). A non-finite state now terminates and resets the episode and is counted.

### Tier 2 path tracer (measured)

`metalsim.render.tier2` over `shaders/pathtrace.metal`: progressive accumulation, next-event
estimation for MuJoCo lights and the DR light, MuJoCo headlight on the primary hit, Lambert + GGX
with mixture-pdf sampling, Russian roulette after 3 bounces, sky/clear-colour misses, rgb/depth/seg/
normal outputs, GPU path ordered against `BatchSim` by events. `tests/test_render_tier2.py`:

| test | result |
|---|---|
| Lambertian plane, directional light, analytic radiance 0.4 | 0.4000 (pixel std 0.0000), 0 and 3 bounces |
| white furnace under a uniform sky of 0.5 | 0.4979 |
| noise vs spp (4 → 64) | rms ratio 4.05 (ideal 4.0) |
| direct light vs tier 0 (bounces 0) | PSNR 43.7–45.1 dB; depth median 0.9 mm; seg IoU 1.0 |
| GPU path from BatchSim == host path | mean abs diff < 1 |
| Cartpole-RGB rollout at tier 2 | runs (32 envs, 30 steps, finite rewards) |

Cartpole-RGB 100×100 full env step at 1024 envs, uncontended pass 2026-09-24: tier 0 47,911; tier 1
43,368; tier 2 36,896 (1 spp, 1 bounce), 35,160 (1 spp, 2 bounces), 19,894 (4 spp, 2 bounces)
env-steps/s. Cartpole state PPO at 4096 envs (Warp rollout): 595K env-steps/s, return 295/300 by
iteration 150. Bench suite: SO-101 physics 376,504 steps/s at 1024; tier-0 primitives 714,156
env-frames/s at 1024×64 px (`runs/clean_suite.json`).
