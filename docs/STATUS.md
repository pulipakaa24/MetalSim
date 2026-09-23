# Status against the plan (2026-09-22, end of day 1)

Machine: Apple M4 Max (40-core GPU, 18.4 TFLOPS FP32 vendor peak), 64 GB, macOS 26.7, Command Line
Tools only. Every figure below is **measured** here unless marked reported. Details, tables and
the incident log are in `PHASES.md`; sources in `RESEARCH.md`; tests in `tests/` (all pass).

## Workstreams

| WS | plan item | state |
|---|---|---|
| WS1 scene layer | USD with UsdPhysics/UsdShade/UsdSemantics; MJCF and URDF importers; compile to device tables; round-trip export | MJCF→USD and URDF→USD importers with Isaac's field mapping; USD→MuJoCo loader (lossless and generic UsdPhysics); compiled scene tables (unique meshes, instances, PBR materials, atlas, lights, semantics, intrinsics, LOD). Not yet: MaterialX graphs, Hydra/usdview (needs a self-built OpenUSD). |
| WS2 GPU physics | MuJoCo Warp on Metal, graph capture, collision audit, deformables later | Running on the innate-inc Warp fork (built from source, branch `orchard-interop`). Test suite on Metal: 1447 pass / 3 fail (flex deformables + 1 known). Graph replay through indirect command buffers (host cost 0.02–0.08 ms/step). Per-world model fields for physics DR. Parity vs MuJoCo C on SO-101, Panda, Go1, G1 at MJWarp tolerances. |
| WS3 interop | zero-copy MPS tensors, event ordering, conformance test | Done: DLPack `kDLMetal` and `from_blob` paths, MTLSharedEvent ordering across Warp/render/torch queues, runtime counters as the no-host-sync conformance instrument (Instruments not available without Xcode). |
| WS4 renderer | tier 0 raster + PBR + shadow maps; tier 1 hybrid RT + denoise + MetalFX; tier 2 path tracer; MaterialX | Tier 0: native Metal, reads physics buffers directly, PBR, model lights + headlight, shadow maps, LOD; parity vs `mujoco.Renderer` (IoU 0.99, depth 0.1 mm, texture corr 0.996), brightness matched (PSNR 19.1). Tier 1: RT soft shadows, AO, mirror reflections from the fragment stage (PSNR 19.7). Not yet: MetalFX denoise/upscale, GI, tier 2, MaterialX compiler. |
| WS5 sensors | lidar/radar/cameras/IMU/contact on shared BVH | Metal RT acceleration structures refit from physics; lidar (vs `mj_ray`: median < 2 mm) and ray-cast depth (= raster depth to 0.1 mm); Isaac-style lidar spec loader; IMU/contact/joint sensors as tensors via MuJoCo sensors. Not yet: beam divergence, multi-return, radar-lite, Isaac USD lidar prims. |
| WS6 data generation | randomizers, annotators, COCO/KITTI writers | Done (GPU randomizers for colours/camera/intrinsics/light/ambient/backgrounds; materials per batch; annotators incl. 2-D/3-D boxes and semantic seg; COCO/KITTI/basic writers). |
| WS7 learner | zero-copy obs/actions, rollout policy in Warp, PPO tuning, profiler | Pixel PPO on MPS (contiguous NCHW buffers: 2.3× faster updates); Warp MLP rollout policy with whole-rollout graph replay and one sync per rollout; rsl_rl KL schedule; time split reported. Not yet: CNN policy in Warp, torch.compile path (measured no gain), physics-only obs for the lift task. |
| WS8 tooling | viewer, debugging, profiling, benchmark suite | Benchmark suite with compute-normalized figures (`orchard.bench.suite`), fidelity benchmark (PSNR/FLIP), race finder (cross-world consistency), gallery (`docs/gallery/index.html`). Not yet: usdview/Storm viewer, powermetrics (needs sudo). |
| WS9 ROS 2 / HIL | network bridge | Not started (out of the hot path; plan says platform-limited). |

## Acceptance tests (plan §2.3)

| test | result |
|---|---|
| Physics: MuJoCo Warp suite green on Metal; trajectory parity vs CPU MuJoCo | 1447/1450 (3 flex deformable failures logged); one-step parity on 4 robots; SO-101 arm trajectory 4.5e-7 over 200 steps |
| Rendering tier 0: silhouette IoU > 0.95, texture corr > 0.99 | 0.994–0.997; 0.996 |
| Rendering tier 1/2 vs Isaac RTX on a shared USD scene | not possible here (no Isaac); vs MuJoCo reference: PSNR 19.7 / FLIP 0.30 |
| Sensors: lidar vs Isaac RTX lidar | vs MuJoCo `mj_ray` instead: median < 2 mm, hit pattern > 97% |
| Pipeline: zero host copies per step, one sync per rollout | verified with runtime counters on physics, render and Warp-rollout loops |
| Training: identical PPO config, identical success rate on SO-101 lift | GPU stack 3M steps: success 0.0 (return 16→127); SB3 baseline on the same reconstructed scene running now (`runs/sb3_baseline/progress.txt`) for the equal-budget comparison |

## Headline throughput (uncontended)

- Isaac Cartpole-RGB equivalent, 1024 envs, 100×100: **48,056 steps/s** (Isaac Lab RTX 4090 published: 50,000).
- Cartpole state, 4096 envs, full PPO loop with the rollout in Warp: **544K steps/s** (Isaac Lab 4090: 510K); rollout only 1.24M.
- SO-101 physics: 541K steps/s at 4096 envs; tier-0 render 705K env-frames/s at 1024×64 px (mjbatch-metal: 72.7K).

## Judgement calls made (to confirm)

1. Product/package name `orchard`; workspace `~/robosim`.
2. The SO-101 lift scene was reconstructed (`assets/so101/scene_box_rl.xml`) because the original lives on another machine; the camera is a 3/4 view I chose. If the original scene and the baseline's `progress.txt` exist, they should replace this.
3. Constraint budget default `njmax=512` per world (MuJoCo Warp's 64 overflowed and produced NaN).
4. Solver iterations are set per model (Metal cannot exit the Newton loop early); cartpole uses 10.
5. MuJoCo/OpenGL light intensities are treated as π× radiance so tier 0 matches MuJoCo's brightness.
6. Tier-1 reflections use MuJoCo's constant reflectance blend rather than Fresnel, to match the reference.
