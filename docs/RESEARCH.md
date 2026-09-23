# Research findings (2026-09-22)

Consolidated from five research passes run at project start. Facts are labelled
**verified** (checked in source or run here), **reported** (third party), or **uncertain**.

## Physics on Apple GPUs

- NVIDIA Warp v1.17.0 (2026-08-31) has **no** Metal backend and no upstream PR for one. The
  backend lives in the innate-inc fork (https://github.com/innate-inc/warp, author David Dobas;
  repos moved from DavidDobas/* on 2026-09-19). `warp-metal` overlay wheels are on GitHub
  releases (v1.17.0.3, 2026-09-20), **not on PyPI**. We build the fork's `main`
  (1.18.0.dev20260917, includes the >32-DOF register-Cholesky fix) from source. verified.
- MuJoCo Warp 3.14.0 (2026-09-22) needs the DavidDobas `metal` patch (4 files: `is_cpu` gates
  that meant "not CUDA", dense Cholesky ≤64 DOF off-CUDA, in-place `tile_matmul`). Applied on
  branch `metal` of `upstream/mujoco_warp`. Full suite on Metal here: **1447 passed, 3 failed**
  (`flex_test::FlexConstraintTest::test_constraint_parity2/3`, and the known
  `io_test::test_put_data_nefc_zero_dense`), 39 skipped, 174 s. verified.
- mjlab 1.6.0 pins `mujoco~=3.11`, `mujoco-warp~=3.11`, Python <3.14; `MJLAB_AGENT_DEVICE=mps`
  is only on the DavidDobas `metal` branch. Newton 1.6.0 is CPU-only on macOS officially;
  untested on `metal:0`. Isaac Lab 3.0.0-EA (2026-09-16) targets Isaac Sim 6.1, Warp 1.16,
  Newton 1.5.2, backend `newton_mjwarp`. reported.
- Genesis 1.4.1 has an official Metal path via its Quadrants compiler; jax-metal is unmaintained.
- Metal constraints (fork docs): no float64; tile kernels forward-only; 32 KB threadgroup memory;
  32-bit atomics only (64-bit add split); no spinlocks; Apple compiler miscompiles large unrolled
  matrix products (Warp works around its own matmul). verified in docs.

## PyTorch MPS interop

- torch 2.14.0 (2026-09-02), macOS 14+ wheels for cp310–cp314. verified.
- **MPS storage data pointer is the `id<MTLBuffer>`**; byte offset is `storage_offset`.
  DLPack maps MPS ↔ `kDLMetal`; export puts the buffer handle in `data` and the offset in
  `byte_offset`; import calls `at::from_blob(data, ..., device MPS)` with no validation. So a
  foreign buffer becomes a zero-copy MPS tensor (MLX ships the same convention). verified, and
  exercised by `tests/test_interop.py` both via DLPack and via `at::from_blob` in our extension.
- `torch::mps::get_command_buffer()` (ends kernel coalescing) + `encodeWaitForEvent` /
  `encodeSignalEvent` + `torch::mps::commit()` on `get_dispatch_queue()` gives event ordering
  on torch's queue. verified (tests prove the wait blocks).
- Warp and torch hold the **same `id<MTLDevice>` object** on this machine. verified.
- No CUDA-graph equivalent on MPS; inductor MPS backend is an elementwise prototype. Launch
  count is the only lever for the learner. reported.
- MLX 0.32.2 has zero-copy Metal DLPack with torch (PR #3531) but no Python event/queue API.

## Metal and PyObjC

- Runtime MSL compilation works with Command Line Tools only; the offline `metal` compiler
  needs Xcode.app + the Metal Toolchain component. verified here. Ship MSL source.
- M4 Max = Apple9 family (hardware RT, `supportsRaytracing`), Metal 4 objects available on
  macOS 26 (`MTL4CommandQueue`, `MTL4ArgumentTable`, `MTLTensor`, `MTL4Compiler`,
  `MTL4MachineLearningCommandEncoder`); Metal 3 APIs still work and interoperate through events.
- MetalFX temporal, denoised (`MTLFXTemporalDenoisedScaler`, needs color/depth/motion/normal/
  roughness/diffuse+specular albedo) and frame interpolation all work **headless** on offscreen
  textures. verified here (0.55 ms for 320×240→640×480).
- `encodeWaitForEvent` only between passes; MTLEvent orders across queues on one device;
  MTLSharedEvent also across device objects/processes and from the CPU.
- Page size is 16 KB; `newBufferWithBytesNoCopy` needs page alignment.
- Tile shaders/imageblocks: 32 KB tile memory, memoryless attachments; deferred lighting in one
  pass is Apple's reference structure.
- PyObjC 12.2.2 covers Metal 4 and MetalFX; bug #690 over-retains instance `new*` methods (leak,
  fixed in unreleased 12.2.3); `contents().as_buffer(n)` gives zero-copy numpy; ~41 µs per
  Python-driven single-dispatch command buffer. verified.
- No Apple-published rays/s figures; Blender Open Data medians: M4 Max 40c 5,208 samples/min,
  M5 Max 8,113, M5 Ultra 13,113. reported.

## Scene layer

- `usd-core 26.8` wheels for cp39–cp314 (universal2) include UsdPhysics, UsdShade, UsdLux,
  UsdSemantics (LabelsAPI), UsdRender; **exclude** Hydra/Storm/HgiMetal, usdview, MaterialX
  plugins. Storm-on-Metal needs a self-built OpenUSD with imaging + MaterialX. reported.
- MaterialX 1.39.5 wheels (cp39–cp314 arm64) include the **MSL shader generator**
  (`PyMaterialXGenMsl.MslShaderGenerator`), OpenPBR 1.1.1, standard_surface, and an
  MDL *generator* (no MDL importer). reported.
- Isaac Sim 6.x importers are built on `mujoco-usd-converter` 0.5.0 / `urdf-usd-converter`
  0.3.3 (newton-physics org, Apache-2.0, Python <3.13): field mappings documented in
  RESEARCH notes above (gainprm/biasprm → DriveAPI, frictionloss → PhysxJointAPI.jointFriction,
  armature → PhysxJointAPI.armature, `mjc:` attrs retained). SimReady Foundation defines
  runtime variants (PhysX/Newton/MuJoCo) and semantic/nonvisual-material capabilities.
- Cycles: Metal backend with MetalRT on M3+; Hydra delegate builds against USD ≥25.11; no
  official macOS-Hydra statement. `gatling` (pablode) is a Hydra path tracer with Apple M3+
  hardware RT support. reported.

## Published standards to test against

- Isaac Lab (Isaac Sim 6.1, RTX 4090): Cartpole-Direct 4096 envs 1.10M/910K/510K steps/s
  (step / +inference / +train); **Cartpole-RGB-Camera 1024 envs, 100×100 tiled: 50K/45K/32K**;
  G1 rough 4096: 94K/88K/82K; Shadow repose 8192: 200K/190K/170K.
- MuJoCo Playground (A100): CartpoleBalance 718,626; Go1 joystick 417,451; PandaPickCube
  140,386; pixels via Madrona: CartpoleBalance ~403K, PandaPickCubeCartesian ~37K (4090).
- MJX humanoid: M3 Max CPU 650K SPS; A100 950K (batch 8192).
- mjwarp-testspeed primitives render N=512 on 4090: 522,101 steps/s (PR #1113).
- Physics tolerances: MJWarp forward tests 5e-4 (smooth), 1e-3 (step), 5e-3 (implicit);
  solver tests atol=rtol=0.1 on qacc/efc_force; Newton Menagerie FK atol 2e-6, dynamics
  1e-6..5e-4 per robot.
- Rendering: Isaac's only published thresholds are PSNR ≥25 / SSIM ≥0.90 (nurec harness);
  NVIDIA FLIP via `flip-evaluator` 1.7; glTF Sample Assets and MaterialX test suite as scenes.
- Sensors: Isaac RTX lidar schema attributes (OmniLidar, `OmniSensorGenericLidarCoreAPI`,
  emitter-state arrays; per-material `omni:simready:nonvisual:*` tokens); IMU/contact/camera
  attribute lists recorded in the agent report.
- RL configs: Isaac Lab rsl_rl PPO (Cartpole 16 steps/env, 150 iters, MLP 32×32; Anymal rough
  24 steps/env, 1500 iters, MLP 512/256/128); Playground brax PPO configs per task.

## Addendum (2026-09-22, evening)

- NVIDIA's `mujoco-usd-converter` / `urdf-usd-converter` (the basis of Isaac Sim 6.x importers)
  cannot be installed here: their dependency `usd-exchange` has no macOS wheels (verified:
  only manylinux and win_amd64 for 3.0.0). `orchard.scene.mjcf_to_usd` / `usd_to_mjcf` are the
  in-house equivalents, following the documented Isaac field mapping.
- MuJoCo Warp `put_data` defaults `njmax=64` constraint rows per world; contact-rich scenes
  overflow silently (NEFC flag) and the elliptic Newton solver then yields NaN. Verified with the
  cross-world consistency tool; `njmax=512` fixes it. Worth an upstream issue.
