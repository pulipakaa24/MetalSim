# 3D Gaussian splat rendering in MetalSim (`metalsim/render/splat/`, `shaders/splat.metal`)

Written 2026-10-08, ported to main 2026-10-13. Machine for every timing: M4 Max (40-core GPU), 64 GB.

**Status:** working. Verified against an independent float64 reference on synthetic scenes (tests, always run) and on a
public InteriorGS scene (§4). Renders N cameras per call into tier0-layout buffers and composites `Tier0Renderer`
meshes (robot, objects) with correct occlusion.

**Headline numbers** (640×360):
- InteriorGS showroom (0.97M splats): 0.5–0.8 ms per camera, about 1,300–2,000 cameras/s batched (§5).

## 1. What it is

A compute-shader, tile-based 3DGS rasterizer on native Metal. It renders N cameras per call and writes RGB / depth straight
into learner-owned Warp/MPS buffers in tier0's layout: `rgb` (N, H, W, 3) uint8 and `depth` (N, H, W) float32, planar metres,
0 = nothing. Optional outputs are `depth_expected` and `alpha`. Cameras follow tier0's conventions: MuJoCo camera frames
from `cam_xpos` / `cam_xmat` plus the same 28-float `CameraSpec` per env (intrinsics, pose deltas, `cam_id`). It has a
GPU path (`render(sim)`, event-ordered against BatchSim and torch, no host sync) and a host path (`render_host`).
Robot and objects drawn by a `Tier0Renderer` are composited with correct occlusion (`mesh=`).

| file | content |
|---|---|
| `render/splat/__init__.py` | the public API (§7): `load_scene`, `make_transform`, `floor_plane_transform`, `SplatRenderer` (lazy) |
| `render/splat/formats.py` | loaders → `GaussianCloud`: SPZ v1–3 (gzip), 3DGS `.ply`, SuperSplat compressed `.ply` (InteriorGS), antimatter15 `.splat`, NVIDIA NuRec `.nurec` (§7d); `save_ply` |
| `render/splat/scene.py` | `prepare` / `load_scene`: affine file→sim transform, crop, alpha cull, Morton order, GPU packing, disk cache; `make_transform`, `floor_plane_transform` |
| `render/splat/renderer.py` | `SplatRenderer` (batched, GPU/host paths, mesh composite, profiling, capacity management) |
| `render/shaders/splat.metal` | kernels: cull, project (EWA + SH), emit keys, scans, 5-pass radix sort, blend |
| `render/splat/reference.py` | slow independent float64 reference (per pixel, no tiles, exact depth sort); `kernel="3dgut"` ray-max 3D evaluation (§7d) |
| `render/splat/backend.py` | `SplatSceneRenderer`: a single-camera `render(pos, quat) -> {"rgb", "depth"}` contract (optionally mesh depth) |
| `render/splat/views.py` | camera poses: `yaw_pitch_mat`, `look_at`, `room_views` (InteriorGS room polygons), `orbit_views`; fetched-data paths |
| `tests/test_render_splat.py` | 12 tests (15 cases): formats incl. SuperSplat compressed PLY, transforms, disk cache, float64 reference analytic (CPU); reference parity, batching/determinism, capacity, analytic Gaussian, mesh composite, GPU path, blend variants (Metal) |
| `tests/test_render_splat_nurec.py` | 6 tests: msgpack decoder, synthetic `.nurec` round trip (fp16/fp32, SH order, activations), unsupported-activation error, prepared `.npz` + eps2d hint, 3DGUT reference kernel |
| `tests/test_render_splat_public.py` | 6 tests: on the fetched public scenes (§7b; skipped with the fetch command when absent) InteriorGS decode / frame / crop, Metal vs float64 reference at robot-camera views, dxgl CC0 plain-PLY decode and render, optional NuRec lounge decode; and (offline) the fetch script's missing-login message |
| `scripts/fetch_splat_scenes.sh` | fetches the public reference scenes into `assets/splats/` (git-ignored; §7b) |
| `scripts/splat_bench.py` | `verify` (Metal vs reference, §4) and `bench` (throughput, §5) on the reference scene; `md` writes the tables below |

## 2. Research notes (sources) and the decision

Research pass on 2026-10-08. Summary:

- **Original 3DGS rasterizer** (Kerbl et al. 2023, graphdeco-inria/diff-gaussian-rasterization):
  - 16×16 tiles; 64-bit keys `(tile << 32) | float depth` sorted by a CUB radix sort.
  - 3σ circle radius, 0.3 px² dilation, α ≤ 0.99, skip α < 1/255, stop at T < 1e-4, near cull z ≤ 0.2.
  - Code: `forward.cu:82-111, 229-232, 336-347`, `rasterizer_impl.cu:101-104, 300-308`.
- **gsplat** (Ye et al. 2024, arXiv 2409.06765):
  - The same math, plus an opacity-aware box `extend = min(3.33, sqrt(2 ln(255 o)))` (from StopThePop).
  - `rasterize_mode="antialiased"` multiplies opacity by `sqrt(det Σ / det(Σ + 0.3 I))` (Mip-Splatting 2D filter).
  - Batched cameras share one sort through image-id key bits (`IntersectTile.cu`).
- **Faster binning:**
  - Speedy-Splat (2412.00578): SnugBox (exact α = 1/255 box) + AccuTile, about 2× together.
  - FlashGS (2408.07967): exact ellipse/tile test.
  - Balanced-3DGS / AdR-Gaussian: load balance and early culling.
  - GSCore (ASPLOS'24): subtile skipping, 15.9× vs a mobile GPU.
- **Apple / web:**
  - MetalSplatter: hardware quads, CPU sort.
  - msplat: Metal compute tiles, per-tile bitonic sort ≤ 2048 entries, about 350 fps on M4 Max (resolution not stated).
  - Spark: WebGL2 quads, CPU counting sort over 16-bit depth buckets, LOD in 2.0.
  - antimatter15: CPU 16-bit counting sort.
  - Brush: wgpu compute tiles, GPU radix sort.
  - PlayCanvas WebGPU: compute cull + GPU radix sort + quads. On **M4 Max at 1298×962: 8M splats at 125 fps, 20M at 98 fps**,
    the only published Apple-silicon number at this scale.
- **Hardware rasterization:** 2505.18764 (fragment-shader blending, 3× faster training), GauRast, HTGS. These win when
  there are few cameras and full-screen views at VR resolutions.
- **LOD:** Hierarchical 3DGS (2406.12080), Octree-GS, LODGE, FLoD, Spark-2 LOD trees. All need a precomputed hierarchy,
  which common capture formats (SPZ, 3DGS `.ply`) do not carry.
- **Robotics batching:** GaussGym (2510.15352), > 100k steps/s at 4,096 envs on a 4090, 640×480 at 10 Hz camera.
- **Depth and compositing:** gsplat's "ED" expected depth, RaDe-GS median depth. Compositing practice: opaque mesh first,
  splats depth-tested against it.
- **SPZ:** an open format with an MIT-licensed reference library (`spz`, `load-spz.cc`, `splat-utils.h`); its decode
  formulas are reimplemented in numpy, no code copied. The `0x1` flag means trained antialiased. Positions are stored RUB.
- **SuperSplat compressed PLY:** PlayCanvas engine (MIT) `gsplat-compressed-data.js` formulas, reimplemented.

**Decision: compute tiles + one global radix sort per sub-batch of cameras. Hardware quad rasterization was not chosen.**

Why, from the above and our workload: many cameras per call, 640×360, millions of splats, indoor scenes with heavy overdraw.

1. **Per-pixel early termination** (T < 1e-4) is the main cost lever indoors: with heavy overdraw, a pixel's
   transmittance is used up after a small fraction of the splats in its tile list. Fixed-function blending cannot stop early.
2. **All cameras of a sub-batch share one sort and one dispatch** (gsplat's image-id-in-key trick, in 32 bits).
3. **Median/expected depth and the mesh composite** are a few lines in the blend kernel.
4. **Exact parity with the original math**, which can be checked against a slow reference.

Hardware quads stay the documented alternative for single huge views (§8).

## 3. Method

**Preprocessing** (`scene.prepare`, cached in `$METALSIM_SPLAT_CACHE`, default `~/.cache/metalsim/splat/<name>_<hash>.npz`):
1. Apply the affine `x_world = A x_file + t` exactly to every Gaussian: covariance `A Σ Aᵀ` is formed on the GPU from the
   file-frame quaternion/scales and A. SH directions are mapped back with `normalize(A⁻¹ d)`.
2. Crop to the robust box (§6) and drop α < 1/255 splats, which is lossless.
3. Morton-order the splats so each 256-splat threadgroup is spatially coherent, with a per-chunk bounding sphere.
4. Pack per splat:
   - `g0` float4 (mean, opacity);
   - `g1` 8 × half (quaternion, scales, conservative cull radius);
   - SH row of 64 B (half DC + 45 uint8 bands, the SPZ quantisation, lossless) or float16 bands for .ply.

**Per frame, per sub-batch of ≤ 32 cameras** (one serial compute encoder, all dispatch sizes GPU-driven, indirect):
1. `splat_cull` (chunk sphere test, then per-splat conservative sphere test) → camera bitmask per splat and counts per
   (camera, block).
2. Scan → deterministic record offsets, in splat order per camera.
3. `splat_project`: 3DGS/gsplat EWA projection, with
   - the 1.3·tan(fov) clamp and asymmetric principal point;
   - 0.3 px² dilation, and the antialiased compensation if the file says so;
   - opacity as half;
   - the exact α ≥ 1/255 support box from the conic;
   - SH degree ≤ 3 colour, clamped at 0;
   - count of tiles the support ellipse actually touches (exact ellipse/rectangle minimum test).

   Records are 32 bytes: float mean, conic and depth, half colour and opacity.
4. Scan → key offsets. `splat_emit` writes keys `(camera·tiles + tile) << 16 | depth[23:8]`; each value holds the record
   index (24 bits) with `depth[7:0]` in its top byte.
5. Stable 8-bit LSD radix sort, **5 passes** (value byte first): histogram / scan / scatter with simd-ballot multisplit
   ranking. Depth is log-quantised to 24 bits between near and 500 m (relative step 5.5e-7, 1.6 µm at 3 m).
6. `splat_blend`: one 16×16 threadgroup per tile; the tile's key range by binary search.
   - Batches of 256 splats go to threadgroup memory. Each loaded splat gets a 32-bit mask of the 4×2-pixel subtiles its
     support touches (exact test).
   - Each simdgroup (2 pixel rows) ballot-compacts the batch to the entries touching its subtile row.
   - Front-to-back: α = min(0.99, o·exp(power)), skip < 1/255, stop before T < 1e-4, stop at the mesh depth.
   - C += T·background (the mesh colour where the mesh hit). Median depth is z at the T = 0.5 crossing (mesh depth if the
     crossing is the mesh); expected depth is Σwz / Σw when Σw ≥ 0.5.
   - Writes straight into the output buffers.

**Capacity.** Records and keys are bounded buffers. The host path checks the frame's counters and, on overflow, resizes:
it sizes per-camera need ×1.25, then picks the largest sub-batch under the 24-bit record index and `scratch_budget_gb`
(default 3) and re-renders. The GPU path reads the previous frame's counters without waiting, warns, and resizes for the
next frame. The 1M-splat InteriorGS scene runs full 32-camera sub-batches (§5 table); scenes with several times more
visible splats per camera get smaller sub-batches automatically.

**Determinism.** Every frame is bit-identical across runs and across batch splits (tested). Records are compacted
deterministically and the sort is stable; ties on the 40-bit key resolve by splat index.

## 4. Correctness

**Synthetic scenes** (tests, 4 cameras each, SH degree 0–3, uint8/f16 SH, antialiased on/off): ≤ 0.2 % of pixels differ
by more than 1 level from the float64 reference, max 3 levels. Single-Gaussian analytic check: α within 2e-3, colour
within 1 level (and the reference itself against the closed form, CPU). Composite vs the reference composite: max 3
levels. Batched vs single-camera vs sub-batched: bit-identical. GPU path vs host path: ≤ 1 level and 1e-4 m, with no host
sync in the loop.

**Public reference scene** (InteriorGS 0204_840158, §7b), robot-camera views on the room-polygon grid, Metal vs the
independent float64 reference (`reference.py`; written by `scripts/splat_bench.py verify` + `md`):

<!-- splat-bench:begin -->
**Verification** (InteriorGS 0204_840158, 972,596 splats; 8 views at 640x360, HFOV 88.9; Apple M4 Max, 64 GB, macOS 26.7.1; 2026-10-14 00:16:52; `scripts/splat_bench.py verify`)

| | PSNR dB | SSIM | pixels > 1 level | max level diff | median depth abs err | median-depth valid agree | expected depth p99 rel err | alpha max err |
|---|---|---|---|---|---|---|---|---|
| worst over views | **58.9** | **0.99995** | 0.0148 % | 7 | 4.5e-07 m (p99 1.6e-06 m) | 100.00 % | 6.5e-05 | 0.0084 |
| mean | 58.9 | 0.99998 | 0.0036 % | — | — | — | — | — |

**Throughput** (InteriorGS 0204_840158, 972,596 splats; 640x360, HFOV 88.9; Apple M4 Max, 64 GB, macOS 26.7.1; 2026-10-14 00:00:40; `scripts/gpu_run.sh splat_bench timing -- python scripts/splat_bench.py bench`; 20 frames back to back, best of 3)

| cameras | ms / frame | ms / camera | cameras / s | sub-batch | records | keys | stage ms (one profiled frame) |
|---|---|---|---|---|---|---|---|
| 1 | 0.68 | 0.684 | **1462.9** | 1 | 28,861 | 123,089 | cams 0.0, cull 0.05, scan_records 0.01, project 0.1, scan_keys 0.01, emit 0.1, sort 0.15, blend 0.25 |
| 8 | 3.46 | 0.433 | **2309.1** | 8 | 892,560 | 2,064,279 | cams 0.0, cull 0.07, scan_records 0.01, project 0.38, scan_keys 0.04, emit 0.18, sort 0.52, blend 2.32 |
| 32 | 16.24 | 0.507 | **1970.9** | 32 | 4,231,787 | 10,022,826 | cams 0.0, cull 0.18, scan_records 0.01, project 1.69, scan_keys 0.11, emit 0.62, sort 3.53, blend 10.18 |
| 64 | 31.55 | 0.493 | **2028.3** | 32 | 9,164,023 | 20,695,485 | cams 0.0, cull 0.37, scan_records 0.02, project 2.96, scan_keys 0.24, emit 1.24, sort 6.8, blend 19.49 |
<!-- splat-bench:end -->

The differences come from float32 vs float64 arithmetic, half-precision colour/opacity storage and fast-math `exp`. On
synthetic scenes the 24-bit depth key gives no pixel more than one level off from ordering; the 16-bit key gave 1.3 %
(§6).

**Splat depth vs mesh depth.** Splat median depth is photometric: on captured scenes it sits off the true surface by a
few centimetres where the capture is glossy, thin or poorly observed (NVIDIA NuRec scenes against their proxy meshes:
1.0–3.0 cm median, §7d). Disagreement concentrates at object edges and thin structures. For geometry-consistent depth
(collision, safety layers) take depth from the collision proxy mesh instead (`SplatSceneRenderer(depth_source="mesh")`).

## 5. Performance

**Method:** `SplatRenderer.bench`: frames encoded back to back with one sync, no output copy (the GPU path's cost). Stage
times come from `profile_frame` (each stage in its own command buffer). Taken under MetalSim's GPU queue
(`scripts/gpu_run.sh NAME timing`), device idle at start; the table on this branch is in §4's generated block.

**InteriorGS 0204_840158** (972,596 splats; 2026-10-08, clean window, 20 frames back to back, best of 3; 0.83 m, 22° down,
640×360, HFOV 88.9; poses from the room polygons):

| cameras | ms / frame | ms / camera | cameras / s | sub-batch | records | keys | stage ms (one profiled frame) |
|---|---|---|---|---|---|---|---|
| 1 | 0.76 | 0.765 | **1307.9** | 1 | 128,526 | 298,717 | cull 0.06, project 0.08, emit 0.06, sort 0.18, blend 0.37 |
| 8 | 4.75 | 0.594 | **1684.6** | 8 | 1,303,209 | 2,903,363 | cull 0.07, project 0.46, scan_keys 0.05, emit 0.22, sort 0.71, blend 3.17 |
| 32 | 17.63 | 0.551 | **1814.9** | 32 | 4,936,028 | 11,409,505 | cull 0.18, project 1.53, scan_keys 0.13, emit 0.65, sort 3.7, blend 11.22 |
| 64 | 32.69 | 0.511 | **1958.0** | 32 | 8,038,335 | 20,883,551 | cull 0.35, project 2.65, scan_keys 0.21, emit 1.21, sort 6.79, blend 21.1 |

**GPU memory per renderer.**
- Scene: about 140 bytes per splat with SH degree 3 (InteriorGS: 132 MB), plus the chunk spheres.
- Scratch: records, keys and sort buffers sized from the measured per-camera need; bounded by `scratch_budget_gb`
  (default 3), which caps the sub-batch size.
- Outputs: N×H×W×7 bytes (65 MB at 64 cameras of 640×360).
- `currentAllocatedSize` grows across renderers built in one process (PyObjC over-retains Metal objects,
  `metal_context.py` note), so it is not a per-renderer number.

**Option timings and quality** are measured on InteriorGS by `scripts/splat_bench.py options` (§6 table).

**Published Apple-silicon numbers** (different workloads, for orientation only):
- PlayCanvas WebGPU on M4 Max: 8M splats at 125 fps for one 1298×962 view (156 Mpix/s), possibly with streaming/LOD.
  Ours: 451 Mpix/s on the 1M-splat InteriorGS scene in batches of 640×360 cameras, with 24-bit sorting and exact 3DGS
  rules.
- msplat (Metal compute) reports about 350 fps on Mip-NeRF 360 scenes (resolution not stated).
- No published batched multi-camera Apple numbers exist that we could compare with.

## 6. Options considered (archived)

Quality is PSNR against the default renderer, timing is GPU ms per frame for 16 cameras; both on InteriorGS 0204_840158
(robot-camera views, 640×360), written by `scripts/splat_bench.py options` + `md`:

<!-- splat-options:begin -->
(InteriorGS 0204_840158, 16 cameras at 640x360; Apple M4 Max, 64 GB, macOS 26.7.1; 2026-10-14 00:00:47; `scripts/splat_bench.py options`, 20 frames back to back, best of 3)

| option | splats | PSNR vs default, mean (min) dB | GPU ms / frame (16 cameras) |
|---|---|---|---|
| default | 972,596 | — | 8.47 |
| depth key 16 bit | 972,596 | 54.5 (49.5) | 8.14 (-4 %) |
| no subtile masks | 972,596 | identical | 10.05 (+19 %) |
| blend 2 px/thread | 972,596 | identical | 7.94 (-6 %) |
| SH degree 0 | 972,596 | 29.3 (26.3) | 8.28 (-2 %) |
| SH degree 1 | 972,596 | 32.7 (29.2) | 8.32 (-2 %) |
| antialiased forced on | 972,596 | 22.7 (17.1) | 7.57 (-11 %) |
| crop: none | 972,596 | identical | 8.52 (+1 %) |
| crop: tight (q1-99 box + 10 %) | 972,478 | 68.2 (51.2) | 8.54 (+1 %) |
<!-- splat-options:end -->

| option | verdict | how to re-enable |
|---|---|---|
| depth key 16 bit (4 radix passes instead of 5) | 24 bit default (fidelity first; with 16 bits, synthetic scenes had 1.3 % of pixels > 1 level off, 0.6 % > 3; 24 bit: 0) | `SplatRenderer(depth_bits=16)` |
| crop: none | default crops: lossless by construction (the grown box keeps everything visible from inside), and captures often carry a sparse shell of floaters far outside | `load_scene(crop=None)` or a (2, 3) box |
| crop: tight (q1–99 box + 10 % per axis) | rejected: cuts content seen through windows and doors | `auto_crop_box(grow=0.1)` |
| SH degree 0 / 1 (default 3) | degree 3: glossy and transparent surfaces are view-dependent | `load_scene(sh_degree=0/1)` |
| antialiased forced on (2D-filter compensation) | follow the file flag (SPZ 0x1); unknown (PLY) → off: forcing it changes the image of a model not trained that way | `SplatRenderer(antialiased=True/False)` |
| no subtile masks in blend | masks on (identical image up to ties at the exact 1/255 support boundary; timing in the table above) | `defines={"EXP_NO_MASK": 1}` |
| blend 2 pixels per thread | 1 px per thread (bit-identical, tested; timing in the table above) | `SplatRenderer(blend="2px")` |
| ranges kernel vs binary search per tile | binary search (identical output, one full-key pass less) | removed (superseded) |
| 64-byte records | 32-byte records (identical output, less memory traffic in project / emit) | removed (superseded) |
| hardware quad rasterization (MetalSplatter / Spark / PlayCanvas style) | compute tiles (§2); not built | not implemented |

## 7. How to use

```python
from metalsim.render.splat import SplatRenderer, load_scene, make_transform
from metalsim.render.splat.views import room_views

A, t = make_transform(up="z")                  # file -> sim: up axis ("z" | "y" | "-y"), scale, yaw_deg, translate
s = load_scene("assets/splats/interiorgs/0204_840158/3dgs_compressed.ply", A, t)   # cached; crop="auto" | None | (2,3) box
r = SplatRenderer(s, n_envs, width=640, height=360, hfov_deg=88.9, outputs=("rgb", "depth"))
out = r.render_host(cam_pos, cam_quat=q_wxyz)  # host path: numpy (N, H, W, 3) uint8 / (N, H, W) float32
v = r.render(sim); r.after(v)                  # GPU path: r.out.rgb / r.out.depth are MPS tensors, cameras from sim.d.cam_x*
```

**Compositing robot / objects.** Build a `Tier0Renderer` with the same N, size and outputs `("rgb", "depth")` and pass it as
`mesh=`; the splat renderer takes the mesh renderer's intrinsics and Metal context:

```python
from metalsim.render.tier0 import Tier0Renderer
mesh = Tier0Renderer(model, n_envs, width=640, height=360, camera="cam", outputs=("rgb", "depth"))
r = SplatRenderer(s, n_envs, mesh=mesh, outputs=("rgb", "depth"))
r.render(sim)                                            # mesh pass then splat pass, one queue, no host sync
out = r.render_host(cam_pos, cam_mat=mats, mesh_datas=[d] * n_envs)   # host path: MjData per env for the mesh pass
```

Per pixel the splat blend stops at the mesh depth and the mesh colour fills the remaining transmittance. Every geometry
the mesh renderer draws occludes the splats, so build the mesh renderer from a model that holds only what should be
seen in front of the scene (robot, objects); keep floor / wall collision proxies out of what it draws:
`Tier0Renderer(..., include_planes=False, draw_hfields=False, max_group=k)` with the proxies in a geom group ≥ `k` (not drawn).

**Transforms.** `make_transform(up=, scale=, yaw_deg=, translate=)` covers the usual cases: InteriorGS is z-up metres with
the floor at z = 0 (identity); SPZ and many NuRec volumes are +y up (`up="y"`); COLMAP-frame 3DGS training output is
usually -y up (`up="-y"`). `floor_plane_transform(normal, point)` levels a tilted floor exactly (a shear). Any other
affine (A, t) works; a non-affine map (e.g. a ground warp) is baked into a prepared `.npz` with `PackedSplats.save` and
loaded with an identity transform.

**Single camera** (`SplatSceneRenderer`): `render(pos, quat_wxyz) -> {"rgb", "depth"}`, with `depth_source="mesh"` taking
depth from any mesh renderer with the same signature (geometry-consistent depth for collision / safety logic).

**New formats:** add a function returning `GaussianCloud` to `formats.LOADERS`.

## 7b. Reference scenes (public) and licences

Nothing here is committed: `scripts/fetch_splat_scenes.sh` downloads into `assets/splats/` (git-ignored; override with
`METALSIM_SPLAT_DATA`). `tests/test_render_splat_public.py` and `scripts/splat_bench.py` skip / exit with the fetch
command when the data is absent.

| set | what | licence | access |
|---|---|---|---|
| `interiorgs` (default reference) | InteriorGS scene `0204_840158`: showroom, 972,596 splats, SH 3, SuperSplat compressed `.ply` (60 MB), + `structure.json`; pinned dataset revision and SHA-256 | InteriorGS Terms of Use (below) | gated: accept the terms on Hugging Face once, then `HF_TOKEN` or a Hugging Face CLI login |
| `dxgl` | `modern_arm_chair.ply` from dxgl/multiview-datasets: nerfstudio splatfacto, standard 3DGS `.ply`, SH 3, trained on renders of a Poly Haven model | CC0-1.0 | open |
| `nurec_lounge` (optional) | NVIDIA PhysicalAI-Robotics-NuRec `zh_lounge/usd/export_last.nurec`: indoor lounge, 3DGUT, 118 MB | CC-BY-4.0 (attribution: NVIDIA) | gated by a contact-information click-through; same token handling |

- **Hugging Face access** (InteriorGS, NuRec). Before downloading, the fetch script checks access with a one-byte
  request. On failure it prints the next step instead of a curl error, and exits with 3:
  - no login on this machine: `hf auth login`, or `HF_TOKEN`;
  - HTTP 401: the token is invalid;
  - HTTP 403: this account has not accepted the dataset's terms; the message links the terms page.

  The token is read from `HF_TOKEN` or the file that `hf auth login` writes, and is never printed.
  (2026-10-14: the first version deleted its auth-header file early, because a zsh EXIT trap set inside a function
  fires when that function returns, so curl failed with rc 26 on both Macs.)
- **InteriorGS** (SpatialVerse / Manycore, https://huggingface.co/datasets/spatialverse/InteriorGS; Miao et al., arXiv
  2510.21307). 1,000 indoor scenes as `3dgs_compressed.ply`, the SuperSplat/PlayCanvas chunk-quantised PLY
  (`splat-transform`): per 256-splat chunk position, log-scale and colour ranges; per splat 11-10-11 position and
  log-scale, 2-10-10-10 smallest-three quaternion, 8-8-8-8 colour + opacity, uint8 SH (b·8/255 − 4). Decoded by
  `formats.load_ply_compressed`; loads in 0.1–0.34 s. Frame: z-up metres, floor at z = 0 (README "XYZ = (Right, Back,
  Up)"; `structure.json` right-handed z-up), so the sim transform is the identity. No antialiasing metadata: classic
  0.3 px dilation. Splats below the floor are expected (glossy-floor reflections are fitted as a mirrored room).
  **Terms of Use** (read 2026-10-08): non-commercial research and education only; no redistribution in whole or in part;
  cite the dataset; no warranty, the user indemnifies Manycore; access can be revoked; a researcher's for-profit employer
  is bound too; PRC law. The owner approved its use as the reference (2026-10-13: MetalSim is an open-source research
  project). Consequences: each user fetches it with their own Hugging Face account; nothing from it (data, renders,
  derived files) is committed; only measured numbers are.
- **dxgl/multiview-datasets** (https://huggingface.co/datasets/dxgl/multiview-datasets, file host `dx.gl/splat/`): CC0-1.0,
  ten object splats of CC0 Poly Haven models. Used for the plain 3DGS `.ply` path on a real trained file.
- **Other public sources checked** (2026-10-13): NVIDIA PhysicalAI-Robotics-NuRec (Hugging Face, CC-BY-4.0, gated,
  78.5 GB of NuRec scenes; `zh_lounge` is the optional `nurec_lounge` set above, the only scene there shipped as a
  plain `.nurec`; the others are USDZ); the original 3DGS pretrained models (Gaussian-Splatting
  License, non-commercial); Mip-NeRF 360 / Tanks and Temples trained splats on Hugging Face (mostly the 3DGS licence or
  unstated). InteriorGS was chosen as the reference because it is indoor, robot-scale, z-up metric with floor at 0 and
  ships room polygons for placing cameras.

## 7d. NVIDIA Omniverse NuRec scenes (`.nurec`, 3DGUT)

Date 2026-10-09.

**Container.** A NuRec USDZ holds `default.usda` → `gauss.usda` (a `Volume` with `omni:nurec:isNuRecVolume`, two
`OmniNuRecFieldAsset`s pointing at the `.nurec`, `omni:nurec:ccmR/G/B` colour-correction rows (identity in all 7 scenes seen),
`omni:nurec:offset`, crop bounds, `proxy` rel to the mesh) and `mesh.usd` (proxy/collision mesh, its own `upAxis`/`metersPerUnit`).

**`.nurec` layout** (decoded by `formats.load_nurec`; reverse-engineered from the files, version 0.2.576): gzip → msgpack
`{"nre_data": {"version", "model": "nre", "config", "state_dict"}}`.
- `config.layers.gaussians`: `name "sh-gaussians"`, `density_activation sigmoid`, `scale_activation exp`, `rotation_activation
  normalize`, `precision 16` (tensor dtype), `particle {density_kernel_degree 2, density_kernel_planar false,
  density_kernel_min_response 0.0113, density_kernel_density_clamping true, radiance_sph_degree 3}`, `transmittance_threshold 1e-4`.
- `config.renderer`: `3dgut-nrend` (unscented-transform projection, `global_z_order`, `kbuffer` mode with k = 0, rolling-shutter
  iterations, `min_projected_ray_radius` 0.548 = √0.3 used for bounding only), `appearance_embedding` / `background`: skipped.
- `state_dict` (raw little-endian bins + `<name>.shape`, prefix `.gaussians_nodes.gaussians.`): `positions` (N,3), `rotations`
  (N,4) **wxyz**, `scales` (N,3) log, `densities` (N,1) logit, `features_albedo` (N,3) SH DC, `features_specular` (N,45)
  **coefficient-major / channel-last**, `n_active_features` int64 = SH degree, `extra_signal` (N,0).
- Mapping onto `GaussianCloud` is direct (3DGS semantics: colour = 0.5 + SH). The decoder is a 60-line msgpack reader that maps
  bins zero-copy; the gzip stream is decompressed in 64 MB chunks into an unlinked temp file (in `METALSIM_SPLAT_CACHE`) and
  memory-mapped, so the float16 SH (1.5 GB for 16 M splats) stays page-cache backed. A 16.35 M-splat scene (1.93 GB) loads
  in 0.8 s, peak footprint 1.5 GB.

**How the conventions were verified** (7 NVIDIA warehouse / lab scenes):
- SH order: mean r/g/b correlation within a band 0.85–0.90 read coefficient-major vs −0.06…0.04 channel-major.
- Quaternion order: disc-shaped floor splats' thin axis vs the floor normal, median |cos| 0.68–0.79 (wxyz) vs 0.61 (xyzw); and
  visually: xyzw renders are streaky/smeared, wxyz are sharp. 3DGRUT's PLY import/export passes `rot_0..3` through, i.e. the
  3DGS wxyz order.
- Frame: Gaussians are in the proxy mesh's frame. Some scenes are **+y-up** although the stage says `upAxis = "Z"` (no
  transform authored; use `make_transform(up="y")`); others have +z-up Gaussians and a +y-up `mesh.usd` with an
  `over "mesh"` x+90° transform in `gauss.usda`. `mesh.usd` claims `metersPerUnit 0.01`, but its points are metres (they
  match the Gaussians); Omniverse does not rescale references, so the 0.01 is ignored there too.

**3DGUT vs our EWA rasterizer.** 3DGUT (NVIDIA 3DGRUT) evaluates each particle in 3D: the response along a pixel ray is the
Gaussian's maximum on the ray, exp(−½|o_g × d_g|²/|d_g|²) with o_g, d_g in the particle's unit frame; no 2D linearisation and
**no 0.3 px² dilation**; a hit needs response ≥ `min_response` (0.0113 ≈ 3σ) and α = min(0.99, σ·response) ≥ 1/255; the order is
the global per-particle depth (as 3DGS). Changes with an EWA rasterizer: (1) the 0.3 px² dilation must be off (the model never
saw it: it fattens the many sub-pixel particles); (2) the 2D-projected ellipse approximates the ray-max response, exact to first
order, worse for large particles close to the camera; (3) the 3σ cutoff vs our α-based support box (equivalent within
<1/255 α); (4) rolling shutter / distortion are not modelled (pinhole only). Measured against the float64 reference with the
3DGUT kernel (`reference.render(kernel="3dgut")`, a 16 M-splat warehouse, 4 robot-camera views at 320×180, 1.2–2.7 M visible):

| Metal EWA vs 3DGUT reference | PSNR median (min) | pixels > 8 levels | median-depth |Δ| | α mean |Δ| |
|---|---|---|---|---|
| **eps2d 0 (NuRec default)** | **45.2 dB** (32.9: a close-up of large particles) | 0.8 % | 0.001 mm | 1.1e-4 |
| eps2d 0.3 (3DGS default) | 24.1 dB (22.6) | 18.7 % | 18.8 mm | 2.5e-4 |
| float64 EWA, eps2d 0 | 45.2 dB (32.9) | 0.8 % | 0 | 1.1e-4 |

Metal = float64 EWA to the digit, so the remaining gap is the EWA linearisation itself. **Decision:** `load_nurec` sets
`meta["eps2d_hint"] = 0` and `antialiased = False`; `SplatRenderer(eps2d=None)` (default) takes the scene's hint, else 0.3
(every other format unchanged). A 3D-evaluation blend kernel (exact 3DGUT) is the documented next step if close-range
quality matters (cost: a 3×3 matrix per record instead of the conic).

**Colour.** `ccm` rows are identity in every scene and `default.usda` sets `rtx:post:registeredCompositing:invertToneMap 1`,
i.e. Isaac treats the NuRec radiance as display-referred: SH colour is written as is (no tone map). No Isaac Sim reference
renders were available; the check is splat vs proxy mesh and vs the 3DGUT reference above.

**Splat vs proxy mesh** (tier0 depth of the 2 cm vertex-clustered proxy mesh, 16–40 robot-camera views per scene): median
|Δ| 1.0–3.0 cm in 6 scenes, 4.3 cm in an outdoor lot; bias +0.5…+1.8 cm; 91–97 % within 10 cm (outdoor lot 74 %); 0 % holes.

**Performance** (16.3 M splats after the crop; clean window, 2026-10-09): host path p50/p90 over 40 robot-camera poses
**9.0 / 19.4 ms** at 640×360, **10.7 / 20.6 ms** at 640×480 HFOV 79 pitch 0, 7.6 / 13.8 ms at pitch 30; 8 cameras batched
12.0 / 13.2 / 7.0 ms per camera. Many large low-opacity 3DGUT particles make it costlier per splat than 3DGS captures.

**Licence of the scenes measured here:** the 7 scenes were not from the CC-BY-4.0 PhysicalAI-Robotics-NuRec release and
their licence is unknown, so nothing from them is in the repo. Use the CC-BY-4.0 release (§7b) for anything shared.

**Loading a scene:** `formats.load("x.nurec")` (any of `LOADERS`), or a pre-transformed `PackedSplats` `.npz` through
`load_scene` (needed when the file→sim map is not affine, e.g. a ground warp; A/t must then be identity).

## 8. Limits and next steps
- **Pinhole only**: no fisheye or rolling shutter. The EWA linearisation shows the usual 3DGS artefacts for large splats
  near the image border; the 1.3·tan(fov) clamp is as in 3DGS/gsplat.
- **No LOD.** Every splat that passes the cull is projected and sorted for every camera, so cost grows with the visible
  splat count; very large scenes or wide outdoor views would need a hierarchy (H3DGS / Spark-style), which common
  capture formats do not carry.
- **Popping.** Order is by splat centre depth per tile, as in 3DGS: interpenetrating splats can swap order between views.
  StopThePop's per-pixel sort would fix it, at about 4 % cost there.
- **Sub-batch size** is bounded by the 24-bit record index and the scratch budget (32 cameras on InteriorGS). More cameras run as more sub-batches in the same command buffer; there is no hard cap on N.
- **Lighting.** Splat colours are baked capture radiance. There is no relighting and no shadow from the robot onto the
  splat. Composited meshes keep tier0's lighting, so a robot or object can look "pasted in".
- **Mesh composite** uses tier0's z-buffer and far plane: mesh geometry beyond the mesh renderer's far plane is not
  composited.
- **Formats.** `.sog` (PlayCanvas: WebP-coded attributes with codebooks) and SPZ v4 (zstd multi-stream) are not
  implemented: v4 raises a clear error. Both plug into `formats.LOADERS`.
- **Next optimisations**, by measured cost share (InteriorGS stage times, §5: blend and sort dominate):
  1. **Blend** (the two-pixel variant is timed in §6). Options: lower-precision evaluation (an fp16
     power/exp path) and an exact early-out per simdgroup.
  2. **Sort**: a single-pass "onesweep" radix sort with a local reorder before the scatter, for coalesced writes.
  3. **Project**: SH evaluation could be skipped for records whose support misses every tile.
