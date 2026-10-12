# Render coverage: what the simulator simulates vs. what the renderers draw

Date 2026-10-12. Reference state: `origin/metalsim-mpm` (565ae2f). Machine: M4 Max.

**Question.** Policies that learn from cameras only see what the renderer draws. If the simulator simulates
something that the renderer does not draw, or draws in the wrong place, the policy cannot observe it. For each simulated
feature, this audit asks whether it shows up in RGB, depth and segmentation for every MetalSim renderer.

**Method.**
1. Listed the simulated features from the code and tests: MuJoCo / MuJoCo Warp model elements, flex deformables,
   PhysX-port cloth, Newton (XPBD / VBD / MPM), USD import, and per-world randomization.
2. Traced each renderer's input path:
   - `scene_tables.build_scene_tables` is the only scene compiler. Tiers 0, 1 and 2 and the ray tracer all use it.
   - Per-frame inputs are `geom_xpos`, `geom_xmat`, `cam_xpos`, `cam_xmat`, `light_xpos` and `light_xdir`. The only
     other per-frame input is `geom_size`, and only for the `tslot*` terrain boxes.
3. Rendered probe scenes with `scripts/diagnostics/render_coverage_audit.py`. MuJoCo's own `mujoco.Renderer` is the
   reference: it draws every simulated visual, and its segmentation gives each object's silhouette. The script measures
   the fraction of each reference silhouette that is covered in MetalSim's RGB (pixel differs from the clear colour),
   depth (> 0) and segmentation (the right geom id).
   - Command: `scripts/gpu_run.sh render_coverage check 10 -- .venv/bin/python scripts/diagnostics/render_coverage_audit.py`.
   - Raw numbers are in `docs/measurements/render_coverage.json`.
   - Resolution 160×120. Tier 2 used 16 spp × 2 passes with centre sampling. `raydepth` is `RayTracer.make_depth_camera`,
     which uses the same acceleration structures as the lidar, radar and tier-1 rays.

Where a gap cannot be measured on this machine (Newton, MPM, the PhysX-port cloth), the finding comes from code reading:
there is no code path. Those rows are marked "code".

**Work in progress elsewhere.** These are listed so the table reflects them, not audited as shipped:
- Uncommitted render changes in the main checkout (another session):
  - `Tier0Renderer(far=...)` makes the 10 m far plane configurable (landed by workstream C with the same signature, §3c).
  - `textures="native"` (`render/textures.py`) adds full-resolution textures with mips, anisotropic filtering and
    per-material wrap.
  - The 3D Gaussian splat renderer (`render/splat/`, `shaders/splat.metal`).
- MPM particle rendering: being built by another agent (branch `metalsim-particle-render`, no commits yet).

## 1. Coverage table

Status per renderer:
- ✅ drawn correctly
- ◐ partial / wrong
- ❌ invisible
- n/a: not applicable

Columns:
- **T0** = tier 0 raster.
- **T1** = tier 1 (raster plus ray-traced shadows, AO and reflections).
- **T2** = tier 2 path tracer.
- **RT** = ray-traced sensors (depth camera, lidar, radar), which share T1/T2's acceleration structures.
- **Ann** = Replicator annotators (`metalsim/replicator`). These derive everything from the renderer's id, depth and
  normal buffers or from geom poses, so they inherit the renderer's coverage.
- **Splat** = splat composite. It is WIP: the splat background is RGB and depth only, and the robot and objects come
  from a T0 mesh pass.

### 1a. MuJoCo / MuJoCo Warp model elements

Measured coverage of the reference silhouette is shown as rgb / depth / seg.

| feature | T0 | T1 | T2 | RT | Ann | evidence |
|---|---|---|---|---|---|---|
| sphere, capsule, ellipsoid, cylinder, box, mesh (convex hull) | ✅ | ✅ | ✅ | ✅ | ✅ | 0.996–1.0 / 0.996–1.0 / 0.996–1.0; depth error ≤ 0.7 mm |
| finite plane | ✅ | ✅ | ✅ | ✅ | ✅ | 1.0 / 1.0 / 1.0 |
| infinite plane (`size="0 0"`, every locomotion floor) | ✅ | ✅ | ✅ | ✅ | ✅ | **Fixed (workstream C, §3c).** Drawn to the far plane under each env's camera (T0/T1), to the sensor range (RT/T2), with MuJoCo's plane texture mapping. Ground coverage of MuJoCo's ground pixels 0.369 → 1.0 (T0/T1/T2/RT). |
| hfield | ✅ | ✅ | ✅ | ✅ | ✅ | **Fixed (§3c).** Top surface plus MuJoCo's sides and base: coverage 0.58 → 1.0 (side view) and 0.90 → 1.0 (from below), every renderer; geometry equals `mj_ray` on 400 random rays (max 1e-15 m). |
| SDF geom (`type="sdf"`, plugin or mesh SDF) | ✅ | ✅ | ✅ | ✅ | ✅ | **Fixed 2026-10 (B)**: drawn from its mesh as MuJoCo does. Was 0.0 / 0.0 / 0.0; now 1.0 / 1.0 / 1.0 in T0, T2 and the ray-traced depth camera, depth error 0. |
| sites (group 0–2, visible in MuJoCo by default) | ✅ | ✅ | ✅ | n/a (by design) | ✅ | **fixed (workstream A, §5)**: 0.993–1.0 / 0.993–1.0 / 0.993–1.0, depth error ≤ 0.4 mm (was 0.0). Instanced slots posed from `site_xpos` / `site_xmat`. The ray-traced sensors leave sites out by default, as `mj_ray` does (a lidar's own mounting site would occlude every beam); `visual=mujoco.MjvOption()` includes them. Isaac draws no sites. |
| spatial tendon with `width` (cables, wires, MuJoCo tendon-driven hands) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0, depth error 0 (was 0.0); tendons wrapping a cylinder, a sphere and across a pulley in §5. Capsules per wrap segment (MuJoCo's tessellation, half width inside a wrapping geom, none across a pulley). Not reproduced: MuJoCo's catenary for a slack two-site spring tendon and the limit-impedance recolouring of default-grey tendons. Isaac draws no tendon geometry. |
| flex 2D (cloth) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0 of 277 px, depth error ≤ 0.1 mm (was 0.0). MuJoCo's skin: smoothed surface ±radius, sides closed. |
| flex 3D (soft volume) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0 of 152 px, depth error 0 (was 0.0). Shell offset by +radius (flat-shaded when `flatskin`). |
| flex 1D (cable) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: open cylinders per edge (`mjVIS_FLEXEDGE`). The first audit's cable sat just above the image (0 px in MuJoCo too); moved mid-image it is 1.0 / 1.0 / 1.0 in every renderer. |
| skin (bone-skinned mesh) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0 of 236 px, depth error 0 (was 0.0). Linear blend skinning, area-weighted normals, `inflate`. |
| transparency (`rgba` alpha < 1, material opacity, USD `opacity`) | ✅ | ✅ | ✅ | ◐ (opaque) | ✅ | **Fixed 2026-10 (B)**. T0/T1 blend as MuJoCo does: objects sorted per env, drawn front-to-back then back-to-front, back faces culled. T2 treats alpha as presence, like a USD opacity in Isaac's path tracer. Red box behind 25 % blue glass, colour change the glass makes over the box: MuJoCo (−64, 0, +65); T0/T1 before (−146, 0, +146), after (−64, 0, +64); T2 before (−122, 0, +125), after (−53, 0, +53). Annotations default to MuJoCo's: segmentation and depth of the glass (1.904 m, equal to MuJoCo). `transparent_ids="behind"` gives Isaac RTX Real-Time's opacity behaviour (seg / depth from what is behind). Lidar and depth camera still treat it as opaque. |
| geometry beyond 10 m | ✅ (`far=`) | ✅ (`far=`) | ✅ | ✅ (range-limited) | ✅ | **Fixed (§3c).** `Tier0Renderer(far=...)` (default 10 m, unchanged); boxes at 15 m and 40 m 0.0 → 1.0 with MuJoCo's far plane; reversed-z depth keeps the error relative at any far. |
| geom groups ≥ 3 hidden | ✅ | ✅ | ✅ | ✅ | ✅ | Matches MuJoCo's default `geomgroup` |
| `geom_rgba` / `mat_rgba` | ✅ | ✅ | ✅ | n/a | ✅ | |
| emission | ✅ | ✅ | ✅ (not sampled as a light) | n/a | | |
| mesh normals (hard edges of CAD / OBJ / STL meshes) | ✅ | ✅ | ✅ | n/a | normals annotator ✅ | **Fixed 2026-10 (B)**: compiled `mesh_normal` / `mesh_facenormal`, and on meshes without UVs a corner more than ~37° (dot < 0.8) off its face takes the face normal (`mjr_uploadMesh`). Mesh-box luminance std: before 43.7 (T0) / 43.5 (T2), after 1.1 / 3.1, MuJoCo 2.9; MAD vs MuJoCo 37.4 → 1.3 (T0). |
| 2D texture on box / plane / hfield / mesh with UVs | ✅ | ✅ | ✅ | n/a | | MuJoCo's coordinates (box faces, plane, object-linear mapping for meshes without UVs and heightfields), `texrepeat` / `texuniform` as `settexture`; sRGB textures decoded as MuJoCo's GL_SRGB8. Box checker corr 0.866 (T0) / 0.909 (T2), unchanged: MuJoCo samples with mipmaps, the atlas has none. |
| mesh UV orientation | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: the atlas stores `tex_data` rows unflipped, so t = 0 is the first row, as MuJoCo uploads it (its OBJ loader stores 1 − v). The USD importer now writes `1 − t` for `st`. MuJoCo top/bottom blue/red, T0 and T2 now blue/red (were red/blue); T0 corr 0.50 → 0.90. |
| 2D texture on cylinder / capsule | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: MuJoCo's builtin cylinder / disk / half-sphere coordinates. Corr vs MuJoCo: cylinder — / 0.03 → 0.93 / 0.95, capsule 0.03 / 0.005 → 0.93 / 0.95 (T0 / T2). |
| 2D texture on sphere / ellipsoid | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: MuJoCo's sphere tessellation and coordinates (`vis.quality`). Corr −0.94 / −0.97 → 0.95 / 0.97 |
| cube texture (`type="cube"`) on box / sphere | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: OpenGL cube-map lookup of MuJoCo's object coordinates, with one face or a 6-face strip. Corr 0.08 / 0.06 → 0.97 / 0.98 |
| texture roles other than RGB (normal, roughness, metallic, ORM, emissive, opacity layers) | ❌ | ❌ | ❌ | n/a | | Only `mat_texid[mjTEXROLE_RGB]` is read (code) |
| `texuniform`, `texrepeat` | ✅ | ✅ | ✅ | | | Code, plus `test_texture_pattern_parity` corr 0.996 |
| skybox | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: the first skybox texture is drawn as MuJoCo's cube map of the view direction (d.x, d.z, −d.y). In T0 it is the background; in T1 it is also what reflections miss into. In T2 it is the radiance of every escaping ray, so it lights the scene; a `dome` or `set_environment` replaces it. Background luminance std 0.8 / 2.8 → 17.8 / 18.3 (MuJoCo 17.8), corr 0.996 / 0.988, MAD 0.1 / 0.2. `set_clear_color` turns it off. Not drawn: MuJoCo's haze over infinite planes. |
| directional / point / spot lights moving with bodies | ✅ | ✅ | ✅ | n/a | | `light_xpos` / `light_xdir` per world |
| more than 8 lights | ✅ | ✅ | ✅ | n/a | | **Decided 2026-10 (B): every light, as Isaac / RTX** (no light limit). MuJoCo's fixed-function GL path stops at 8. Patches 9 and 10: T0 0.0 / 0.0 → 42.8 / 42.6 (T2 42.4 / 42.4). `Tier0Renderer(max_lights=8)` restores MuJoCo's cap. MuJoCo also lights only directional and spot lights; we also light point lights. |
| spot-light shadows | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**. T0 renders a perspective shadow map per shadow-casting spot light, over the full cone; MuJoCo covers cutoff × shadowscale. T1 traces a ray to the light. Plate shadow ratio: MuJoCo 0.00; T0 1.04 → 0.054; T1 1.04 → 0.052; T2 0.033. Spot + directional together: MuJoCo 0.20, T0 0.81 → 0.21. |
| point-light shadows | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: T0 renders six 90° cube faces per point light; T1 traces a ray. T0 1.02 → 0.053, T1 1.02 → 0.051, T2 0.035. MuJoCo does not light point lights at all; Isaac RTX shadows them. |
| directional shadows | ✅ (every caster) | ✅ (every caster) | ✅ | | | 0.052 / 0.049 / 0.041 vs MuJoCo 0.0 (unchanged). Every further shadow-casting directional light now gets its own map in T0 and its own soft ray in T1. With a DR light on, the first model directional light stays unshadowed, as before. |
| image lights (`mjLIGHT_IMAGE`) | ◐ | ◐ | ◐ | | | Mapped to directional (`build_lights`, code) |
| area lights (USD Rect / Disk / Cylinder, SphereLight radius) | ◐ | ◐ | ◐ | | | Imported as spot / point lights with far-field radiometry (§3c); no area sampling, so shadows stay hard. SphereLight radius still dropped. |
| debug visuals (joints, contacts, COM, inertia, equality constraints, perturbations) | n/a | n/a | n/a | | | Not camera content in Isaac either |

### 1b. Per-world randomization (things the physics can vary per env)

| feature | T0/T1 | T2 | RT | evidence (code) |
|---|---|---|---|---|
| per-env colour (`Randomizer.colors`, event-term colour) | ✅ | ✅ | n/a | `t_colors` (N, G, 4) |
| camera pose, intrinsics, DR light, ambient, backgrounds | ✅ | ✅ (no backgrounds) | | `t_cam_spec` |
| HDR dome map per episode | ❌ | ✅ | | `Randomizer.environment` is tier 2 only |
| per-world `geom_size` (MuJoCo Warp batched size, scale randomization) | ✅ | ✅ | ✅ | **Fixed (§3c).** Every primitive is a unit mesh scaled per world in the GPU path (capsules: radius and length exact, caps scaled). Silhouette IoU vs MuJoCo at the world's size 0.36–0.82 → ≥ 0.989. Meshes: MuJoCo Warp has no per-world mesh scale (physics neither). |
| per-world `geom_rgba` / `mat_rgba` / `geom_matid` written into the MuJoCo Warp model | ✅ | ✅ | | **Fixed (§3c)** (`render/model_sync.py`): read every frame, MuJoCo's colour rule (geom rgba over the material's unless default). Mean RGB error vs the world's compiled model 17–29 → 0.0 (T0), ≤ 5 (T2 noise). `geom_matid` switches the colour, not the texture. |
| per-world light fields of the model (`light_diffuse` / `specular` / `ambient` / `active` / `type` / `castshadow` / `cutoff` / `exponent` / `attenuation`) | ✅ | ✅ | | **Fixed (§3c).** Per-env light records from the model each frame; an inactive light emits nothing (also `active="false"` lights, which the static table drew lit: code). Tier 2's `usd_lights` keep their USD intensity. |
| per-world `cam_fovy` / intrinsics in the model | ❌ (use `t_cam_spec`) | ❌ (`set_fovy`) | | |
| roughness / metallic randomization | ◐ (shared by all envs) | ◐ | | `Randomizer.materials` writes one table through the renderer's `set_materials`. **Fixed (§3c):** the call crashed on a legacy-mode `Tier2Renderer` (`AttributeError`). |
| texture swap randomization (`rep.randomizer.texture`) | ❌ | ❌ | | Already listed as not done in GAPS (Replicator row) |

### 1c. Deformables, particles and other physics engines

| simulated thing | where it is simulated | renderer status | evidence |
|---|---|---|---|
| MuJoCo Warp flex cloth / cable / soft volume | `physics/deformable.py`, `tests/test_deformable.py` | ✅ every renderer and sensor (workstream A) | GPU path from `DeformableSim` / `BatchSim` buffers: 4 randomized worlds after 40 steps, every env ≥ 0.99 rgb / depth / seg against MuJoCo's renderer at the same vertices (`test_gpu_path_mujoco_warp_flex`). |
| MetalSim XPBD cloth / rope (the default cloth and rope engine per PARITY) | `physics/deformable.py` | ✅ (workstream A) | `render(xpbd, sources={"flexvert_xpos": xpbd.x})`: drawn as MuJoCo draws the model's flexes; 3 worlds ≥ 0.99 (`test_gpu_path_xpbd_and_physx_cloth`) |
| PhysX-port particle cloth | `physics/physx_cloth.py`, `tests/test_physx_cloth.py` | ✅ (workstream A) | `extras.DeformableMesh(name, tris, nvert, radius=...)` with `sources={name: sim.x}`; 3 worlds ≥ 0.99 in T0 / T1 / T2 / RT. Isaac draws PhysX cloth as its zero-thickness mesh (`radius=0`). |
| Newton models (G1 XPBD, VBD cloth / soft / rods, coupled proxy) | `physics/newton_backend.py` | ✅ rigid shapes in T0 / T1 / T2 / RT / Ann through `render/newton_bridge.py` (2026-10-12, [NEWTON_RENDER.md](NEWTON_RENDER.md)); cloth / soft-body surfaces and free particles ✅ once the dynamic-mesh and particle paths land (verified on merges of their WIP branches) | Measured: per-body silhouette coverage 1.0 (Menagerie Panda 0.986: z-fighting between coincident links, the same with the MJCF twin), depth p99 ≤ 0.97 mm (T0) and ≤ 0.1 mm (RT) against the same MJCF through the MuJoCo path; batched and spaced worlds. Before: no Newton `State` → renderer path. |
| Newton MPM particles (sand, fluids, Franka pour) | `physics/mpm/` | ❌ now; **in progress elsewhere** (`metalsim-particle-render`) | Code |
| Replicator motion vectors / 3D boxes / point cloud of deformables | `replicator/isaac.py` | ◐ | RGB / depth / segmentation / instance ids ✅ (extras are unlabelled instances with their own `instance_id`); motion vectors and 3D boxes stay geom-pose based (a deformable gets no motion vector and no 3D box) |

### 1d. USD import (`scene/usd_to_mjcf.py`) visual coverage

| USD content | status | note |
|---|---|---|
| Cube / Sphere / Cylinder / Capsule / Plane / Mesh gprims | ✅ | |
| UsdPreviewSurface: diffuse colour / texture, roughness, metallic, ior, opacity | ◐ | Opacity is stored and then drawn opaque. Roughness, normal and occlusion textures are dropped. |
| OmniPBR: diffuse constant / texture / tint, roughness, metallic, opacity, specular level, emission intensity | ◐ | `texture_scale` / rotate / translate are ignored. Emissive colour is reduced to intensity. Normal and ORM maps are dropped. OmniGlass is opaque. |
| MaterialX | ◐ | As PARITY §2 (coat, sheen, transmission and normal maps not mapped) |
| mesh UVs | ✅ (importer) | **Fixed (§3c):** `st` / `st0` / `UVMap` / `uv` or the texture reader's varname; faceVarying (indexed or not), vertex / varying, uniform; quads and n-gons fan-triangulated with the positions; USD v-up mapped to MuJoCo's convention. Texture match on the probe plates (MuJoCo renderer) 0.24–0.37 → 1.00. With workstream B's unflipped atlas, tiers 0 / 2 show them as MuJoCo does. |
| authored mesh normals | ❌ | Ignored; recomputed smooth (see the mesh-normals row) |
| GeomSubset per-face materials | ✅ | **Fixed (§3c):** one visual geom per `materialBind` subset (`<geom>_<subset>`, plus `<geom>_rest`), the whole mesh kept as an undrawn collider. Right-colour fraction top / sides 0.0 / 0.0 → 0.99 / 1.00. |
| `displayColor` per vertex / face | ◐ | First value only |
| `visibility = invisible`, purpose `proxy` | ◐ | Not honoured (purpose `guide` → hidden group 3) |
| DistantLight, SphereLight | ◐ | Sphere → point light (radius, shaping / cone and exposure dropped; diffuse clipped to 10) |
| DomeLight (colour, texture) | ◐ | Tier 2 only (environment map) |
| RectLight, DiskLight, CylinderLight | ◐ | **Fixed (§3c), far-field:** Rect / Disk → spot along -Z (cutoff 90° or `shaping:cone:angle`, cosine falloff, I = L·A), Cylinder → point light (I = L·2rℓ broadside); `normalize`, exposure honoured; shape / size / radiance recorded in custom text `usd_area_lights`. Before: a stage with one failed to load. Irradiance vs exact: +20–25 % at 1 light size, +2–3 % at 3, < 0.5 % at 10; soft area-light shadows not drawn. |
| Camera: focal length, vertical aperture | ◐ | Horizontal aperture (aspect), clipping range, orthographic projection and fisheye / f-theta are ignored |
| Points, BasisCurves, PointInstancer, TetMesh / deformable prims | ❌ | Dropped |

### 1e. Camera model (Isaac Lab `TiledCamera` / `Camera` parity)

| feature | status | note |
|---|---|---|
| pinhole intrinsics, per-env pose / intrinsics | ✅ | |
| several cameras per env in one call | ◐ | One camera per renderer instance (per-env `cam_id`); use one renderer per camera |
| depth convention | ✅ | z-depth. Background is 0 in the raw buffer and inf through `replicator/isaac.py`, as Isaac. |
| fisheye / f-theta / lens distortion | ✅ T2, RT; ◐ T0 | **Added (§3c)** `render/lens.py`: Isaac Lab `FisheyeCameraCfg`'s default f-theta polynomial (`Lens.ftheta()` / `Lens.from_isaac`) and Kannala-Brandt / OpenCV fisheye. T2 and the ray-traced depth camera trace the lens per pixel; T0 resamples cube faces (`render/fisheye.FisheyeTier0`, 5 faces for 200°). Rolling shutter not done. |
| far plane | ✅ | `Tier0Renderer(far=10.0)` (Isaac Lab's default `clipping_range` far is 1e6; MuJoCo's is `zfar` x extent), reversed-z depth; T2 `clip_far` |

## 2. What Isaac does for each gap, effort and priority

Priority is for camera-based learning:
- **P0**: visibly wrong or missing on common assets, or simulated state that the policy needs.
- **P1**: wrong on a class of tasks.
- **P2**: niche, or fidelity only.

Effort is engineering days for one person including tests, assuming the existing tier 0/2 structure.

| # | gap | Isaac (RTX / Kit) | fix | effort | priority |
|---|---|---|---|---|---|
| 1 | **Deformables invisible** (flex cloth / soft / cable, XPBD cloth and rope, PhysX-port cloth). **Done 2026-10-12** (§5) | Deformable bodies and cloth render as meshes skinned to simulation nodes (RTX); they appear in RGB, depth and segmentation | Add a dynamic-mesh slot class. Per frame, a compute kernel writes vertices and normals from `flexvert_xpos` (or any (N, V, 3) tensor) into a per-env vertex buffer. The raster draws it non-instanced; T1/T2/RT refit a per-env BLAS (`MTLAccelerationStructure` refit). Flex 2D gives triangles directly, 3D gives the surface faces (`flex_shell`), 1D gives capsules or tubes. Gets one seg id per flex. | 4–6 (raster 2, BLAS refit 2, tests 1–2) | **P0**: deformable manipulation from pixels is impossible today |
| 2 | **Newton models not renderable** (rigid and deformable). **Done 2026-10-12** ([NEWTON_RENDER.md](NEWTON_RENDER.md)) | Isaac Lab 3.0 renders Newton scenes through RTX (`body_q` / `particle_q` written to Fabric) or Newton's Warp `SensorTiledCamera` (own world + world -1 per camera) | `newton_bridge.NewtonRenderSource`: a render-twin `MjModel` (from the Newton shapes, or the source MJCF) plus graph-captured Warp kernels that write its pose buffers from `State.body_q` per world. Cloth / soft surfaces → gap 1's dynamic-mesh path; free particles → the particle renderer | done (rigid); deformables via #1 and the particle path | **P0** for the Newton path; P1 overall |
| 3 | **Mesh normals smoothed across hard edges** | Uses authored normals, or face normals split at sharp edges | Use `mesh_normal` (MuJoCo's own per-vertex normals with its smoothing rule) or split vertices at a crease angle in `_tris_to_indexed`; read USD `normals` in the importer | 0.5–1 | **P0**: every CAD robot and object is shaded wrong (spread 43.7 vs 2.9), which is a sim-to-real appearance gap on every frame |
| 4 | **Infinite plane drawn as 10 × 10 m**. **Done 2026-10-12 (§3c)** | `GroundPlaneCfg`: 100 m (default) plane; the horizon is visible | Size planes to the camera far plane (or a large default) in the raster. In the RT, an infinite plane conflicts with the per-env spatial offset (`stride`), so use a separate per-env analytic plane test in the shaders, or clip to the stride. | 1–2 | **P0** for locomotion / navigation cameras, P2 for tabletop |
| 5 | **T0/T1 10 m far plane**. **Done 2026-10-12 (§3c)**: `far=`, reversed z, tier-1 ray limit | Camera `clipping_range` default (0.01, 1e6)/(0.1, 1e5) | **WIP in the main checkout** (`far=`). Still to do: T1's 50 m and T2's 1000 m ray ranges, the splat composite far plane, and depth precision (TEXTURES.md "Far plane") | 0.5 (land the WIP) | **P0** for outdoor / nav; P2 tabletop |
| 6 | **Transparency / opacity** | RTX: OmniGlass refraction, opacity cutout, alpha blending; depth and segmentation of transparent objects configurable | T0: sorted alpha-blend pass for alpha < 1 (order-independent transparency or per-object sort), seg / depth from opaque only or configurable. T2: `any_hit` alpha test + a dielectric BTDF. Importer: keep opacity. | 2 (T0 blend + cutout), +2 (T2 glass) | **P1**: glassware, bottles and windows are common in manipulation datasets |
| 7 | **Sites and tendons invisible**. **Done 2026-10-12** (§5; sites on by default as in MuJoCo, off in the ray-traced sensors) | n/a for sites (no USD equivalent); tendons become visual cables/meshes authored by the asset | Optional: draw sites (opt-in group flag) as primitives. Draw spatial tendons as per-frame capsule chains from `ten_wrapadr` / `wrap_xpos` (dynamic instances, gap 1 infrastructure). | 1 (sites), 2 (tendons) | P1 for tendon-driven hands and cable tasks; P2 for sites (MuJoCo-visual parity only) |
| 8 | **Skins invisible**. **Done 2026-10-12** (§5) | Skinned meshes (UsdSkel) render | Linear blend skinning kernel into the dynamic-mesh path (#1) | 1 on top of #1 | P2 (rare in robot models; composite cables and older flexcomps) |
| 9 | **SDF geoms invisible** | n/a (USD has meshes; SDF is a collision representation) | Draw the SDF's mesh asset (`geom_dataid` → mesh, as MuJoCo does) | 0.5 | P1: anything using SDF plugins (gears, nuts and bolts) is invisible |
| 10 | **Spot / point-light shadows in T0/T1** | All lights cast shadows in RTX | T1: ray-traced shadow for every light (the loop already exists, gate on `castshadow`). T0: cube or spot shadow maps per light, or route shadowed lights to T1. | 0.5 (T1), 3 (T0 maps) | P1: indoor / spot-lit scenes |
| 11 | **Texture mapping errors** (vertical flip on mesh UVs, no UVs on cylinder / capsule, sphere phase, cube textures) | Asset UVs used as authored; MDL projections | Flip V in the atlas path (with a regression image), MuJoCo-style generated coordinates for primitives, cube-map sampling (`texturecube`) for `type="cube"` | 1.5 | P1: textured objects and DR textures look wrong; mirrored decals and labels |
| 12 | **Per-world `geom_size` (scale DR) ignored**. **Done 2026-10-12 (§3c)** | Isaac scale randomization changes the rendered prim | Generalize `SizedGeoms` (already done for terrain slots) to every primitive, plus a per-world scale for meshes | 1 | P1: size randomization is invisible to the policy |
| 13 | **Per-world model colour / light fields ignored**. **Done 2026-10-12 (§3c)** | Replicator writes USD attributes the renderer reads | Read MuJoCo Warp's batched `geom_rgba` / `mat_rgba` / `light_*` if the event terms write them, or document `t_colors` as the only path (it already exists) | 0.5 | P2 (a workaround exists) |
| 14 | Non-RGB texture roles, normal maps, USD GeomSubsets, UV scale / rotate / translate | RTX uses all of them | Material table extension + importer | 3–5 | P2 (fidelity) |
| 15 | MJCF skybox as a mean colour | Dome texture background | Convert skybox to an equirect / cube environment (T2 has the sampler; T0 needs a background pass from it) | 1 | P2 (backgrounds are usually randomized anyway) |
| 16 | Area lights, more than 8 lights in T0/T1 | Unlimited lights, area lights with soft shadows | Raise `MAX_LIGHTS` (loop cost), area-light sampling in T2 | 0.5 / 2 | P2 |
| 17 | hfield sides / base. **Done 2026-10-12 (§3c)** | Terrain meshes are closed | Add skirt triangles | 0.25 | P2 |
| 18 | Fisheye / lens distortion. **Done 2026-10-12 (§3c)**: f-theta and Kannala-Brandt in T2 / RT, T0 cube faces | `FisheyeCameraCfg`, f-theta | Per-pixel ray directions in T2 and RT (trivial); T0 needs a distortion warp pass | 1–2 | P2 (P1 for wide-FOV nav cameras) |
| 19 | MPM particles | Points / mesh-from-particles (fluid isosurface) | **in progress (other agent)** | — | P0 for the MPM tasks |
| 20 | Splat composite: no segmentation for the splat background, baked lighting (no robot shadows), 10 m composite limit | Isaac Sim 5.x renders 3DGUT Gaussian scenes inside RTX | **WIP elsewhere**; seg = "background" class by design | — | P2 |

## 3. Ranked fix list

| rank | fix | effort (days) | why first |
|---|---|---|---|
| 1 | Dynamic-mesh path for flex / XPBD / PhysX-port cloth (#1), with skins (#8) and tendons (#7) on the same infrastructure: **done 2026-10-12** (§5) | 4–6 (+1 skins, +2 tendons) estimated; done in about 4.5 hours | A whole simulated category, which PARITY claims at parity with Isaac's deformables, is invisible to cameras |
| 2 | Mesh normals (#3): **done 2026-10 (§3b)** | 0.5–1 | Wrong on every frame of every mesh robot; cheapest high-impact fix |
| 3 | Land the far-plane WIP (#5) and make the infinite plane infinite (#4) | 1.5–2.5 | Locomotion and navigation cameras see a 10 m world with a floor edge |
| 4 | Newton → renderer adapter (#2): **done 2026-10-12** (`render/newton_bridge.py`) | 2–3 | Without it the Newton backend has no camera at all |
| 5 | SDF geoms (#9): **done 2026-10 (§3b)** | 0.5 | Silently invisible objects |
| 6 | Texture mapping fixes (#11): **done 2026-10 (§3b)** | 1.5 | Mirrored and missing textures on common primitives |
| 7 | Per-world size DR (#12) | 1 | Randomized sizes invisible to the policy |
| 8 | Transparency (#6): **done 2026-10 (§3b)** | 2–4 | Glass / plastic objects |
| 9 | Spot / point shadows in T1 / T0 (#10): **done 2026-10 (§3b)** | 0.5–3 | Indoor lighting cues |
| 10 | The P2 items (#13–#18) | ~10 total | Fidelity |

MPM particle rendering (#19) is not ranked: it is being built by another agent.

## 3b. Static-appearance fixes (2026-10, workstream B)

This workstream fixed gaps #3, #6, #9, #10, #11, #15 and #16 (lights), measured against `mujoco.Renderer`.

- **Reference.** MuJoCo 3.14.0's own renderer source:
  - `render_context.c`: `mjr_uploadMesh`, `makePlane`, `sphere`, `halfSphere`, `cylinder`, `disk`, the box list, `mjr_uploadTexture`.
  - `render_gl3.c`: `settexture`, `mjr_render`'s transparent and skybox passes, `isSupportedLight`.
  - `engine_vis_visualize.c`: `setMaterial`, `mjv_initGeom`.
- **Isaac / RTX behaviour** comes from NVIDIA's RTX renderer and Replicator docs, forum answers, Isaac Lab issues #2956 and #3847, and the UsdLux / UsdGeom / UsdPreviewSurface specs.
- **Measurements:**
  - Before/after numbers: `scripts/diagnostics/render_appearance_parity.py`, written to `docs/measurements/render_appearance.json`.
  - Tests: `tests/test_render_appearance.py`.
  - Cost: `scripts/diagnostics/render_appearance_cost.py`.
- **Legacy path.** `appearance="legacy"` (on `Tier0Renderer`, `Tier2Renderer`, `RayTracer` and `build_scene_tables`) rebuilds the old tables for comparison. The old behaviour of each feature has its own switch: `transparency=False`, `shadow_lights="caster"`, `max_lights=8`, `skybox=False`.

| item | what changed | MuJoCo | Isaac / RTX | where MetalSim follows which |
|---|---|---|---|---|
| mesh normals | Uses MuJoCo's compiled normals and its 0.8 dot rule, which keeps hard edges (`builtin_geoms.mesh_triangles`). Decimated LODs use `MakeNormal` plus the same rule. | the rule | Authored `normals` on a polygonal (`subdivisionScheme = none`) mesh; Isaac Lab's spawners author `bilinear` and no normals. A mesh with the catmullClark default ignores its normals. | MuJoCo |
| SDF geoms | Drawn from `geom_dataid`'s mesh in every renderer and in the RT structures | the same | n/a (an SDF is a collision representation) | MuJoCo |
| geom colour | rgba resolved as `setMaterial`: the geom's own rgba wins unless it is the default (0.5, 0.5, 0.5, 1). Geoms whose resolved alpha is 0 are skipped. Previously the material always won. | the same | n/a | MuJoCo |
| texture coordinates | Primitives use MuJoCo's tessellation (`vis.quality`) and coordinates. Meshes without UVs and heightfields use the object-linear mapping. `texrepeat` / `texuniform` follow `settexture`. Atlas rows are unflipped. sRGB 2D textures are decoded. The atlas pads with wrapped texels instead of black. | the same | Authored `st` with (0, 0) at the image's lower left. The importer now writes 1 − t, the same as MuJoCo's OBJ loader. | MuJoCo; one convention for any later texture backend: t = 0 is the first `tex_data` row |
| cube textures | OpenGL cube-map lookup (one face, or a 6-face strip) of MuJoCo's object coordinates | the same | n/a (MDL projections) | MuJoCo |
| skybox | Cube map of the view direction: T0 background, T1 reflection misses, T2 environment radiance | Unlit background; also haze over infinite planes, which is not ported | The dome light is both background and light | T0 / T1 MuJoCo; T2 lights the scene with it, as a dome |
| transparency | T0/T1: per-env object sort (`sort_transparent`), back faces culled, blended front-to-back then back-to-front. The draws are rank-major instanced and pull vertices, so two draws cover every env. T2: presence sampling, and shadow rays carry ∏(1 − α). | Blended as described. Depth and segmentation come from the glass surface; on this probe `mujoco.Renderer`'s depth is 1.904 m, at the glass. Shadows are cast in full. | Real-Time (Isaac Sim ≤ 5): opacity below 1 is cut out, so the object is gone from RGB, depth and segmentation while its shadow stays (fractional cutout is off by default). Path tracer and Real-Time 2.0: stochastic presence. Glass: depth on the glass. | Colour: T0/T1 MuJoCo, T2 Isaac's path tracer. Annotations: MuJoCo by default; `transparent_ids="behind"` gives Isaac Real-Time's opacity behaviour. |
| lights | Every light is lit (`MAX_LIGHTS` compile-time cap, default 2^20) | 8 in practice (fixed-function GL); directional and spot only | No documented limit; sampled direct lighting for thousands of lights | Isaac. `max_lights=8` gives MuJoCo's cap. |
| shadows | Every `castshadow` light. T0 renders extra depth layers in the caster's per-env tile layout: orthographic for directional, the cone for spot, 6 faces for point; spot and point layers store linear distance. T1 traces one ray per light. | Directional and spot lights, shadow maps, spot cone × `shadowscale` | Every light casts shadows by default (`shadow:enable` falls back to true) | Isaac for point lights; both for spot and directional |

**Limits:**
- Heightfields drawn by the culled tile path (`hfield_cull`, the default) cast only the caster's shadow in T0. T1 and T2 trace them.
- Lidar and the ray-traced depth camera still treat transparent geoms as opaque.
- A capsule's cube-map coordinates use one scale for the whole capsule; MuJoCo's hemispheres differ slightly.
- Planes seen from behind are not made more transparent, as MuJoCo does.

**Cost** (`render_appearance_cost.py`; M4 Max; 128×128; best of 3×10 frames): **preliminary, measured as a contended `check` job.** The timing queue was deadlocked: waiters running an older `gpu_lock.py` were waiting behind newer ones. Up to 3 jobs shared the GPU. Before and after were rendered back to back, so their ratio is indicative.

| scene | renderer, envs | before ms (env-frames/s) | after ms (env-frames/s) | why |
|---|---|---|---|---|
| SO-101 box task, full-resolution meshes | T0, 1024 | 152 (6,737) | 297 (3,450) | Vertices 162 K → 619 K: hard edges split the corners, the same count MuJoCo uploads. Indices are unchanged (975 K per env), so the raster is vertex-bound. |
| SO-101 | T1, 256 | 62 (4,103) | 80 (3,200) | same |
| SO-101 | T2, 64 (4 spp, 2 bounces) | 104 (615) | 95 (673) | No measurable change (ray tracing does not depend on the vertex count). |
| Franka Panda (Menagerie) | T0, 1024 | 68 (15,059) | 193 (5,300) | The shadow-casting spot light gets its own shadow pass; vertices 69 K → 396 K |

Not measured yet:
- SO-101 decimated to 3000 faces per mesh. This is the lift task's training setting, `LiftConfig.decimate_faces`; its vertex counts are far smaller.
- G1 rough-terrain head camera. It has no meshes with hard edges in its collision set and one directional light, so no extra shadow pass.
- Exclusive timings.

The second run stalled on Panda tier 1 and the watchdog killed it at 1500 s. That was during the window when this Mac's GPU hung (02:34; the coordinator attributes the hang to another agent's kernel). The stall is not explained yet.

**Speed levers:**
- `decimate_faces` (LOD normals use the same MuJoCo rule).
- `shadow_lights="caster"`.
- `appearance="legacy"`, the old tables and old cost.


## 3c. World extent, per-environment fields, camera models (2026-10-12, workstream C)

Probes: `scripts/diagnostics/render_world_audit.py` (synthetic MJCF; mujoco.Renderer as the visibility reference, the
same renderer on a model compiled with a world's values as the per-world reference), run on the code before and after
(`docs/measurements/render_world_audit.json`). Tests: `tests/test_render_world.py`, `tests/test_render_newton_perworld.py`.
Coverage = fraction of MuJoCo's object pixels with RGB / depth / segmentation.

| item | before | after | notes |
|---|---|---|---|
| far plane: boxes at 15 m / 40 m, T0 / T1 | 0.0 / 0.0 (fixed 10 m) | 1.0 / 1.0 with `far=` MuJoCo's far (depth rel. error 1e-6); default `far=10` unchanged (0.0) | T2 / RT: 1.0 before and after |
| depth precision, `far=1e6` (Isaac's default), median abs. error at 200 m / 2 km | standard z: 0.033 m / 21.4 m | reversed z: 1.5e-5 m / 0.0 m | the previous convention stays as `reverse_z=False` |
| infinite ground (far-plane scene): ground coverage T0 / T1 / T2 / RT | 0.369 (10 × 10 m square) | 1.0 / 1.0 / 1.0 / 0.978 (RT probe range 100 m) | T0 at the default 10 m: 0.691 |
| locomotion view, Menagerie checker floor: coverage, texture corr. vs MuJoCo within 8 m | 0.645; corr 0.04 (T0) / 0.03 (T2) | 1.0; corr 0.42 (T0) / 0.85 (T2) | MuJoCo's plane mapping (u = x / 2, -0.5 offset, texuniform ignored on infinite axes). T0's remaining gap is aliasing of the thin checker marks (no mips: TEXTURES.md, workstream B) |
| plane texture orientation (gradient, top view) corr. T0 / T2 | 0.30 / -0.003 | 0.992 / 0.999 | |
| lidar returns from the ground at 15 m | 0.0 | 1.0 | |
| hfield side view / from below, every renderer | 0.58 / 0.90 | 1.0 / 1.0 | |
| per-world `geom_size` (4 worlds, 6 primitives): silhouette IoU vs MuJoCo | 0.36–0.82 | 0.995–1.0 | T0 and T2, GPU path from BatchSim (on workstream B's MuJoCo tessellation) |
| per-world `geom_rgba` / `mat_rgba` / `light_diffuse` / `light_active`: mean abs. RGB error vs compiled | 17–30 | 0.0 (T0), ≤ 5 (T2, Monte Carlo) | |
| `Randomizer.materials` on a legacy `Tier2Renderer` | AttributeError | ok (T0, T2 legacy, T2 OmniPBR) | |
| USD import: Rect / Disk / Cylinder lights on a probe stage | stage failed to load | 4 of 4 lights; irradiance +2–3 % at 3 light sizes, < 0.5 % at 10 | `scripts/diagnostics/usd_import_visuals.py`, `tests/test_scene_usd.py` |
| USD import: GeomSubset materials (top / sides right-colour fraction) | 0.0 / 0.0 | 0.99 / 1.00 | MuJoCo renderer |
| USD import: vertex / faceVarying / indexed UVs on quads; faceVarying triangles (texture match) | 0.367 (untextured); 0.239 (upside down) | 1.00 | MuJoCo renderer; workstream B unflipped the atlas, so tiers 0 / 2 agree with MuJoCo |
| camera projection at the image edges and corners (`tests/test_camera_projection.py`: small spheres / 2 cm capsules on pixel centres, fovy camera, physical camera with an offset principal point, the audit's tilted camera) | physical cameras with a principal offset mirrored (sign of `cx` / `cy`; 24 px at the probe's 0.3 mm offset) | every object pixel-exact vs mujoco.Renderer in T0 / T1 / T2 / RT (IoU 1.0, centroid offset 0.0 px), top rows included | no half-pixel or fovy convention difference was found; MuJoCo's frustum (`getFrustum`, `glFrustum`) re-derived in the test |
| fisheye (Isaac default f-theta 200°, Kannala-Brandt): sphere centroid vs the lens model at 0 / 60 / 95° | not supported | ≤ 0.27 px (T2, RT, T0 cube faces) | z-depth negative beyond 90°, as the optical-axis definition gives |
| Newton bridge: per-world `shape_scale` / `shape_color` | read once at build | per frame; IoU > 0.95 (capsule > 0.85) vs the MJCF at that world's values, colour error < 4 (T0) | mesh texture / UVs / roughness / metallic kept in the twin (texture corr. 1.0 vs the MJCF path) |

**Speed** (`scripts/diagnostics/render_world_cost.py`, exclusive `timing` job, device idle at start and end, M4 Max,
2026-10-12 11:14; ms per frame, median of 30 GPU-path renders; base = 471d156, after = this change):

| scene | base | after (defaults) | after, far = 100 m | each fix off in turn (skirts / sizes / model sync / reversed z) |
|---|---|---|---|---|
| G1 flat, tier 0, 1024 envs, 128 × 128 head camera | 1.90 | 1.93 | 1.95 | 1.92 / 1.94 / 1.91 / 1.91 |
| G1 rough (heightfield + terrain slots), same camera | 34.95 | 34.69 | 98.5 (the camera sees 100 m of terrain) | 34.54 / 34.60 / 34.67 / 34.58 |
| primitives, tier 0, 4096 envs, 64 × 64 | 7.88 | 7.86 | 7.81 | 7.86 / 7.90 / 7.80 / 7.80 |
| primitives, tier 2 (1 spp, 2 bounces), 1024 envs, 64 × 64 | 8.82 | 9.09 | – | – / 9.19 / 9.03 / – |

The defaults cost nothing measurable in tier 0 (within ±2 %) and +3 % in tier 2 (the per-frame model sync and the
per-env light records). The first version drew the heightfield sides whole in every env and cost +22 % on G1 rough
(34.4 → 42.0 ms); they are now culled per env in 32-triangle chunks like the top surface's tiles. An earlier run
measured while another process held the GPU at 100 % gave the same relative changes at 30–45 % higher absolute
times, so that utilisation reading was real contention, not the known false reading.

Design notes:
- **Far plane and depth.** `Tier0Renderer(far=10.0, near=0.01, reverse_z=True)`. Reversed z (near → 1, far → 0, compare
  greater, clear 0) keeps float32 depth error ~1e-7 x distance, so a far plane of Isaac Lab's default 1e6 costs nothing
  in precision. The shadow map keeps the standard convention. Tier 1's ray-traced effects are skipped on fragments
  beyond its ray-tracing region (`RayTracer.ray_limit`), so a large far does not let rays reach other envs.
- **Infinite planes.** A disc of octagon rings (apothems doubling from 0.25 m; ~170 triangles for 1e6 m) keeps
  triangles small near the viewer. T0/T1 re-centre it under each env's camera in the vertex stage with apothem
  2 x far x (corner ray length); the ray tracer builds it with apothem extent + range (+ the plane's offset) and grows
  the env stride ~4 % to keep neighbouring discs apart. Isaac's `GroundPlaneCfg` is a finite 100 x 100 m visual
  (`TerrainImporter(terrain_type="plane")`: 2e6 m), so an infinite MuJoCo plane is the closer match to both.
- **Per-world fields.** `render/model_sync.py` writes per-env light records and per-instance colours each frame (one
  dispatch, inside the render command buffer); the material table the shaders read carries rgba = 1. A sim can supply
  the fields as `render_fields` (the Newton bridge does) instead of `m`. Sizes: `render/sized.py` folds the per-world
  scale into a renderer-owned `geom_xmat`; normals use its inverse transpose in every shader.
- **Fisheye.** Isaac Lab exposes `fisheyePolynomial` (default, theta = A + B r + ... + F r^5 in nominal pixels,
  `fisheye_max_fov` 200°), `fisheyeKannalaBrandtK3`, `fisheyeRadTanThinPrism`, `fisheyeSpherical`; only the first
  carries coefficients (the others write the token only), and no shipped Isaac Lab task uses a fisheye camera. Isaac
  Lab develop adds `OpenCvFisheyeDistortionCfg` (Kannala-Brandt k1..k4), mapped by `Lens.from_isaac("opencvFisheye")`.
  Unconfirmed in NVIDIA's docs: how RTX scales the nominal resolution to the render resolution (implemented: per axis)
  and what fills pixels beyond `max_fov` (implemented: black, depth 0).

## 4. Reproduce

```
scripts/gpu_run.sh render_coverage check 10 -- .venv/bin/python scripts/diagnostics/render_coverage_audit.py --out runs/render_coverage
```

- Writes `results.json` and contact sheets (`geoms.png`, `nongeom.png`, `far.png`, `textures.png`; MuJoCo | T0 | T2)
  to `--out`.
- All probe scenes are synthetic MJCF built inside the script.
- No splat scene or other external asset is used.

## 5. Deformables, cables, tendons, skins and sites (workstream A, 2026-10-12)

**What is drawn.** Every renderer and sensor now draws the non-geom objects MuJoCo's own renderer draws, under an
`mujoco.MjvOption` (`visual=`; default MuJoCo's default option, `visual=False` geoms only), with MuJoCo's geometry
(`mjv_updateActiveFlex`, `mjv_updateActiveSkin`, `addSpatialTendonGeoms`, `render_gl3.c`):

| object | geometry (as MuJoCo) | T0 | T1 | T2 | RT | Ann |
|---|---|---|---|---|---|---|
| site (visible groups) | its primitive, posed by `site_xpos` / `site_xmat` | 0.993–1.0 | 0.993–1.0 | 0.993–1.0 | off by default (as `mj_ray`) | ✅ |
| flex 2D (cloth) | smoothed surface ±radius, sides closed (`flatskin`: face normals) | 1.0 | 1.0 | 1.0 | 1.0 | ✅ |
| flex 3D (soft volume) | shell offset +radius | 1.0 | 1.0 | 1.0 | 1.0 | ✅ |
| flex 1D (cable); any flex with `mjVIS_FLEXEDGE` and no skin | open cylinder per edge; spheres with `mjVIS_FLEXVERT`; `mjVIS_FLEXFACE`: flat offset faces | 1.0 | 1.0 | 1.0 | 1.0 | ✅ |
| skin | linear blend skinning, area-weighted normals, `inflate` | 1.0 | 1.0 | 1.0 | 1.0 | ✅ |
| spatial tendon | capsule per wrap segment, half width inside a wrapping geom, none across a pulley | 1.0 (straight); 0.95–1.0 wrapping / pulley | same | 1.0 | 1.0 | ✅ |
| user `DeformableMesh` (XPBD, PhysX-port cloth, Newton cloth via the bridge) | the surface (radius 0) or a 2D flex skin | ≥ 0.99 | ≥ 0.99 | ≥ 0.99 | ≥ 0.99 | ✅ |

Coverage is rgb / depth / segmentation of MuJoCo's silhouette (all three equal unless noted), depth error ≤ 0.4 mm,
160×120. Sources: `scripts/diagnostics/render_coverage_audit.py` (non-geom probe) and `tests/test_render_extras.py`
(deformed states: MuJoCo C stepped, MuJoCo Warp `DeformableSim` / `BatchSim` on the GPU path, `XPBDSim`,
`PhysXClothSim`; three envs in different states per render). The geometry itself is checked on the CPU against
`mjv_updateScene` (flex faces and normals, skin vertices and normals, tendon capsules, edge cylinders) to 2e-5 m, and
the Metal kernels against a NumPy reference of them.

The 0.95 for wrapping / pulley tendons is the rasterizer, not the geometry: those tendons are 1.5–2 px wide, and a
pixel-centre test of the same triangles on the CPU covers all of MuJoCo's pixels, but 6 of the pulley's 120 lie less
than 1/256 px inside any triangle (edge-on slivers at the silhouette), below the raster's sub-pixel precision. The
ray-traced paths cover them. Also seen: in the top ~5 rows of an image MetalSim and MuJoCo disagree by about one row for
every object, geoms included (a 2 cm capsule geom in rows 1–4: T0 0.54, T2 0.77; mid-image 1.0), a camera-model
difference outside this work (the first audit's cable sat there).

**How.** `metalsim/render/extras.py` adds the objects to the scene tables after the geoms (slots `G0 …`; segmentation
id `ngeom + 1 + j` in geom mode). Sites are ordinary instanced slots whose pose rows follow the geom rows (a gather
kernel appends `site_xpos` / `site_xmat`). Flexes, skins, tendons and user meshes are dynamic slots:
`metalsim/render/dynamic_mesh.py` generates their vertex records (position, normal, uv) for every env in five compute
dispatches per frame (copy the simulator's vertices, skinning, vertex normals from host-built adjacency, surface
records, tube records from a template), all loops bounded by validated tables. The raster draws each object with one
instanced draw (instance = env); the ray tracers keep one primitive acceleration structure per object holding every
env's triangles (env-grid offset baked in), refit each frame and rebuilt every 32 frames, with one identity instance in
the instance structure after the rigid ones; the particle renderer's grain instances follow those
(`RayTracer.n_inst` / `prim_as` include them). No shader that existed before changed: the dynamic records use the static
vertex layout, so hits on them resolve through the same `inst_tab` / `slot_mesh` / `mesh_info` lookups.

**Unchanged when absent.** Scenes without visible sites / flexes / skins / tendons render byte-identically: 87 of 92
outputs identical on six scenes (prims, textured heightfield, SO-101, Panda, Shadow hand, Go2) against the previous
integration-branch commit, every renderer and the depth camera; the five differing outputs were the Go2 lidar hitting
the lidar's own `imu` site, since fixed by leaving sites out of the sensors (`scripts/diagnostics/render_identity_check.py`;
`test_scenes_without_extras_are_byte_identical`). Go2's camera images were identical (its site is inside the body).

**Isaac / RTX.** Isaac renders deformables (PhysX FEM / particle cloth, Newton cloth) as the simulated mesh itself:
zero thickness, two-sided, no ±radius skin (`DeformableMesh(radius=0)` reproduces that; MuJoCo's skin adds 2 × radius,
1 px or less at usual distances). Sites, MJCF tendons and MJCF skins have no rendered counterpart in Isaac: the MJCF
importer has site and tendon primitives (`AddTendons`, site prims) but they carry no render geometry, and skins are not
imported; drawing them is MuJoCo-viewer parity, switchable per group with `visual=`. Isaac's instance segmentation
labels a deformable by its prim; here it is an unlabelled instance with its own `instance_id`
(`/World/<kind>/<name>`).

**Not done.** MuJoCo's catenary for a slack two-site spring tendon and the limit-impedance recolouring of default-grey
tendons; motion vectors and 3D boxes of deformables (annotators stay geom-pose based); flex / skin textures use the
first texture coordinate of each vertex (seams with different coordinates per element are not split).
