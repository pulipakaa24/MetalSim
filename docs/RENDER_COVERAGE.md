# Render coverage: what the simulator simulates vs. what the renderers draw

Date 2026-10-12. Reference state: `origin/metalsim-mpm` (565ae2f). Machine: M4 Max. Sections 3b-3e and 5 add the
fixes through 2026-10-13; status marks below were brought to main `3670f4b8` on 2026-10-13.

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
  **Landed since (2026-10-13):** merged to main with PR #4 (tier 2 + tier 0 render grains, two-level grain structures and
  Metal's acceleration-structure limits), PR #9 (grain chunks of 2^24 - 1, 25 M-grain validation on the 14-inch) and
  PRs #14 / #15 (25 M-grain benchmark on the 16-inch, logs); see the
  row in §1c and `metalsim/render/PARTICLES.md`.

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
| spatial tendon with `width` (cables, wires, MuJoCo tendon-driven hands) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0, depth error 0 (was 0.0); tendons wrapping a cylinder, a sphere and across a pulley in §5. Capsules per wrap segment (MuJoCo's tessellation, half width inside a wrapping geom, none across a pulley). **Catenary and limit colour (§3d, 2026-10-13)**: a slack two-site tendon with a deadband spring or an unreached length limit hangs as MuJoCo's catenary (capsules equal to `mjv_updateScene`'s to 2e-5 m; coverage of MuJoCo's silhouette ≥ 0.97 in T0 / T1 / T2 / RT, `tendons="straight"` ≤ 0.6 on the hanging ones), and default-grey tendons at a limit take `vis.rgba.constraint` by the limit impedance (colour equal to MuJoCo's scene to 2e-6). Isaac draws no tendon geometry. |
| flex 2D (cloth) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0 of 277 px, depth error ≤ 0.1 mm (was 0.0). MuJoCo's skin: smoothed surface ±radius, sides closed. |
| flex 3D (soft volume) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0 of 152 px, depth error 0 (was 0.0). Shell offset by +radius (flat-shaded when `flatskin`). |
| flex 1D (cable) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: open cylinders per edge (`mjVIS_FLEXEDGE`). The first audit's cable sat just above the image (0 px in MuJoCo too); moved mid-image it is 1.0 / 1.0 / 1.0 in every renderer. |
| skin (bone-skinned mesh) | ✅ | ✅ | ✅ | ✅ | ✅ | **fixed (workstream A, §5)**: 1.0 / 1.0 / 1.0 of 236 px, depth error 0 (was 0.0). Linear blend skinning, area-weighted normals, `inflate`. |
| transparency (`rgba` alpha < 1, material opacity, USD `opacity`) | ✅ | ✅ | ✅ | ✅ | ✅ | **Fixed 2026-10 (B)**. T0/T1 blend as MuJoCo does: objects sorted per env, drawn front-to-back then back-to-front, back faces culled. T2 treats alpha as presence, like a USD opacity in Isaac's path tracer. Red box behind 25 % blue glass, colour change the glass makes over the box: MuJoCo (−64, 0, +65); T0/T1 before (−146, 0, +146), after (−64, 0, +64); T2 before (−122, 0, +125), after (−53, 0, +53). Annotations default to MuJoCo's: segmentation and depth of the glass (1.904 m, equal to MuJoCo). `transparent_ids="behind"` gives Isaac RTX Real-Time's opacity behaviour (seg / depth from what is behind). **RT: fixed 2026-10-13 (§3e)**: the ray-traced depth camera takes the same `transparent_ids` (default the surface: equal to `mj_ray` and MuJoCo's depth to < 0.2 mm) plus Isaac's `alpha_threshold`; the lidar transmits through transparent geoms with partial returns (default), or follows `mj_ray` (`transparency="surface"`); lidar materials (clear glass, mirror ghosts) and return modes. Scenes without transparency byte-identical. |
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
| skybox | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: the first skybox texture is drawn as MuJoCo's cube map of the view direction (d.x, d.z, −d.y). In T0 it is the background; in T1 it is also what reflections miss into. In T2 it is the radiance of every escaping ray, so it lights the scene; a `dome` or `set_environment` replaces it. Background luminance std 0.8 / 2.8 → 17.8 / 18.3 (MuJoCo 17.8), corr 0.996 / 0.988, MAD 0.1 / 0.2. `set_clear_color` turns it off. |
| haze over infinite planes (`mjRND_HAZE`, on by default) and fog (`mjRND_FOG`, off) | ✅ | ✅ | ✅ | n/a | | **Added (§3d, 2026-10-13)**: MuJoCo's faceted haze cone over the first infinite plane (with a skybox, camera above the plane) composited per pixel at the first opaque surface, and GL linear fog on eye-space depth (`fog=True`). Haze alpha vs mujoco.Renderer: ≤ 3 levels (p99 1) in T0 / T1 / T2. `haze=False` / `fog=False` switch them off; compiled out when unused. Depth and segmentation unchanged (MuJoCo's depth buffer holds the cone). |
| directional / point / spot lights moving with bodies | ✅ | ✅ | ✅ | n/a | | `light_xpos` / `light_xdir` per world |
| more than 8 lights | ✅ | ✅ | ✅ | n/a | | **Decided 2026-10 (B): every light, as Isaac / RTX** (no light limit). MuJoCo's fixed-function GL path stops at 8. Patches 9 and 10: T0 0.0 / 0.0 → 42.8 / 42.6 (T2 42.4 / 42.4). `Tier0Renderer(max_lights=8)` restores MuJoCo's cap. MuJoCo also lights only directional and spot lights; we also light point lights. |
| spot-light shadows | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**. T0 renders a perspective shadow map per shadow-casting spot light, over the full cone; MuJoCo covers cutoff × shadowscale. T1 traces a ray to the light. Plate shadow ratio: MuJoCo 0.00; T0 1.04 → 0.054; T1 1.04 → 0.052; T2 0.033. Spot + directional together: MuJoCo 0.20, T0 0.81 → 0.21. |
| point-light shadows | ✅ | ✅ | ✅ | n/a | | **Fixed 2026-10 (B)**: T0 renders six 90° cube faces per point light; T1 traces a ray. T0 1.02 → 0.053, T1 1.02 → 0.051, T2 0.035. MuJoCo does not light point lights at all; Isaac RTX shadows them. |
| directional shadows | ✅ (every caster) | ✅ (every caster) | ✅ | | | 0.052 / 0.049 / 0.041 vs MuJoCo 0.0 (unchanged). Every further shadow-casting directional light now gets its own map in T0 and its own soft ray in T1. With a DR light on, the first model directional light stays unshadowed, as before. |
| image lights (`mjLIGHT_IMAGE`) | ◐ | ◐ | ◐ | | | Mapped to directional (`build_lights`, code) |
| area lights (USD Rect / Disk / Cylinder, SphereLight radius) | ◐ (far-field) | ✅ | ✅ | | | **Fixed 2026-10-13 (§3e)**: T2 samples the emitter (soft shadows; floor irradiance within 0.1 % mean, 1.3 % per 4 x 4 block of the analytic value at 0.5-3 light sizes, was +19 % at one size); T1 samples it with a shadow ray per sample over the closed-form irradiance; T0 keeps the far-field spot / point lights. SphereLight radius, exposure and `normalize` honoured. |
| debug visuals (joints, contacts, COM, inertia, equality constraints, perturbations) | n/a | n/a | n/a | | | Not camera content in Isaac either |

### 1b. Per-world randomization (things the physics can vary per env)

| feature | T0/T1 | T2 | RT | evidence (code) |
|---|---|---|---|---|
| per-env colour (`Randomizer.colors`, event-term colour) | ✅ | ✅ | n/a | `t_colors` (N, G, 4) |
| camera pose, intrinsics, DR light, ambient, backgrounds | ✅ | ✅ (no backgrounds) | | `t_cam_spec` |
| HDR dome map per episode | ❌ | ✅ | | `Randomizer.environment` is tier 2 only |
| per-world `geom_size` (MuJoCo Warp batched size, scale randomization) | ✅ | ✅ | ✅ | **Fixed (§3c).** Every primitive is a unit mesh scaled per world in the GPU path (capsules: radius and length exact, caps scaled). Silhouette IoU vs MuJoCo at the world's size 0.36–0.82 → ≥ 0.989. Meshes: MuJoCo Warp has no per-world mesh scale (physics neither). |
| per-world `geom_rgba` / `mat_rgba` / `geom_matid` written into the MuJoCo Warp model | ✅ | ✅ | | **Fixed (§3c)** (`render/model_sync.py`): read every frame, MuJoCo's colour rule (geom rgba over the material's unless default). Mean RGB error vs the world's compiled model 17–29 → 0.0 (T0), ≤ 5 (T2 noise). `geom_matid` switches the whole material (texture with MuJoCo's mapping for the geom, PBR parameters, colour) since §3d: each env equals the renderer on the model compiled with that world's materials (T0 mean |diff| < 1 level). |
| per-world light fields of the model (`light_diffuse` / `specular` / `ambient` / `active` / `type` / `castshadow` / `cutoff` / `exponent` / `attenuation`) | ✅ | ✅ | | **Fixed (§3c).** Per-env light records from the model each frame; an inactive light emits nothing (also `active="false"` lights, which the static table drew lit: code). Tier 2's `usd_lights` keep their USD intensity. |
| per-world `cam_fovy` / intrinsics in the model | ❌ (use `t_cam_spec`) | ❌ (`set_fovy`) | | |
| roughness / metallic randomization | ◐ (shared by all envs) | ◐ | | `Randomizer.materials` writes one table through the renderer's `set_materials`. **Fixed (§3c):** the call crashed on a legacy-mode `Tier2Renderer` (`AttributeError`). |
| texture swap randomization (Isaac Lab `randomize_visual_texture_material`, `rep.randomizer.texture`) | ✅ | ✅ | n/a | **Added (§3d)**: `set_texture_pool(images, projection="uvw", texture_scale, rotations_deg)` + per-(env, slot) ids in `t_material_ids` (GPU tensor): each env's own pick, OmniPBR-style box projection (or the surface's coordinates), texel-exact vs an analytic reference on a box face. |

### 1c. Deformables, particles and other physics engines

| simulated thing | where it is simulated | renderer status | evidence |
|---|---|---|---|
| MuJoCo Warp flex cloth / cable / soft volume | `physics/deformable.py`, `tests/test_deformable.py` | ✅ every renderer and sensor (workstream A) | GPU path from `DeformableSim` / `BatchSim` buffers: 4 randomized worlds after 40 steps, every env ≥ 0.99 rgb / depth / seg against MuJoCo's renderer at the same vertices (`test_gpu_path_mujoco_warp_flex`). |
| MetalSim XPBD cloth / rope (the default cloth and rope engine per PARITY) | `physics/deformable.py` | ✅ (workstream A) | `render(xpbd, sources={"flexvert_xpos": xpbd.x})`: drawn as MuJoCo draws the model's flexes; 3 worlds ≥ 0.99 (`test_gpu_path_xpbd_and_physx_cloth`) |
| PhysX-port particle cloth | `physics/physx_cloth.py`, `tests/test_physx_cloth.py` | ✅ (workstream A) | `extras.DeformableMesh(name, tris, nvert, radius=...)` with `sources={name: sim.x}`; 3 worlds ≥ 0.99 in T0 / T1 / T2 / RT. Isaac draws PhysX cloth as its zero-thickness mesh (`radius=0`). |
| Newton models (G1 XPBD, VBD cloth / soft / rods, coupled proxy) | `physics/newton_backend.py` | ✅ rigid shapes in T0 / T1 / T2 / RT / Ann through `render/newton_bridge.py` (2026-10-12, [NEWTON_RENDER.md](NEWTON_RENDER.md)); cloth / soft-body surfaces and free particles ✅ through the dynamic-mesh and particle paths, both on main since PR #4 (2026-10-13; first verified on merges of their WIP branches) | Measured: per-body silhouette coverage 1.0 (Menagerie Panda 0.986: z-fighting between coincident links, the same with the MJCF twin), depth p99 ≤ 0.97 mm (T0) and ≤ 0.1 mm (RT) against the same MJCF through the MuJoCo path; batched and spaced worlds. Before: no Newton `State` → renderer path. |
| Newton MPM particles (sand, fluids, Franka pour) | `physics/mpm/` | ✅ T0 and T2 (render grains: RGB, depth, normals, segmentation id `n_slots + 1`), from numpy or directly from a Warp particle buffer such as Newton's `state.particle_q`; ◐ T1 draws the same raster grains, lit without ray-traced shadows (the grains are not in tier 1's ray-tracing structure; `Tier0Renderer.set_particles`, PARTICLES.md "Limits"; no tier-1 test); ❌ the ray tracer (RT depth / lidar) has no particle path. Merged in PRs #4, #9, #14, #15 (was: in progress on `metalsim-particle-render`) | `tests/test_render_particles.py` (12 tests: single-grain coverage and irradiance within 2 %, white furnace within 3 %, batched envs, bitwise determinism, two-level vs flat, T0 segmentation IoU > 0.9 against T2, Warp input = numpy input); 25.12 M grains build in 654 ms, 1080p 128 spp frame 5.1 s on the 16-inch (`runs/sand25m_16in/`); `metalsim/render/PARTICLES.md` |
| Replicator motion vectors / 3D boxes / point cloud of deformables | `replicator/isaac.py` | ◐ | RGB / depth / segmentation / instance ids ✅ (extras are unlabelled instances with their own `instance_id`); motion vectors and 3D boxes stay geom-pose based (a deformable gets no motion vector and no 3D box) |

### 1d. USD import (`scene/usd_to_mjcf.py`) visual coverage

| USD content | status | note |
|---|---|---|
| Cube / Sphere / Cylinder / Capsule / Plane / Mesh gprims | ✅ | |
| UsdPreviewSurface: diffuse colour / texture, roughness, metallic, ior, opacity | ◐ | Opacity goes to the geom's alpha and is drawn transparent since §3b (was: stored and then drawn opaque). Roughness, normal and occlusion textures are dropped. |
| OmniPBR: diffuse constant / texture / tint, roughness, metallic, opacity, specular level, emission intensity | ◐ | `texture_scale` / rotate / translate are ignored. Emissive colour is reduced to intensity. Normal and ORM maps are dropped. OmniGlass is opaque. |
| MaterialX | ◐ | As PARITY §2 (coat, sheen, transmission and normal maps not mapped) |
| mesh UVs | ✅ (importer) | **Fixed (§3c):** `st` / `st0` / `UVMap` / `uv` or the texture reader's varname; faceVarying (indexed or not), vertex / varying, uniform; quads and n-gons fan-triangulated with the positions; USD v-up mapped to MuJoCo's convention. Texture match on the probe plates (MuJoCo renderer) 0.24–0.37 → 1.00. With workstream B's unflipped atlas, tiers 0 / 2 show them as MuJoCo does. |
| authored mesh normals | ❌ | Ignored; the renderers use MuJoCo's compiled normals with its hard-edge rule since §3b (was: recomputed smooth; see the mesh-normals row) |
| GeomSubset per-face materials | ✅ | **Fixed (§3c):** one visual geom per `materialBind` subset (`<geom>_<subset>`, plus `<geom>_rest`), the whole mesh kept as an undrawn collider. Right-colour fraction top / sides 0.0 / 0.0 → 0.99 / 1.00. |
| `displayColor` per vertex / face | ◐ | First value only |
| `visibility = invisible`, purpose `proxy` | ◐ | Not honoured (purpose `guide` → hidden group 3) |
| DistantLight, SphereLight | ✅ | Sphere → point light with UsdLux radiometry (I = L π r², exposure, `normalize`), recorded as an area light unless `treatAsPoint` (§3e); lights from `mjcf_to_usd` keep MuJoCo's values (`mjc:diffuse`, `mjc:attenuation`). Sphere shaping / cone dropped. |
| DomeLight (colour, texture) | ◐ | Tier 2 only (environment map) |
| RectLight, DiskLight, CylinderLight | ◐ | **Fixed (§3c), far-field:** Rect / Disk → spot along -Z (cutoff 90° or `shaping:cone:angle`, cosine falloff, I = L·A), Cylinder → point light (I = L·2rℓ broadside); `normalize`, exposure honoured; shape / size / radiance recorded in custom text `usd_area_lights`. Before: a stage with one failed to load. Irradiance vs exact: +20–25 % at 1 light size, +2–3 % at 3, < 0.5 % at 10 for these far-field lights (tier 0); tiers 1 and 2 sample the true emitter (§3e). |
| Camera: focal length, vertical aperture | ◐ | Horizontal aperture (aspect), clipping range, orthographic projection and fisheye / f-theta are ignored |
| Points, BasisCurves, PointInstancer, TetMesh / deformable prims | ❌ | Dropped |

### 1e. Camera model (Isaac Lab `TiledCamera` / `Camera` parity)

| feature | status | note |
|---|---|---|
| pinhole intrinsics, per-env pose / intrinsics | ✅ | |
| several cameras per env in one call | ◐ | One camera per renderer instance (per-env `cam_id`); use one renderer per camera |
| depth convention | ✅ | z-depth. Background is 0 in the raw buffer and inf through `replicator/isaac.py`, as Isaac. |
| fisheye / f-theta / lens distortion | ✅ T2, RT; ◐ T0 | **Added (§3c)** `render/lens.py`: Isaac Lab `FisheyeCameraCfg`'s default f-theta polynomial (`Lens.ftheta()` / `Lens.from_isaac`) and Kannala-Brandt / OpenCV fisheye. T2 and the ray-traced depth camera trace the lens per pixel; T0 resamples cube faces (`render/fisheye.FisheyeTier0`, 5 faces for 200°). |
| rolling shutter | ✅ T0, T1, T2 | **Added (§3d, 2026-10-13)** `render/rolling.py`: RTX's per-pixel shutter-time convention (0 = previous frame, 1 = current; Isaac Lab exposes none) per image row, from per-env pose history (previous frame's geom / site / camera / light poses), rendered in row bands at interpolated poses. Edge of a moving bar and of a panning camera within 1 px of the analytic skew per row (one band per row); 8 bands within half a band's motion. Deforming meshes and particles at the current state. RT depth camera / lidar: not applied. |
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
| 3 | **Mesh normals smoothed across hard edges**. **Done 2026-10 (§3b)** | Uses authored normals, or face normals split at sharp edges | Use `mesh_normal` (MuJoCo's own per-vertex normals with its smoothing rule) or split vertices at a crease angle in `_tris_to_indexed`; read USD `normals` in the importer | 0.5–1 | **P0**: every CAD robot and object is shaded wrong (spread 43.7 vs 2.9), which is a sim-to-real appearance gap on every frame |
| 4 | **Infinite plane drawn as 10 × 10 m**. **Done 2026-10-12 (§3c)** | `GroundPlaneCfg`: 100 m (default) plane; the horizon is visible | Size planes to the camera far plane (or a large default) in the raster. In the RT, an infinite plane conflicts with the per-env spatial offset (`stride`), so use a separate per-env analytic plane test in the shaders, or clip to the stride. | 1–2 | **P0** for locomotion / navigation cameras, P2 for tabletop |
| 5 | **T0/T1 10 m far plane**. **Done 2026-10-12 (§3c)**: `far=`, reversed z, tier-1 ray limit | Camera `clipping_range` default (0.01, 1e6)/(0.1, 1e5) | **WIP in the main checkout** (`far=`). Still to do: T1's 50 m and T2's 1000 m ray ranges, the splat composite far plane, and depth precision (TEXTURES.md "Far plane") | 0.5 (land the WIP) | **P0** for outdoor / nav; P2 tabletop |
| 6 | **Transparency / opacity**. **Done 2026-10 (§3b); in the ray-traced sensors 2026-10-13 (§3e)** | RTX: OmniGlass refraction, opacity cutout, alpha blending; depth and segmentation of transparent objects configurable | T0: sorted alpha-blend pass for alpha < 1 (order-independent transparency or per-object sort), seg / depth from opaque only or configurable. T2: `any_hit` alpha test + a dielectric BTDF. Importer: keep opacity. | 2 (T0 blend + cutout), +2 (T2 glass) | **P1**: glassware, bottles and windows are common in manipulation datasets |
| 7 | **Sites and tendons invisible**. **Done 2026-10-12** (§5; sites on by default as in MuJoCo, off in the ray-traced sensors) | n/a for sites (no USD equivalent); tendons become visual cables/meshes authored by the asset | Optional: draw sites (opt-in group flag) as primitives. Draw spatial tendons as per-frame capsule chains from `ten_wrapadr` / `wrap_xpos` (dynamic instances, gap 1 infrastructure). | 1 (sites), 2 (tendons) | P1 for tendon-driven hands and cable tasks; P2 for sites (MuJoCo-visual parity only) |
| 8 | **Skins invisible**. **Done 2026-10-12** (§5) | Skinned meshes (UsdSkel) render | Linear blend skinning kernel into the dynamic-mesh path (#1) | 1 on top of #1 | P2 (rare in robot models; composite cables and older flexcomps) |
| 9 | **SDF geoms invisible**. **Done 2026-10 (§3b)** | n/a (USD has meshes; SDF is a collision representation) | Draw the SDF's mesh asset (`geom_dataid` → mesh, as MuJoCo does) | 0.5 | P1: anything using SDF plugins (gears, nuts and bolts) is invisible |
| 10 | **Spot / point-light shadows in T0/T1**. **Done 2026-10 (§3b)** | All lights cast shadows in RTX | T1: ray-traced shadow for every light (the loop already exists, gate on `castshadow`). T0: cube or spot shadow maps per light, or route shadowed lights to T1. | 0.5 (T1), 3 (T0 maps) | P1: indoor / spot-lit scenes |
| 11 | **Texture mapping errors** (vertical flip on mesh UVs, no UVs on cylinder / capsule, sphere phase, cube textures). **Done 2026-10 (§3b)** | Asset UVs used as authored; MDL projections | Flip V in the atlas path (with a regression image), MuJoCo-style generated coordinates for primitives, cube-map sampling (`texturecube`) for `type="cube"` | 1.5 | P1: textured objects and DR textures look wrong; mirrored decals and labels |
| 12 | **Per-world `geom_size` (scale DR) ignored**. **Done 2026-10-12 (§3c)** | Isaac scale randomization changes the rendered prim | Generalize `SizedGeoms` (already done for terrain slots) to every primitive, plus a per-world scale for meshes | 1 | P1: size randomization is invisible to the policy |
| 13 | **Per-world model colour / light fields ignored**. **Done 2026-10-12 (§3c)** | Replicator writes USD attributes the renderer reads | Read MuJoCo Warp's batched `geom_rgba` / `mat_rgba` / `light_*` if the event terms write them, or document `t_colors` as the only path (it already exists) | 0.5 | P2 (a workaround exists) |
| 14 | Non-RGB texture roles, normal maps, USD GeomSubsets, UV scale / rotate / translate | RTX uses all of them | Material table extension + importer | 3–5 | P2 (fidelity) |
| 15 | MJCF skybox as a mean colour. **Done 2026-10 (§3b)** | Dome texture background | Convert skybox to an equirect / cube environment (T2 has the sampler; T0 needs a background pass from it) | 1 | P2 (backgrounds are usually randomized anyway) |
| 16 | Area lights, more than 8 lights in T0/T1. **Done 2026-10-13 (§3e; lights: §3b)** | Unlimited lights, area lights with soft shadows | Raise `MAX_LIGHTS` (loop cost), area-light sampling in T2 | 0.5 / 2 | P2 |
| 17 | hfield sides / base. **Done 2026-10-12 (§3c)** | Terrain meshes are closed | Add skirt triangles | 0.25 | P2 |
| 18 | Fisheye / lens distortion. **Done 2026-10-12 (§3c)**: f-theta and Kannala-Brandt in T2 / RT, T0 cube faces | `FisheyeCameraCfg`, f-theta | Per-pixel ray directions in T2 and RT (trivial); T0 needs a distortion warp pass | 1–2 | P2 (P1 for wide-FOV nav cameras) |
| 19 | MPM particles. **Done 2026-10-13 (T0, T2; PRs #4, #9, #14, #15; `metalsim/render/PARTICLES.md`)**; T1 draws the raster grains without ray-traced shadows; RT not covered | Points / mesh-from-particles (fluid isosurface) | Render grains (instanced grain meshes) in T0 and T2 | — | P0 for the MPM tasks |
| 20 | Splat composite: no segmentation for the splat background, baked lighting (no robot shadows), 10 m composite limit | Isaac Sim 5.x renders 3DGUT Gaussian scenes inside RTX | **WIP elsewhere**; seg = "background" class by design | — | P2 |

## 3. Ranked fix list

| rank | fix | effort (days) | why first |
|---|---|---|---|
| 1 | Dynamic-mesh path for flex / XPBD / PhysX-port cloth (#1), with skins (#8) and tendons (#7) on the same infrastructure: **done 2026-10-12** (§5) | 4–6 (+1 skins, +2 tendons) estimated; done in about 4.5 hours | A whole simulated category, which PARITY claims at parity with Isaac's deformables, is invisible to cameras |
| 2 | Mesh normals (#3): **done 2026-10 (§3b)** | 0.5–1 | Wrong on every frame of every mesh robot; cheapest high-impact fix |
| 3 | Land the far-plane WIP (#5) and make the infinite plane infinite (#4): **done 2026-10-12** (§3c) | 1.5–2.5 | Locomotion and navigation cameras see a 10 m world with a floor edge |
| 4 | Newton → renderer adapter (#2): **done 2026-10-12** (`render/newton_bridge.py`) | 2–3 | Without it the Newton backend has no camera at all |
| 5 | SDF geoms (#9): **done 2026-10 (§3b)** | 0.5 | Silently invisible objects |
| 6 | Texture mapping fixes (#11): **done 2026-10 (§3b)** | 1.5 | Mirrored and missing textures on common primitives |
| 7 | Per-world size DR (#12): **done 2026-10-12** (§3c) | 1 | Randomized sizes invisible to the policy |
| 8 | Transparency (#6): **done 2026-10 (§3b)** | 2–4 | Glass / plastic objects |
| 9 | Spot / point shadows in T1 / T0 (#10): **done 2026-10 (§3b)** | 0.5–3 | Indoor lighting cues |
| 10 | The P2 items (#13–#18): #13, #15, #16, #17, #18 done (§3b, §3c, §3e); #14 (non-RGB texture roles, normal maps, UV transforms) open | ~10 total | Fidelity |

MPM particle rendering (#19) is not ranked: it was built by another agent (now merged, §1c row).

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
- Heightfields drawn by the culled tile path (`hfield_cull`, the default) cast only the caster's shadow in T0 (sides and base too since §3d; `skirt_shadows=False` = before). T1 and T2 trace them.
- Lidar and the ray-traced depth camera: transparent geoms handled since 2026-10-13 (§3e).
- A capsule's cube-map coordinates use one scale for the whole capsule; MuJoCo's hemispheres differ slightly.
- Planes seen from behind are not made more transparent, as MuJoCo does.

**Cost.**
- **Setup:** `scripts/diagnostics/render_appearance_cost.py`; M4 Max, exclusive `timing` job, 2026-10-12 08:08 after the reboot; render only, 128×128, best of 3×10 frames; raw rows in `docs/measurements/render_appearance_cost.jsonl`.
- **Configurations:**
  - **before:** `appearance="legacy"`, `transparency=False`, `shadow_lights="caster"`, `max_lights=8`, `skybox=False`.
  - **split:** the fixes with the split-vertex layout and no shadow-view culling (the first version).
  - **corner_vs:** vertex pulling.
  - **after:** the defaults, `normal_layout="corner_fs"` and `shadow_cull=True`.

| scene | renderer, envs | before | split | corner_vs | after (default) | note |
|---|---|---|---|---|---|---|
| SO-101 box task, full meshes | T0, 1024 | 142.1 ms | 271.0 | 256.6 | **141.9** | vertices 162,026 → 618,831 (split) → 161,927 (shared) |
| SO-101, decimated to 3000 faces per mesh (the lift task's training setting) | T0, 1024 | 19.19 ms | 27.96 | 33.87 | **19.18** | |
| G1 rough terrain, head camera (the G1 task's camera path) | T0, 1024 | 34.69 ms | 34.43 | 34.34 | **34.32** | no change |
| Franka Panda | T0, 1024 | 53.97 ms | 175.2 | 109.9 | **78.8** | +1 shadow pass for its shadow-casting spot light; culling 78.4 vs 78.8 (the cone covers the robot) |
| point + spot light probe | T0, 1024 | 2.51 ms | 3.88 | 3.06 | **3.33** | 7 extra shadow views; culling 4.16 → 3.33 |
| glass pane | T0, 1024 | 2.09 ms | 1.95 | 2.11 | **2.47** | transparent pass |
| SO-101 / SO-101 decimated / Panda | T1, 256 | 44.4 / 21.5 / 22.3 ms | | | **43.9 / 21.2 / 22.1** | |
| glass pane / point + spot | T1, 256 | 1.00 / 1.62 ms | | | **1.98 / 1.65** | transparent pass |
| SO-101 / Panda | T2, 64 (4 spp, 2 bounces) | 18.6 / 22.2 ms | | | **18.7 / 22.5** | |
| glass pane | T2, 64 | 1.71 ms | | | **8.48** | presence through the pane, attenuated shadow rays; the before path drew it opaque |

**Result.** With `corner_fs`, correct hard-edge normals cost nothing measurable: SO-101 full meshes 141.9 vs 142.1 ms, decimated 19.18 vs 19.19 ms. The remaining additions are features that were missing before:
- Panda's spot-light shadow pass, +25 ms at 1024 envs.
- The transparent passes on scenes that have transparent geoms.

**Shadow-view culling.** Per env, against each view's frustum. It pays where the views see a fraction of the scene (point lights: 4.16 → 3.33 ms) and is neutral otherwise. The Panda spot cone covers the whole robot, so its shadow pass has nothing to cull.

**Tier-1 Panda kernels, run alone first after the reboot.** Exclusive `timing` job under `watchdog_run.py` (60 s stall limit), as the coordinator asked, because the contended run had been on Panda tier 1 when the GPU hung. Tier 1 at 1, 4, 64 and 256 envs, legacy and default settings: every launch returned (≤ 0.41 s), no stall.

Every loop added to the shaders is bounded:
- `sort_transparent`: at most 64 objects.
- `transmit`: 8 surfaces.
- Presence passes: 16.
- The `shadow_cull` plane test.

**Hard-edge normals without duplicated vertices** (`Tier0Renderer(normal_layout=...)`, default `"corner_fs"`). The first version's slowdown (the split column above) came from the split-vertex layout: one vertex per distinct (position, normal, uv), so hard edges quadrupled SO-101's vertices (162 K → 619 K) and the raster is vertex-bound. Two alternatives keep vertices shared by position and store each triangle corner's normal / uv in a per-corner table in index order (`scene_tables` `corner_attributes`):

- **`corner_fs`.** The fragment stage reads the triangle's three corners by `primitive_id` and blends them with the hardware's perspective-correct `barycentric_coord`, then rotates by the geom's `geom_xmat`. This is a function-constant specialization of `geom_fs` / `geom_fs_rt`.
- **`corner_vs`.** A non-indexed draw: the vertex stage pulls the position by index and the normal / uv per corner (`geom_vs_pull`).

Height-field tiles keep their vertex normals. The transparent pass pulls the corner table in `trans_vs`. Shadow passes need positions only, and now draw the shared ones.

**Measured on the 14-inch M4 Max** (`scripts/diagnostics/render_normal_layout.py`, check job; ratios, not headline timings; 256 envs, 128×128, render only; the exclusive numbers are in the cost table above):

| scene | `split` | `corner_fs` | `corner_vs` | vertices (split → shared) |
|---|---|---|---|---|
| SO-101, full meshes | 71.5 ms | **33.2 ms** (2.15×) | 67.9 ms | 618,831 → 161,927 |
| Franka Panda | 45.7 ms | **20.2 ms** (2.26×) | 28.7 ms | 395,565 → 69,358 |

`corner_vs` runs the vertex stage once per corner (975 K for SO-101), which loses most of the gain on Apple's GPU. `corner_fs` keeps post-transform reuse, and its three extra corner reads per fragment cost less than the vertex work they replace.

**Equivalence** against `split` (SO-101, Panda, glass, mesh box, point + spot shadows, textures; tier 0 and tier 1; rgb, depth, segmentation, normals):

| variant | rgb | depth / segmentation | normal output |
|---|---|---|---|
| `corner_vs` | bit-identical | bit-identical | bit-identical |
| `corner_fs` | ≤ 1 level on 0.18 % of SO-101 pixels; ≤ 2 levels on 0.07 % of Panda's; identical on the probes | bit-identical | within 5e-4, the 16-bit target's rounding |

The `corner_fs` differences come from blending the corner normals in the fragment stage rather than interpolating the rotated vertex normals.

**Culling the extra shadow views** (`shadow_cull=True`, default): each extra shadow view draws only the instances whose bounding sphere meets its frustum, per env. A reset and a cull kernel build compacted per-view instance lists; the draws are indirect. The result is bit-identical in every scene above. Panda's spot cone covers the whole robot, so its time is unchanged (20.2 vs 20.2 ms). The saving is for point lights (six faces) and narrow spots.

**Speed levers:**
- `normal_layout="corner_fs"` (default).
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

## 3d. Render gaps 2: rolling shutter, per-env textures, haze, tendons, heightfield-side shadows; render cost by feature (2026-10-13)

Branch `render-gaps-2` on `origin/metalsim-mpm` 532577fe (PR #4 merged). M4 Max, Warp 95f6f121 (the pinned fork). References:
MuJoCo 3.14.0 source and `mujoco.Renderer`, Isaac Lab clones and NVIDIA's docs; citations in the research note
`research_notes/Render sensor fidelity/gaps_and_perf.md` (outside the repo). Tests: `tests/test_render_rolling.py`,
`tests/test_render_env_textures.py`, `tests/test_render_tendons.py`, `tests/test_render_haze.py`; costs:
`scripts/diagnostics/render_gaps2_cost.py`, `scripts/diagnostics/render_perf_profile.py` (logs and JSON in `runs/render_gaps2/`).

| item | reference | MetalSim | measured | switch |
|---|---|---|---|---|
| rolling shutter | Isaac Lab: none. Omniverse RTX (Kit 109): per-pixel shutter time in [0, 1] (0 = start of the previous frame, 1 = current), instantaneous per-pixel exposure, global shutter by default | `render/rolling.py`: per-row time in RTX's convention, poses of the previous frame kept per env on the GPU (geoms, sites, cameras, lights), the image rendered in row bands at interpolated poses (linear / slerp); T0 / T1 re-render the frame per band and write its rows, T2 refits the structure per band and traces its rows | moving bar and panning camera: left edge within 1 px of the analytic skew in every row (T0, T2, one band per row); 8 bands within half a band's motion; GPU path equals the host path given the previous state | `rolling_shutter=None` (default: global shutter) |
| per-env textures / materials | Isaac Lab `randomize_visual_texture_material`: a new OmniPBR per prim with a random diffuse texture, `project_uvw`, `texture_rotate` (needs `replicate_physics=False`, redraws all envs); MuJoCo: a world's `geom_matid` is the whole material | `render/env_materials.py`: (env, slot) entry ids resolved on the GPU each frame (the renderer's choice, else a per-world `geom_matid`); entries are full material blocks (MuJoCo mapping per geom) or texture-pool images (2D texture array, box projection or the surface's coordinates, scale, rotation) | per-world `geom_matid`: each env equals the renderer on the model compiled with that world's materials (T0 mean |diff| < 1 level; T2 < 3 with the same seed), its box texture correlates with mujoco.Renderer > 0.8; texture pool: box top face texel-exact vs the analytic projection (median error < 6 levels, T0 / T2), 90-degree rotation checked | nothing set: compiled out, byte-identical |
| haze over infinite planes, fog | `render_gl3.c` skybox pass (haze cone, `map.haze`, `rgba.haze`), GL linear fog on eye depth (`map.fogstart / fogend`, `rgba.fog`; measured z-depth, not radial) | `render/haze.py`, `shaders/haze.h`: the faceted cone per pixel at the first opaque surface (T0 / T1 fragments and background, T2 camera ray), fog of fragments, skybox and haze | haze alpha vs mujoco.Renderer at H = 20 m, pitch -20 deg: T0 / T1 / T2 max 3 levels (p99 1) over the image | `haze=True` (MuJoCo's default), `fog=False`; compiled out when neither applies |
| tendon catenary | `mjv_isCatenary` / `mjv_catenary`: slack two-site spring or unreached limit under gravity, numslices + 1 points, bead when L > 100 h | catenary points per (tendon, env) on the GPU (float32 Newton), capsules per segment | capsules equal to `mjv_updateScene`'s to 2e-5 m (CPU reference of the kernels; Metal within 2e-4 m of it); coverage of MuJoCo's silhouettes ≥ 0.97 rgb / depth / seg in T0 / T1 / T2 / RT; `tendons="straight"` < 0.6 on the hanging ones | `tendons="straight"` |
| tendon limit colour | default-grey tendons without material mixed toward `rgba.constraint` by the limit impedance (`efc_KBIP[2]`) | impedance from `tendon_solimp_lim` at the limit distance, per env, written over the synced colour | tint table and GPU colour equal MuJoCo's scene rgba to 2e-6; image colour within 12 levels of mujoco.Renderer on the tendon | `tendons="straight"` |
| heightfield sides / base in T0 shadows | sides cast shadows (MuJoCo, T1 / T2) | culled side chunks also drawn into the caster's shadow map (swept to the scene's lowest point) | floor shadow mask IoU vs tier 1's traced shadow 0.818 → 0.991 | `skirt_shadows=False` |

**Byte identity** (`scripts/diagnostics/render_identity_check.py`, 8 scenes incl. Menagerie Panda / Shadow hand / Go2, the
extras and drape scenes; T0 rgb / depth / seg / normals, T1, T2, ray depth, lidar; against 532577fe): with the new switches
off 121 of 121 outputs identical (this includes tier 2's any-hit shadow path and the shader specializations below); with
the defaults 118 of 121, the 3 others the SO-101 scene's rgb, which shows MuJoCo's haze (skybox + infinite floor).

**Cost of the additions** (exclusive `timing` jobs, ms per frame, best of 3 × 10, GPU path):

| addition | scene, renderer, envs | on | off |
|---|---|---|---|
| haze | Menagerie Go2 (skybox + infinite plane), 128², T0 1024 / T1 256 / T2 64 | 123.6 / 33.9 / 2.68 | 123.5 / 33.1 / 2.43 |
| heightfield-side shadows | raised 64 × 64-cell heightfield over a floor, T0 1024 | 8.58 | 8.33 |
| tendons (catenary) | the tendon test scene + floor, T0 1024 / T2 64 | 22.6 / 5.32 | 4.38 / 2.49 |
| per-env textures | the texture test scene, T0 1024 / T2 64: none, texture pool on every box, per-world `geom_matid` | 2.82 / 2.90 / 2.88; 1.64 / 1.65 / 1.66 | |
| rolling shutter | cloth scene 64², T0 1024 / T2 256: global, 8 bands, one band per row | 1.99 / 15.1 / 120; 4.91 / 8.51 / 42.6 | |

The catenary's cost is its geometry: MuJoCo draws numslices (28) capsules per hanging tendon at its full tessellation
(85 K triangles per env against 3.9 K straight; per-pass: colour 2.1 → 9.6 ms, shadow 1.4 → 8.7, generation 3.6); with
`tube_slices=8, cap_stacks=2` the same scene costs 5.2 ms. Rolling shutter costs about one frame per band in T0 / T1 (every
pass is repeated at the band's poses) and one structure build per band in T2 (rows are traced once).

### Render cost by feature: the 2026-10 slowdowns

**Reported**: after PR #4, tier 1 and tier 2 about 30 % slower (cloth at 4096 envs: tier 1 71 ms, tier 2 96 ms) and the glass
pane in tier 2 about 5× slower (1.71 → 8.48 ms at 64 envs). **Reproduced** on the pinned Warp (`render_perf_profile.py`;
trees exported from the branch's commits; cloth = deformable_render_bench's flex scene with one spot light, 64²; tier 2 at
1 spp / 1 bounce; glass at 128², 4 spp / 2 bounces):

| frame (ms) | 2ef6641f (before workstream C) | 5a7cc077 (C merged) | 532577fe (PR #4) | this branch | all fidelity switches off |
|---|---|---|---|---|---|
| tier 1 cloth, 1024 envs | 13.23 | 18.06 | 17.91 | **14.68** | 13.76 |
| tier 1 cloth, 4096 envs | 52.86 | 71.70 | 71.41 | **58.49** | 55.51 |
| tier 2 cloth, 1024 envs | 17.77 | 24.38 | 24.37 | **18.18** | 18.63 |
| tier 2 cloth, 4096 envs | 69.15 | 96.27 | 96.01 | **71.78** | 75.53 |
| tier 2 glass pane, 64 envs | 4.16 | | 12.15 | **4.60** | 1.63 |

**Where it went.** Per-pass GPU timing (`metalsim/render/gpu_profile.py`: one command buffer per encoder) put the whole
regression in one pass each: tier 1's colour pass (5.6 → 10.4 ms at 1024 envs; the acceleration-structure work, 7.8 ms,
unchanged) and tier 2's `path_trace` kernel (16.9 → 23.5 ms). Shader-variant experiments (temporary `#ifdef` toggles,
`runs/render_gaps2/exp_1024.log`) found the cause: workstream C's infinite-plane continuation (`env_grid.h grid_plane`, a
loop over up to 4 planes inlined into every ray query) costs it in register pressure even when the scene has no infinite
plane and the loop never runs: removing it took tier 1 18.0 → 15.1 ms and tier 2 28.2 → 20.2 ms; per-env light records
(0), the ray-reach checks (0) and the reflection's plane test (0) did not matter. The same effect made this branch's own
first haze composite and any-hit shadow path cost tier 2 another 2.5 ms each until they were specialized too.

**Fixes (identical output):** the ray kernels are compiled per scene: `GRID_PLANES` = the scene's number of continued
infinite planes (0 compiles the continuation out; a constant loop bound otherwise), `HAZE_ON` and `ENV_MATS` only when
those features are active, so a scene pays only for what it uses. Tier 2's shadow rays in an env without a partially
transparent surface (a per-env flag the model sync writes each frame) take an any-hit query instead of the closest-hit
transmission loop (same result; 18.5 → 18.2 ms on cloth). Byte identity: 121 of 121 outputs with the switches off.

**What remains, and why:**
- Tier 1, +11 % over 2ef6641f (58.5 vs 52.9 ms at 4096): 2.7 ms is the spot light's shadow ray (Isaac / RTX shadows every
  light; `shadow_lights="caster"` 55.8 ms), the rest (~3 ms, 5 %) is not attributed to any one switch (all switches off:
  55.5 ms).
- Tier 2 cloth: within 4 % of 2ef6641f (71.8 vs 69.1 ms); the dynamic layout dominates instead: `dyn_layout="per_env"`
  57.7 ms (sbvh is the default for tier 2; §5 measured it 13 % slower on cloth too).
- Glass pane, tier 2: transparency itself (presence sampling of the pane, transmitted shadow rays) is the fidelity cost:
  4.60 ms vs 1.63 ms with `transparency=False`, 2.8×; 2ef6641f had the same feature at 4.16 ms. The compact env grid's
  cell clipping saves time here (legacy grid 6.36 ms).
- An infinite floor (cloth scene with `size="0 0"`): tier 2 22.9 ms vs 18.9 with the floor finite and nothing continued
  (the ground to the horizon is more visible surface to shade), tier 1 15.0 vs 13.9.
- Not measurable as a cost: per-env light records (model_fields=False ±0.1 ms), reversed z, appearance="legacy",
  `max_lights`.

## 3e. Transparent surfaces in the ray-traced sensors; true area lights (2026-10-13)

Research notes: `research_notes/Render sensor fidelity/transparency_area_lights.md` (Isaac Sim RTX lidar and
annotators, MuJoCo's `mj_ray` and renderer, real lidars on glass, UsdLux light units). Measurements on an M4 Max, Warp 95f6f121, on main f3b5c62e
(render-gaps-2 merged); logs in `runs/render_sensor_tr/`.

**What the references do.**

| | MuJoCo | Isaac Sim | real lidars |
|---|---|---|---|
| ray / rangefinder | `mj_ray` skips alpha exactly 0 (the material's alpha when the geom has one); alpha 0.25 or 0.99 stops the ray | – | – |
| depth / segmentation | `mujoco.Renderer` (measured): alpha > 0 is the surface (1.995 m on a pane at 2 m), alpha 0 invisible | glass: depth on the surface (NVIDIA staff); opacity cutout: skipped (Real-Time) or stochastic presence (path tracer); `opacity_threshold` t > 0: opaque when opacity > t | – |
| lidar | (the rangefinder above) | non-visual materials (`clear_glass`, `mirror`, ...); `maxReturns`, `reflection/transmissionPowerFraction`, `returnType` First / Last / Strongest / Multiple; intensity formula unpublished; glass pass-through reported broken in 6.0 (bug 6426827) | ~92 % of the beam passes a pane; a specular return only within a few degrees of normal incidence; mirror ghosts behind the glass; Strongest / Last / First and dual modes |

**Implemented** (`metalsim/sensors/raytrace.py`, `shaders/rt.metal` SENSOR_TR):

- Depth camera `transparent_ids`: `"surface"` (default: MuJoCo and Isaac's glass), `"behind"` (B's Isaac Real-Time
  cutout mode), `alpha_threshold=t` (Isaac's `opacity_threshold`), `"opaque"` (previous).
- Lidar `transparency`: `"transmit"` (default): a pane of alpha a returns `P a rho cos / r^2` and passes `P (1 - a)`,
  echoes grouped and chosen as before; `"surface"` (MuJoCo's rangefinder), `"behind"`, `"opaque"` (previous).
  `lidar_materials` presets named after the SimReady bases (`clear_glass`: Fresnel 0.077 at normal incidence with a
  Beckmann lobe, 92 % transmitted; `mirror`: ghost echoes at the reflected path length; `frosted_glass`, `plexiglass`,
  `water`, `one_way_mirror`, `calibration_lambertian`). `return_mode` `strongest` / `first` / `last` /
  `strongest_last` (Velodyne / Hesai dual) / `first_last` / `first_strongest`; `min_reflectance` at
  `min_reflectance_range` (Isaac `minReflectance`, `minReflectionRangeM`) as the detection threshold.
- One sub-ray and one return run a streaming kernel (`lidar_tr1`, no candidate arrays), bit-identical to the general
  one (points to float rounding) (`test_streaming_kernel_matches_lidar_ext`).
- Scenes without transparent geoms run the unchanged kernels: SO-101, the lidar test scene and ANYmal C, default and
  extended lidar and depth camera, 8 envs, **36 of 36 arrays byte-identical** to `origin/main` (f3b5c62e)
  (`scripts/diagnostics/sensor_identity.py`).

**Checks** (`tests/test_sensor_transparency.py`, glass panes 2 m in front of a wall at 3 m):

| check | result |
|---|---|
| depth `"surface"` vs `mj_ray` through every pixel (48 x 48) | max error < 0.2 mm |
| depth `"behind"` on pane pixels | wall (3.000 m); other pixels unchanged |
| `alpha_threshold` 0.3 | the 0.25 pane cut out, the 0.5 pane kept |
| lidar `"transmit"`, two returns, three angles | ranges to 0.2 mm, intensities `a rho cos / r^2` and `(1 - a) rho cos / r^2` to 1e-4 |
| lidar `"surface"` vs `mj_ray` | equal to 0.2 mm |
| three echoes (two panes, wall): first / last / strongest / three dual modes / threshold | as defined |
| `clear_glass` 4 and 14 deg off normal | no glass echo; wall at (1 - F) rho cos / r^2 |
| `clear_glass` at normal incidence | glass echo pi F D / 4 / r^2 (48.4 vs 0.051 for the wall behind), both reported |
| `mirror` at 45 deg | one echo at the path length, reported behind the mirror along the beam, intensity F rho / r^2 to 1e-3 |
| 0.5 deg divergence, 16 sub-rays | echo powers equal the single ray's (0.2 %); two traces bit-identical |

**Cost** (GPU time per trace, each sensor alone, median of 20; 16 x 360 beams, 64 x 64 depth; `glass` = the lidar test
scene with three alpha-0.3 panes around the sensor, crossed by 33 % of the beams; ramp 1024 then 4096 envs, each run
alone through `scripts/gpu_run.sh`, GPU utilisation back under 10 % after each):

| sensor | plain scene 1024 / 4096 | glass scene 1024 / 4096 |
|---|---|---|
| lidar, default kernel (`"opaque"` in the glass scene) | 1.64 / 6.86 ms | 2.36 / 9.44 ms |
| lidar `"surface"` | – | 2.45 / 9.79 ms |
| lidar `"behind"` | – | 4.07 / 16.28 ms |
| lidar `"transmit"` (default), one return | – | 5.77 / 23.34 ms |
| lidar two returns (`"transmit"`; `strongest_last` 9.61 / 38.22) | 4.12 / 17.11 ms | 9.65 / 38.45 ms |
| depth `"surface"` (no see-through slot: plain kernel) | 0.58 / 2.32 ms | 0.86 / 3.24 ms |
| depth `"behind"` | – | 2.40 / 9.23 ms |

A beam that crosses a pane casts three rays (pane, its exit face, what is behind), so the see-through modes cost
in proportion to the glass in view. `"transmit"` costs about 1.4x `"behind"` for the same rays (its kernel keeps the echo state live across the ray
queries; an inner skip loop, a forced-inline variant and a lighter echo state were tried, within noise). Scenes without transparency pay nothing.

**Area lights** (`metalsim/render/area_lights.py`, `shaders/area_light.h`; tier 2 AREA_LIGHTS, tier 1 AREA_LIGHTS):

- Import: Rect / Disk / Cylinder unchanged (far-field spot / point lights for tier 0) plus the prim's X axis;
  SphereLight now follows UsdLux radiometry (`intensity x 2^exposure x color`, `/ 4 pi r^2` with `normalize`,
  far-field I = L pi r^2; before: diffuse = intensity, radius and exposure dropped) and is recorded as an area light
  unless `treatAsPoint`. Lights written by `mjcf_to_usd` carry `mjc:diffuse` / `mjc:attenuation` and keep MuJoCo's
  values exactly.
- Tier 2 (`area_lights=True`, the default when the model records them): next-event estimation on the emitter,
  uniform in area on the rectangle / disk / the half of the cylinder facing the point, uniform in the subtended cone
  for spheres; one shadow ray per sample (soft shadows). The emitters' frames follow their bodies per env (`xmat`,
  GPU and host paths; `test_area_light_moving_body_gpu_path`).
- Tier 1: `rt_samples` samples on the emitter with a shadow ray each; the unshadowed irradiance is the closed form
  (Lambert's polygon formula for rect / disk, the sphere's formula; ratio estimator: noise only in penumbrae).
  Tier 0 keeps the far-field lights.

Floor irradiance against the analytic value (`tests/test_area_lights.py`; white Lambertian floor, direct light only,
24 x 24 px, 4096 samples per pixel, reference averaged over 3 x 3 points per pixel):

| emitter | 0.5 size: mean / worst 4x4 block | 1 size | 3 sizes |
|---|---|---|---|
| rect 0.4 x 0.2 m, turned 30 deg (exact: Lambert's formula) | +0.07 % / 0.99 % | +0.01 % / 0.52 % | +0.00 % / 0.19 % |
| disk r 0.2 m, `normalize` (4096-gon) | −0.09 % / 0.97 % | −0.03 % / 0.40 % | −0.02 % / 0.15 % |
| sphere r 0.2 m (pi L (R/d)^2 cos) | −0.00 % / 0.15 % | −0.00 % / 0.09 % | −0.01 % / 0.06 % |
| cylinder r 0.05, 0.4 m (quadrature) | +0.07 % / 1.25 % | +0.00 % / 0.81 % | −0.02 % / 0.58 % |
| rect, far-field import (before), on axis at 1 size | +19.1 % | | |

Tier 1 (`rt_samples` 16, 8-bit output): unshadowed floor vs the closed form, rect / disk / sphere at one light size:
median 0.27 / 0.50 / 0.27 %, max 1.5 / 1.5 / 1.2 % (quantisation). Soft shadow of a 12 cm box under a 0.4 x 0.2 m rect
(`rt_samples` 64) vs tier 2's converged image: mean |diff| 0.87 % of the lit level, p99 10.5 % (noise in the
penumbra), 1874 penumbra pixels of 2304 (a point light gives a hard edge). Tier 0 keeps the far-field lights.

Scenes without area lights compile neither AREA_LIGHTS variant (tier 1, tier 2): their shaders are unchanged.

Limits: the area lights are not visible to the camera (Isaac's are when `visibleInPrimaryRay`); a light's shaping cone
softness and IES profile are dropped; particle grains (tier 1) shade an area light with its radiance as a spot light;
the rolling shutter uses the frame's emitter frame.

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
instanced draw (instance = env). The ray tracers take the dynamic objects in one of two layouts (`dyn_layout`, below):
"sbvh" (default for sensors and the path tracer) or "per_env" (default for tier 1). The dynamic records use the static
vertex layout; a hit's slot is its instance's first dynamic slot + the object, its mesh the (env, object) entry, which for
every geom (one geometry) is exactly the previous lookup, so geom hits are unchanged.

**Dynamic geometry in the ray tracers: two layouts, chosen by measurement.**
- *sbvh*: one bounding box per env in a single box structure (`metalsim/sensors/dyn_bvh.py`). A ray entering env e's
  box traverses e's own BVH in software (`shaders/dyn_bvh.h`, inside `intersection_query`; watertight triangle test,
  as hardware tracers use), in the env's frame. The tree shape is shared by every env (median splits over env 0's
  first-frame triangles, leaves of 4); each frame one compute pass refits every env's node bounds and copies its
  triangles in leaf order, level by level, then the box structure (n_envs boxes) is rebuilt. Every
  `dyn_rebuild_every` frames the leaves are re-split from env 0's current triangles on a host thread and swapped in
  through a second order buffer. One hardware structure at any env count. Kernels are compiled with `DYN_SBVH` only
  for tracers using it; every other shader is the plain intersector as before.
- *per_env*: one primitive structure per env, a geometry per object, in the env's frame, refit each frame and rebuilt
  every 32 frames (staggered: structure k on frames where frame + k is a multiple), allocated from one heap (one
  residency call); scratch from a pool of up to 32 regions capped with the heap at a quarter of the recommended working
  set (Cycles' rule); more than 512 structures: generation and refits in their own command buffers of 512 with the
  frame's event waits; `extendedLimits` whenever a count passes Metal's standard limits.
- A third prototype, merged tiles (at most 64 structures, a geometry per (env, object) placed by a per-geometry
  translation), was dropped: never the fastest, tile-local vertices up to a tile's extent of the env stride (tier 2
  depth off by up to 1.4 mm), and 469 ms for the Aero hand's tier 2 at 1024 envs (ramp stopped). Commit 1463fd6e has it.

Equality with per_env (5 envs; wrapping tendons, draped cloth, extras scene): tier 1 byte-identical; ray depth and
tier 2 depth within 0.0007 mm except one pixel per env (an edge-on triangle at a tendon silhouette) and 2 tier-2 pixels
at 0.12 mm (with the legacy env grid per_env itself differed by up to 1.76 mm between env 0 and env 4 in tier 2; see the env grid below).

Ramp (one exclusive queue job per layout; each step its own process under a 60 s watchdog with device utilisation
back under 10 % after it; ms per frame, each frame synchronized, 64×64; per_env / sbvh):

| scene, renderer | 1024 envs | 4096 envs |
|---|---|---|
| cloth (944 dynamic triangles per env), depth camera | 9.6 / 3.3 | 34.5 / 11.7 |
| cloth, tier 1 | 13.1 / 21.7 | 50.0 / 84.8 |
| cloth, tier 2 (1 spp, 1 bounce) | 16.8 / 19.0 | 64.0 / 74.1 |
| Aero hand (42 K triangles per env), depth camera | 71.0 / 23.6 | |
| Aero hand, tier 1 | 96.2 / 148 | |
| Aero hand, tier 2 | 96.4 / 59.1 | |

sbvh is 2.9–3x faster for depth cameras and lidars and 1.6x for the path tracer on many thin tubes, 13 % slower for the
path tracer on cloth, and 1.5–1.7x slower for tier 1, whose shadow and ambient-occlusion rays run in the fragment stage
(software traversal there costs more than the hardware's), hence the per-renderer defaults. Over 300 frames of
draping cloth (256 envs) sbvh's ray-depth frame grows 1.26 → 1.56 ms and a fresh tracer on the final state takes
1.50 ms: the growth is the workload (more cloth on screen), not loosened bounds (per_env 3.08 → 3.13, fresh 2.90).
No anomaly, stall or ownerless GPU in any run of either layout up to 4096 envs.

**Env grid: where the envs sit in the ray tracers' shared world** (`metalsim/sensors/env_grid.py`, `shaders/env_grid.h`;
all ray tracers, all scenes). Every env's instances and rays are offset into one world. Before (now
`env_layout="legacy"`): a 2D grid from the origin with stride 2 (extent + max range) + 1, so tier 2 (max range 1000 m)
put env 31's camera 62 km out, where float32 spacing is 4 mm: the same state rendered differently per env. Now
("compact", default): a grid centred on the origin (3D when that keeps offsets smaller; vertical stacking weighted 8x
because a floor seen at grazing angles amplifies the origin's z rounding) whose cells are the env's content box (every
drawn geom's bounding box, planes by their size, cameras, sites, lights, flex vertices and tendon points at the initial
state, symmetric about the origin in x and y, plus max(1 m, a tenth of its largest side)); every ray ends where it
leaves its env's cell. MuJoCo's infinite planes (workstream C draws them as a disc) are a disc covering the cell in the
acceleration structure and continue analytically past it, in the env's frame, with MuJoCo's plane texture mapping
(`env_grid.h` `grid_plane`, every ray kernel), so the ground still reaches the horizon. If geometry (geom
centres, dynamic objects' bounds) later nears its cell's boundary, the GPU flags it and the cell doubles before the next
frame (a warning says so; `RayTracer(env_margin=...)` sizes it up front). Same state in every env, max |env e − env 0|
depth (p99.9) in mm, legacy → compact:

| scene, envs | ray depth camera | tier 2 depth | lidar |
|---|---|---|---|
| wrapping tendons, 1024 | 0.24 (0.12) → 0.0064 (0.0045) | 7.6 (5.2) → 0.0067 (0.0048) | 0 → 0 |
| draped cloth, 1024 | 0.71 (0.26) → 0.014 (0.014) | 17.3 (8.6) → 0.015 (0.015) | |
| Go2 on a plane, 1024 | 2152 (0.21) → 0.22 (0.025) | 2154 (338) → 0.22 (0.024) | 3.7 (0.21) → 0.074 (0.020) |
| Go2 on a plane, 64 | 269 (0.059) → 0.030 (0.007) | 2153 (2.2) → 0.031 (0.008) | 0.18 (0.053) → 0.014 (0.004) |

The remaining maxima are grazing hits (floor near the horizon, tube silhouettes), where any origin rounding is
amplified; hit masks are identical. Largest offsets at 1024 envs: legacy 1.3–1.4 km (tier 2: 62 km), compact 35–78 m.
Speed is the same or better (cloth 1024 envs: depth 3.3 → 3.2 ms, tier 2 19.0 → 18.8, tier 1 13.1 → 13.3; Aero hand
tier 2 59 → 41). `env_layout="legacy"` reproduces the previous outputs byte for byte (identity check: 111 of 121
outputs identical against 471d156, the 10 others the ray-traced outputs of the two scenes with dynamic objects).

**Unchanged when absent.** Scenes without visible sites / flexes / skins / tendons render byte-identically: 87 of 92
outputs identical on six scenes (prims, textured heightfield, SO-101, Panda, Shadow hand, Go2) against the previous
integration-branch commit, every renderer and the depth camera; the five differing outputs were the Go2 lidar hitting
the lidar's own `imu` site, since fixed by leaving sites out of the sensors (`scripts/diagnostics/render_identity_check.py`;
`test_scenes_without_extras_are_byte_identical`). Go2's camera images were identical (its site is inside the body).
The per-env structures (which changed the ray-tracing shaders' slot / mesh lookup) were checked the same way against the
commit before them (471d156) with the two extras scenes of the tests added: every output of the six scenes without
dynamic objects identical, and the tier 0 / tier 1 images of the two with them; the 10 differing outputs (of 121) are
their ray-traced ones, with float-level differences (depth ≤ 0.14 mm, a different triangle of the same surface at a few
edge hits, tier 2 RGB at 0.08 % of values) from vertices now held in the env's own frame.

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

**Speed** (M4 Max, 64×64 per env, ms per frame, 30 frames back to back; `scripts/diagnostics/deformable_render_bench.py`,
`deformable_render_profile.py`). "Geoms only" is the same renderer with `visual=False` (the deformables invisible, the
previous behaviour).

Cloth scene (10×10 cloth, 12-vertex cable, 4×4×4 soft cube on a box; 1120 dynamic triangles per env, MuJoCo Warp flex):

| envs | T0 geoms only → with | T1 | ray depth camera | T2 (1 spp, 1 bounce) | vertex generation |
|---|---|---|---|---|---|
| 256 | 0.45 → 1.28 | 0.83 → 1.83 | 0.28 → 0.83 | 1.63 → 2.70 | 0.05 |
| 1024 | 1.20 → 1.92 | 2.85 → 6.56 | 0.72 → 2.88 | 6.13 → 10.26 | 0.11 |
| 4096 | 4.58 → 7.46 | 12.5 → 27.1 | 2.67 → 11.0 | (not run) | 0.57 |

The yardstick is the triangles themselves: the same surfaces baked into static mesh geoms (instanced rigid meshes, no
dynamic path) render at 1024 envs in 6.21 ms (T1) and 9.87 ms (T2), so the dynamic path costs 4–6 % over static geometry;
the rest of the increase is drawing and tracing 1120 more triangles per env. Of the 1024-env frame, the acceleration
structure work (vertex generation, 1024 refits, instance structure) is 2.4 ms. Refit per frame beats rebuilding
(1024 envs: T1 6.57 vs 10.48, depth 2.86 vs 6.73, T2 10.2 vs 14.0), hence refit with a rebuild every 32 frames.

How the ray-traced cost came down (T2, 1024 envs): one structure over all envs with the grid offset baked in 110 ms →
blocks of 12 envs 88 → one structure per (object, env) in its own frame 13.0 → one per env with a geometry per object
10.2 (static twin 9.9).
The 10:20 GPU wedge on this Mac (2026-10-12) happened while the old per-block structures, with the grid offsets baked in
(coordinates up to ~3.3 km, 258 structures at 1024 envs), were refitting in tier 1; that pattern is removed (per-env
structures in their own frame). The 18:26 one started on a GPU already saturated by an orphaned 25 M-instance build from
another job (beyond Metal's 2^24-instance limit without `extendedLimits`); the same configurations ran cleanly before
and after.

Tendon-driven hand (Menagerie Tetheria Aero hand, 20 spatial tendons of 1 mm): with MuJoCo's tessellation (28 slices, 8
rings per cap: 118 K triangles per env) at 256 envs T0 16.9 → 26.1, T1 10.5 → 38.5, depth camera 0.53 → 26.1, T2
5.7 → 31.4. These tendons are under a pixel wide; `tube_slices=8, cap_stacks=2` (8.4 K triangles per env) gives
T0 17.0 → 17.9, T1 10.6 → 13.6, depth 0.54 → 4.15, T2 5.7 → 9.5 at 256 envs and T0 66.8 → 69.4, T1 43.5 → 58.3,
depth 1.54 → 15.1, T2 26.3 → 40.5 at 1024. `tube_slices` / `cap_stacks` still force one tessellation for every tube.

**Adaptive tube tessellation (rendering only; physics never sees it).** By default each tube object (tendon, cable
edges, flex vertex spheres) gets its tessellation from its projected width for the camera or sensor that renders it
(`dynamic_mesh.tube_tessellation`): the closest distance from the camera (or, for a sensor, from any model camera or
site) at the initial state, halved as a margin. At 3 px or wider: MuJoCo's (28 slices, 8 rings per cap), so close-ups
match MuJoCo's silhouette; 0.5–3 px: 20 slices, 4 rings (mean silhouette width 0.996 of the circle's, worst orientation
0.988, whatever the pixel size); under 0.5 px: 8 slices, 2 rings. `make_lidar` / `make_depth_camera` warn when a sensor
is finer than its tracer's tessellation was chosen for (`RayTracer(tube_resolution=...)`). Tests: a close-up gets full
tessellation (coverage as before), tendons 9 m away (12–16 px, 1 px wide) 20 slices, covering every MuJoCo pixel MuJoCo's tessellation covers at that view
(all but one end pixel of one tendon, which 28 and 64 slices miss too), 40 m 8.
Aero hand at 64 px (picks 20 slices, 42 K triangles per env): 256 envs T0 16.8 → 20.2, T1 10.6 → 20.5, depth
0.53 → 10.7, T2 5.65 → 16.0; 1024 envs T0 66.2 → 79.3, T1 44.4 → 87.0, depth 1.51 → 40.6, T2 26.5 → 66.4. At 256 px,
256 envs: T0 19.4 → 23.0, T1 43.3 → 67.0, depth 3.0 → 13.5, T2 72.9 → 85.6.
