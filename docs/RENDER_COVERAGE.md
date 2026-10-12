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
  - `Tier0Renderer(far=...)` makes the 10 m far plane configurable.
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
| **infinite plane** (`size="0 0"`, every locomotion floor) | ◐ | ◐ | ◐ | ◐ | ◐ | Drawn as a 10 × 10 m square (`plane_extent=5`). The ground covers only 0.369 of MuJoCo's ground pixels, so the horizon shows a visible edge. |
| hfield | ◐ | ◐ | ◐ | ◐ | ◐ | 0.85 / 0.85 / 0.85. The top surface is exact (depth error 0). MuJoCo's side skirts and base are not drawn. |
| **SDF geom** (`type="sdf"`, plugin or mesh SDF) | ❌ | ❌ | ❌ | ❌ | ❌ | 0.0 / 0.0 / 0.0. `geom_tris` returns `None` and the slot is silently skipped. |
| **sites** (group 0–2, visible in MuJoCo by default) | ❌ | ❌ | ❌ | ❌ | ❌ | 0.0 for sphere and box sites |
| **spatial tendon** with `width` (cables, wires, MuJoCo tendon-driven hands) | ❌ | ❌ | ❌ | ❌ | ❌ | 0.0 |
| **flex 2D (cloth)** | ❌ | ❌ | ❌ | ❌ | ❌ | 0.0 (277 px in MuJoCo) |
| **flex 3D (soft volume)** | ❌ | ❌ | ❌ | ❌ | ❌ | 0.0 (152 px in MuJoCo) |
| flex 1D (cable) | ❌ (code) | ❌ | ❌ | ❌ | ❌ | MuJoCo's own renderer drew 0 px for the cable under default visual flags, so it is not measurable here. No MetalSim code reads `flexvert_xpos`. |
| **skin** (bone-skinned mesh) | ❌ | ❌ | ❌ | ❌ | ❌ | 0.0 (236 px in MuJoCo) |
| **transparency** (`rgba` alpha < 1, material opacity, USD `opacity`) | ◐ | ◐ | ◐ | ◐ | ◐ | Drawn opaque. A red box behind 25 %-alpha blue glass is MuJoCo mean RGB (141, 59, 124) vs T0 (25, 25, 172) and T2 (23, 23, 147). Segmentation is 100 % glass and depth stops at the glass. |
| **geometry beyond 10 m** | ❌ | ❌ | ✅ | ✅ (range-limited) | T0: ❌ | T0/T1 far plane is fixed at 10 m in the shader. Boxes at 14 m and 38–41 m: T0/T1 0.0 / 0.0 / 0.0, T2 and RT 1.0. WIP `far=` in the main checkout fixes this. |
| geom groups ≥ 3 hidden | ✅ | ✅ | ✅ | ✅ | ✅ | Matches MuJoCo's default `geomgroup` |
| `geom_rgba` / `mat_rgba` | ✅ | ✅ | ✅ | n/a | ✅ | |
| emission | ✅ | ✅ | ✅ (not sampled as a light) | n/a | | |
| **mesh normals** (hard edges of CAD / OBJ / STL meshes) | ◐ | ◐ | ◐ | n/a | normals annotator ◐ | Smooth normals are averaged across every edge, and `mesh_normal` is ignored. On a box written as a mesh, the luminance spread inside the object is 43.7 (T0) and 43.5 (T2) vs 2.9 in MuJoCo; the primitive box gives 1.4 / 4.2 / 3.1. Every robot link mesh with sharp edges is shaded as if rounded. |
| 2D texture on box / plane / hfield / mesh with UVs | ◐ | ◐ | ◐ | n/a | | Box with checker: correlation 0.86 (T0) and 0.91 (T2) vs MuJoCo. Atlas path: no mips, so it aliases at distance (TEXTURES.md). |
| **mesh UV orientation** | ◐ | ◐ | ◐ | n/a | | Image vertically mirrored relative to MuJoCo / OBJ / glTF. On an asymmetric image, MuJoCo shows blue at the top and red at the bottom; T0 and T2 show red at the top. This is pre-existing (TEXTURES.md). The WIP native path has `flip_y=True`; the atlas path is unchanged. |
| **2D texture on cylinder / capsule** | ❌ | ❌ | ❌ | n/a | | No UVs are generated, so the geom shows one texel (the texture is not visible). Luminance std 0.0–9 vs 82–91 in MuJoCo; mean colour 6 vs 140. |
| 2D texture on sphere / ellipsoid | ◐ | ◐ | ◐ | n/a | | Different parameterisation from MuJoCo's: correlation −0.94 (T0) and −0.97 (T2), i.e. the checker is in opposite phase |
| **cube texture** (`type="cube"`) on box / sphere | ◐ | ◐ | ◐ | n/a | | The whole 6-face strip is sampled with per-face UVs: correlation 0.08 / 0.06 |
| texture roles other than RGB (normal, roughness, metallic, ORM, emissive, opacity layers) | ❌ | ❌ | ❌ | n/a | | Only `mat_texid[mjTEXROLE_RGB]` is read (code) |
| `texuniform`, `texrepeat` | ✅ | ✅ | ✅ | | | Code, plus `test_texture_pattern_parity` corr 0.996 |
| **skybox** | ◐ | ◐ | ◐ | n/a | | Replaced by its mean colour: background luminance std 0.8 (T0) and 2.8 (T2) vs 17.8 (MuJoCo gradient). T2 has a real equirectangular environment map, but only via `set_environment` or a USD DomeLight, not from an MJCF skybox. |
| directional / point / spot lights moving with bodies | ✅ | ✅ | ✅ | n/a | | `light_xpos` / `light_xdir` per world |
| more than 8 lights | ◐ | ◐ | ✅ | n/a | | `MAX_LIGHTS 8`: lights 9 and 10 leave their patches black in T0 (0.0 / 0.0). T2 lights all 10 (42.4 / 42.4). MuJoCo's own renderer also stops at 8. |
| **spot-light shadows** | ❌ | ❌ | ✅ | n/a | | Shadowed / unshadowed floor luminance under a plate: MuJoCo 0.00, T0 1.04, T1 1.04 (no shadow), T2 0.03. T0/T1 shadow only the first shadow-casting *directional* light. |
| point-light shadows | ❌ | ❌ | ✅ | n/a | | T0 1.02, T1 1.02, T2 0.035. MuJoCo draws none either (1.0), but Isaac RTX does. |
| directional shadows | ✅ (one caster) | ✅ (one caster) | ✅ | | | 0.052 / 0.049 / 0.041 vs MuJoCo 0.0 |
| image lights (`mjLIGHT_IMAGE`) | ◐ | ◐ | ◐ | | | Mapped to directional (`build_lights`, code) |
| area lights (USD Rect / Disk / Cylinder, SphereLight radius) | ❌ | ❌ | ❌ | | | The importer drops Rect, Disk and Cylinder lights. A SphereLight becomes a point light with no radius, so shadows are hard. |
| debug visuals (joints, contacts, COM, inertia, equality constraints, perturbations) | n/a | n/a | n/a | | | Not camera content in Isaac either |

### 1b. Per-world randomization (things the physics can vary per env)

| feature | T0/T1 | T2 | RT | evidence (code) |
|---|---|---|---|---|
| per-env colour (`Randomizer.colors`, event-term colour) | ✅ | ✅ | n/a | `t_colors` (N, G, 4) |
| camera pose, intrinsics, DR light, ambient, backgrounds | ✅ | ✅ (no backgrounds) | | `t_cam_spec` |
| HDR dome map per episode | ❌ | ✅ | | `Randomizer.environment` is tier 2 only |
| **per-world `geom_size`** (MuJoCo Warp batched size, scale randomization) | ◐ | ◐ | ◐ | Read only for the `tslot*` terrain boxes. Every other geom is baked at its compiled size, so a randomized object size is invisible in images while physics uses it. |
| **per-world `geom_rgba` / `mat_rgba` / `geom_matid`** written into the MuJoCo Warp model | ❌ | ❌ | | The renderer never reads them; colour DR must go through `t_colors` |
| per-world light colour / intensity / position fields of the model | ❌ | ❌ | | Only `light_xpos` / `light_xdir` are read |
| per-world `cam_fovy` / intrinsics in the model | ❌ (use `t_cam_spec`) | ❌ (`set_fovy`) | | |
| roughness / metallic randomization | ◐ (shared by all envs) | ◐ | | `Randomizer.materials` writes one table. On a legacy-mode `Tier2Renderer`, `r.mat_buf` is `None`, so the call fails; use `Tier2Renderer.set_materials`. |
| texture swap randomization (`rep.randomizer.texture`) | ❌ | ❌ | | Already listed as not done in GAPS (Replicator row) |

### 1c. Deformables, particles and other physics engines

| simulated thing | where it is simulated | renderer status | evidence |
|---|---|---|---|
| MuJoCo Warp flex cloth / cable / soft volume | `physics/deformable.py`, `tests/test_deformable.py` | ❌ in every renderer and sensor | Measured above. The vertex bodies carry no geoms, and the renderer does not read `flexvert_xpos`. |
| MetalSim XPBD cloth / rope (the default cloth and rope engine per PARITY) | `physics/deformable.py` | ❌ | Code: vertex tensors only; no renderer accepts per-frame vertices (vertex buffers and BLASes are built once) |
| PhysX-port particle cloth | `physics/physx_cloth.py`, `tests/test_physx_cloth.py` | ❌ | Code: same as above |
| Newton models (G1 XPBD, VBD cloth / soft / rods, coupled proxy) | `physics/newton_backend.py` | ❌ | Code: renderers take an `MjModel` and a MuJoCo Warp `BatchSim` (`sim.d.geom_xpos`). There is no Newton `State` → renderer path, so even rigid Newton bodies are not drawn unless an MJCF twin is posed by hand. |
| Newton MPM particles (sand, fluids, Franka pour) | `physics/mpm/` | ❌ now; **in progress elsewhere** (`metalsim-particle-render`) | Code |
| Replicator motion vectors / 3D boxes / point cloud of deformables | `replicator/isaac.py` | ❌ | State-derived from geom poses and the renderer id buffer |

### 1d. USD import (`scene/usd_to_mjcf.py`) visual coverage

| USD content | status | note |
|---|---|---|
| Cube / Sphere / Cylinder / Capsule / Plane / Mesh gprims | ✅ | |
| UsdPreviewSurface: diffuse colour / texture, roughness, metallic, ior, opacity | ◐ | Opacity is stored and then drawn opaque. Roughness, normal and occlusion textures are dropped. |
| OmniPBR: diffuse constant / texture / tint, roughness, metallic, opacity, specular level, emission intensity | ◐ | `texture_scale` / rotate / translate are ignored. Emissive colour is reduced to intensity. Normal and ORM maps are dropped. OmniGlass is opaque. |
| MaterialX | ◐ | As PARITY §2 (coat, sheen, transmission and normal maps not mapped) |
| mesh UVs | ◐ | Only face-varying `st` on all-triangle meshes. Vertex-interpolated UVs or quads give an untextured (one-texel) mesh. |
| authored mesh normals | ❌ | Ignored; recomputed smooth (see the mesh-normals row) |
| GeomSubset per-face materials | ❌ | One bound material per mesh; multi-material meshes collapse to one look |
| `displayColor` per vertex / face | ◐ | First value only |
| `visibility = invisible`, purpose `proxy` | ◐ | Not honoured (purpose `guide` → hidden group 3) |
| DistantLight, SphereLight | ◐ | Sphere → point light (radius, shaping / cone and exposure dropped; diffuse clipped to 10) |
| DomeLight (colour, texture) | ◐ | Tier 2 only (environment map) |
| RectLight, DiskLight, CylinderLight | ❌ | Dropped |
| Camera: focal length, vertical aperture | ◐ | Horizontal aperture (aspect), clipping range, orthographic projection and fisheye / f-theta are ignored |
| Points, BasisCurves, PointInstancer, TetMesh / deformable prims | ❌ | Dropped |

### 1e. Camera model (Isaac Lab `TiledCamera` / `Camera` parity)

| feature | status | note |
|---|---|---|
| pinhole intrinsics, per-env pose / intrinsics | ✅ | |
| several cameras per env in one call | ◐ | One camera per renderer instance (per-env `cam_id`); use one renderer per camera |
| depth convention | ✅ | z-depth. Background is 0 in the raw buffer and inf through `replicator/isaac.py`, as Isaac. |
| fisheye / f-theta / rolling shutter / lens distortion | ❌ | Isaac Lab `FisheyeCameraCfg`; the splat renderer WIP is pinhole only too |
| far plane | ◐ | T0/T1 fixed at 10 m (WIP fix); T2 `clip_far` |

## 2. What Isaac does for each gap, effort and priority

Priority is for camera-based learning:
- **P0**: visibly wrong or missing on common assets, or simulated state that the policy needs.
- **P1**: wrong on a class of tasks.
- **P2**: niche, or fidelity only.

Effort is engineering days for one person including tests, assuming the existing tier 0/2 structure.

| # | gap | Isaac (RTX / Kit) | fix | effort | priority |
|---|---|---|---|---|---|
| 1 | **Deformables invisible** (flex cloth / soft / cable, XPBD cloth and rope, PhysX-port cloth) | Deformable bodies and cloth render as meshes skinned to simulation nodes (RTX); they appear in RGB, depth and segmentation | Add a dynamic-mesh slot class. Per frame, a compute kernel writes vertices and normals from `flexvert_xpos` (or any (N, V, 3) tensor) into a per-env vertex buffer. The raster draws it non-instanced; T1/T2/RT refit a per-env BLAS (`MTLAccelerationStructure` refit). Flex 2D gives triangles directly, 3D gives the surface faces (`flex_shell`), 1D gives capsules or tubes. Gets one seg id per flex. | 4–6 (raster 2, BLAS refit 2, tests 1–2) | **P0**: deformable manipulation from pixels is impossible today |
| 2 | **Newton models not renderable** (rigid and deformable) | Newton backend in Isaac Lab 3.0 renders through RTX from the USD stage | A `NewtonSceneAdapter` that maps `State.body_q` + shape transforms to `geom_xpos` / `geom_xmat` buffers of an MJCF/USD twin (shapes are known at build time). Particles and cloth use gap 1 and MPM (in progress). | 2–3 rigid; deformables via #1 | **P0** for the Newton path; P1 overall |
| 3 | **Mesh normals smoothed across hard edges** | Uses authored normals, or face normals split at sharp edges | Use `mesh_normal` (MuJoCo's own per-vertex normals with its smoothing rule) or split vertices at a crease angle in `_tris_to_indexed`; read USD `normals` in the importer | 0.5–1 | **P0**: every CAD robot and object is shaded wrong (spread 43.7 vs 2.9), which is a sim-to-real appearance gap on every frame |
| 4 | **Infinite plane drawn as 10 × 10 m** | `GroundPlaneCfg`: 100 m (default) plane; the horizon is visible | Size planes to the camera far plane (or a large default) in the raster. In the RT, an infinite plane conflicts with the per-env spatial offset (`stride`), so use a separate per-env analytic plane test in the shaders, or clip to the stride. | 1–2 | **P0** for locomotion / navigation cameras, P2 for tabletop |
| 5 | **T0/T1 10 m far plane** | Camera `clipping_range` default (0.01, 1e6)/(0.1, 1e5) | **WIP in the main checkout** (`far=`). Still to do: T1's 50 m and T2's 1000 m ray ranges, the splat composite far plane, and depth precision (TEXTURES.md "Far plane") | 0.5 (land the WIP) | **P0** for outdoor / nav; P2 tabletop |
| 6 | **Transparency / opacity** | RTX: OmniGlass refraction, opacity cutout, alpha blending; depth and segmentation of transparent objects configurable | T0: sorted alpha-blend pass for alpha < 1 (order-independent transparency or per-object sort), seg / depth from opaque only or configurable. T2: `any_hit` alpha test + a dielectric BTDF. Importer: keep opacity. | 2 (T0 blend + cutout), +2 (T2 glass) | **P1**: glassware, bottles and windows are common in manipulation datasets |
| 7 | **Sites and tendons invisible** | n/a for sites (no USD equivalent); tendons become visual cables/meshes authored by the asset | Optional: draw sites (opt-in group flag) as primitives. Draw spatial tendons as per-frame capsule chains from `ten_wrapadr` / `wrap_xpos` (dynamic instances, gap 1 infrastructure). | 1 (sites), 2 (tendons) | P1 for tendon-driven hands and cable tasks; P2 for sites (MuJoCo-visual parity only) |
| 8 | **Skins invisible** | Skinned meshes (UsdSkel) render | Linear blend skinning kernel into the dynamic-mesh path (#1) | 1 on top of #1 | P2 (rare in robot models; composite cables and older flexcomps) |
| 9 | **SDF geoms invisible** | n/a (USD has meshes; SDF is a collision representation) | Draw the SDF's mesh asset (`geom_dataid` → mesh, as MuJoCo does) | 0.5 | P1: anything using SDF plugins (gears, nuts and bolts) is invisible |
| 10 | **Spot / point-light shadows in T0/T1** | All lights cast shadows in RTX | T1: ray-traced shadow for every light (the loop already exists, gate on `castshadow`). T0: cube or spot shadow maps per light, or route shadowed lights to T1. | 0.5 (T1), 3 (T0 maps) | P1: indoor / spot-lit scenes |
| 11 | **Texture mapping errors** (vertical flip on mesh UVs, no UVs on cylinder / capsule, sphere phase, cube textures) | Asset UVs used as authored; MDL projections | Flip V in the atlas path (with a regression image), MuJoCo-style generated coordinates for primitives, cube-map sampling (`texturecube`) for `type="cube"` | 1.5 | P1: textured objects and DR textures look wrong; mirrored decals and labels |
| 12 | **Per-world `geom_size` (scale DR) ignored** | Isaac scale randomization changes the rendered prim | Generalize `SizedGeoms` (already done for terrain slots) to every primitive, plus a per-world scale for meshes | 1 | P1: size randomization is invisible to the policy |
| 13 | **Per-world model colour / light fields ignored** | Replicator writes USD attributes the renderer reads | Read MuJoCo Warp's batched `geom_rgba` / `mat_rgba` / `light_*` if the event terms write them, or document `t_colors` as the only path (it already exists) | 0.5 | P2 (a workaround exists) |
| 14 | Non-RGB texture roles, normal maps, USD GeomSubsets, UV scale / rotate / translate | RTX uses all of them | Material table extension + importer | 3–5 | P2 (fidelity) |
| 15 | MJCF skybox as a mean colour | Dome texture background | Convert skybox to an equirect / cube environment (T2 has the sampler; T0 needs a background pass from it) | 1 | P2 (backgrounds are usually randomized anyway) |
| 16 | Area lights, more than 8 lights in T0/T1 | Unlimited lights, area lights with soft shadows | Raise `MAX_LIGHTS` (loop cost), area-light sampling in T2 | 0.5 / 2 | P2 |
| 17 | hfield sides / base | Terrain meshes are closed | Add skirt triangles | 0.25 | P2 |
| 18 | Fisheye / lens distortion | `FisheyeCameraCfg`, f-theta | Per-pixel ray directions in T2 and RT (trivial); T0 needs a distortion warp pass | 1–2 | P2 (P1 for wide-FOV nav cameras) |
| 19 | MPM particles | Points / mesh-from-particles (fluid isosurface) | **in progress (other agent)** | — | P0 for the MPM tasks |
| 20 | Splat composite: no segmentation for the splat background, baked lighting (no robot shadows), 10 m composite limit | Isaac Sim 5.x renders 3DGUT Gaussian scenes inside RTX | **WIP elsewhere**; seg = "background" class by design | — | P2 |

## 3. Ranked fix list

| rank | fix | effort (days) | why first |
|---|---|---|---|
| 1 | Dynamic-mesh path for flex / XPBD / PhysX-port cloth (#1), with skins (#8) and tendons (#7) on the same infrastructure | 4–6 (+1 skins, +2 tendons) | A whole simulated category, which PARITY claims at parity with Isaac's deformables, is invisible to cameras |
| 2 | Mesh normals (#3) | 0.5–1 | Wrong on every frame of every mesh robot; cheapest high-impact fix |
| 3 | Land the far-plane WIP (#5) and make the infinite plane infinite (#4) | 1.5–2.5 | Locomotion and navigation cameras see a 10 m world with a floor edge |
| 4 | Newton → renderer adapter (#2) | 2–3 | Without it the Newton backend has no camera at all |
| 5 | SDF geoms (#9) | 0.5 | Silently invisible objects |
| 6 | Texture mapping fixes (#11) | 1.5 | Mirrored and missing textures on common primitives |
| 7 | Per-world size DR (#12) | 1 | Randomized sizes invisible to the policy |
| 8 | Transparency (#6) | 2–4 | Glass / plastic objects |
| 9 | Spot / point shadows in T1 / T0 (#10) | 0.5–3 | Indoor lighting cues |
| 10 | The P2 items (#13–#18) | ~10 total | Fidelity |

MPM particle rendering (#19) is not ranked: it is being built by another agent.

## 4. Reproduce

```
scripts/gpu_run.sh render_coverage check 10 -- .venv/bin/python scripts/diagnostics/render_coverage_audit.py --out runs/render_coverage
```

- Writes `results.json` and contact sheets (`geoms.png`, `nongeom.png`, `far.png`, `textures.png`; MuJoCo | T0 | T2)
  to `--out`.
- All probe scenes are synthetic MJCF built inside the script.
- No splat scene or other external asset is used.
