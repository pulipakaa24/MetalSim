# Cameras on Newton scenes: the Newton → renderer bridge

Date 2026-10-12. Module `metalsim/render/newton_bridge.py`. Tests: `tests/test_render_newton_bridge.py` (renderers)
and `tests/test_newton_bridge_poses.py` (geometry level, no Metal renderer needed). Measurements:
`docs/measurements/newton_render_bridge.json`, produced by `scripts/diagnostics/newton_render_bridge.py` (coverage) and
`scripts/diagnostics/newton_render_bridge_speed.py` (cost).

**Problem.** Every MetalSim renderer compiles its scene from a `mujoco.MjModel` (`scene_tables.build_scene_tables`).
Each frame, it reads the poses from a MuJoCo Warp `BatchSim` (`sim.d.geom_xpos`, ...). Nothing simulated through
Newton reached a renderer, camera sensor or annotator (RENDER_COVERAGE.md gap #2). This covers rigid shapes, meshes,
heightfields, cloth, soft bodies and particles, whether stepped by `SolverMuJoCo` (MuJoCo-via-Newton), XPBD,
Featherstone or VBD.

**Result.** `NewtonRenderSource(model)` is a "sim" object for any Newton model. Tiers 0, 1 and 2, the ray-traced
sensors (depth camera, lidar, radar), `Annotators` and `IsaacAnnotators` take it unchanged. The renderer code is not
modified, so MuJoCo-path renders are byte-identical: `git diff` on `metalsim/render/{tier0,tier2,scene_tables}.py`
and `metalsim/sensors/` is empty.

```python
from metalsim.render.newton_bridge import NewtonRenderSource, Camera, Light
src = NewtonRenderSource(model, cameras=[Camera("cam", pos=(1.5, -1.5, 1.2), lookat=(0, 0, 0.3)),
                                         Camera("wrist", pos=(0, 0, 0.05), quat=(1, 0, 0, 0), body="hand")])
r = Tier0Renderer(src.m, src.n, camera="cam", outputs=("rgb", "depth", "seg"), **src.renderer_kwargs())
src.attach_particles(r)                       # free particles, when the model has any
for _ in range(T):
    solver.step(s0, s1, control, contacts, dt); s0, s1 = s1, s0
    v = src.update(s0)                        # pose kernels (graph replay) + event signal
    r.render(src, v, sources=src.sources)     # sources: cloth / soft-body vertices (dynamic-mesh path)
    r.after(r_value)                          # as for BatchSim
```

For the MuJoCo-via-Newton path, use `NewtonRenderSource(model, twin=mj_model)`, where `mj_model` is the MJCF the
Newton model was imported from. The render then uses the MJCF's own materials, textures, lights and cameras.

## 1. Design

### Render twin

The renderers keep reading an MjModel; the bridge supplies one. There are two kinds.

**Newton twin** (`twin="newton"`, the default). An MjModel compiled once with `mujoco.MjSpec` from the Newton shapes of
one world plus the global (world -1) shapes:

| Newton shape | twin geom |
|---|---|
| sphere, box, capsule, cylinder, ellipsoid | the same MuJoCo primitive and size (Newton's capsule and cylinder axis is Z, as MuJoCo's) |
| plane | plane; Newton's full width and length → MuJoCo half sizes, 0 = infinite as in MuJoCo |
| mesh, convex mesh | a mesh asset with the scale baked in; a negative scale flips the winding |
| cone | `newton.Mesh.create_cone`, the tessellation the Newton viewer draws |
| heightfield | an hfield. `z = min_z + data (max_z - min_z)` is renormalised to MuJoCo's `[0, 1] × size_z`, and the offset is moved into the geom frame |
| `SITE` shapes | MuJoCo sites (group 4, not drawn; as in the Newton viewer and Isaac Lab), usable as lidar / radar mounts |
| Gaussian | not drawn (warning) |

Notes on the Newton twin:
- Twin geoms hang from twin bodies named after the Newton bodies (the last path component of the label). Body-mode
  segmentation, the annotators' `Semantics.from_body_names` and per-body boxes therefore carry Newton's names.
- MuJoCo's compiler recentres meshes (inertial frame). Each geom's per-frame transform is composed with the twin's
  compiled `geom_pos` / `geom_quat`, so the world-space vertices are Newton's exactly.
- `stat.center` / `stat.extent` are recomputed from the initial state of env 0. The compiled values see every geom
  at its body origin, and they set the shadow-map bounds and the ray tracer's env stride.
- Colour comes from `Model.shape_color` and opacity from `Model.shape_opacity` (Newton ≥ 1.6), else from the mesh's
  colour. Mesh UVs, textures, roughness and metallic become a twin material (2026-10-12).

**MJCF twin** (`twin=mj_model`). The MJCF the Newton model was imported from, with its geoms, materials, textures,
lights, cameras and sites:
- Each MuJoCo body is matched per world to the Newton body of the same name, or by order for unnamed bodies.
- A MuJoCo body with no Newton counterpart (Newton collapsed a fixed joint) is chained to its nearest matched
  ancestor through the fixed `body_pos` / `body_quat`. A missing body that has joints is an error.
- Newton's `body_q` is the MuJoCo body frame (the COM is separate in `body_com`). Measured: renders pixel-identical
  to the MuJoCo path, RGB mean absolute difference ≤ 0.007 / 255.

### Per-frame poses

One Warp kernel per frame kind (geoms, bodies, sites, cameras) writes `pos = body_q[b] ∘ X_local` per (env, slot). A
static slot (body -1) uses its world transform; a slot hidden in this env (`HIDDEN_BODY`) is placed 10⁶ m below the
env. A sixth kernel writes the lights, and a seventh gathers particles.
- The tables (Newton body index and local transform per env and slot) are built once on the host.
- The kernels read only `State.body_q` / `State.particle_q`. There are no host round trips per frame.
- `update(state)` replays a captured graph, one per `State` object (Newton ping-pongs two), then signals
  `src.event`. The renderers wait on that event as on `BatchSim.event`.
- `launch(state)` enqueues the kernels without a signal, so they can be recorded into the caller's own solver-step
  graph (`test_pose_kernels_recorded_in_the_solver_graph`). Call `signal()` after `wp.capture_launch`.

### Worlds

Env `e` draws world `e`'s shapes plus the global shapes. This is Isaac Lab 3.0's rule: each camera sees its own world
and world -1 (`isaaclab_newton` renderer, Newton's `SensorTiledCamera` BVH groups).
- **Homogeneous worlds** (same shapes in the same order, per-world transforms may differ; what `replicate` and
  Isaac Lab's cloner produce): the twin holds one world. Per-world shape transforms and bodies come from the tables.
  Per-world colours and scales are read every frame (RENDER_COVERAGE §3c); `src.per_env_colors` is kept but returns ones.
- **Heterogeneous worlds**: the twin holds every world's shapes, and env `e` hides the others. The cost grows with
  the number of worlds, and a warning says so (`test_heterogeneous_worlds_each_env_sees_its_own`).
- **Env frame.** Poses are written relative to an env origin, `env_origins="auto"`: each world's mean initial
  position of its bodies, particles and static shapes, minus the mean over worlds. This is exact for `replicate`
  grids and Isaac Lab's env grid. Passing an (N, 3) array or None is also accepted.
  - The renderers tile envs and the ray tracer offsets each env on a grid by `stride`. Both assume each world lies
    around its own origin, and Newton worlds are physically spaced.
  - Cameras and lights given without a body are in this env frame (Isaac Lab gives world-frame camera poses: subtract
    `env_origins`).

### Cameras, lights, sites

Newton models have no cameras or lights. `Camera` is MuJoCo's pinhole (looks along -Z, +Y up), given by `fovy` or
physically (`sensorsize`, `focal`, `principal`: Isaac's aperture and focal length). It is fixed in the env frame or
mounted on a Newton body (`body=` label, label tail or in-world index). `Light` is a MuJoCo directional or spot light.
- Defaults (Newton twin): one camera looking at (0, 0, 0.3), one directional light with shadows, and MuJoCo's
  headlight.
- `spec_hook(spec)` can add anything else before the compile (materials, more cameras). Its objects are fixed in the
  env frame, or follow their twin body.

### Cloth, soft bodies, particles

Gathered on the GPU per env, in the env frame, and drawn by the paths other workstreams own. The bridge adds no
renderer code.
- **Cloth and soft-body surfaces.** `tri_indices` over `particle_q`: cloth triangles and the boundary triangles
  `add_soft_grid` / `add_soft_mesh` create. They become one `extras.DeformableMesh` (`src.renderer_kwargs()` →
  `deformables=`), and per frame an (N, V, 3) array (`src.sources`). This is workstream A's dynamic-mesh path:
  raster, ray-tracer refit and tier 2.
- **Free particles.** Those in no triangle and no tetrahedron: an (N, P, 3) buffer bound to the particle renderer
  with `src.attach_particles(renderer)` (`set_particles`, one round grain per particle at the particle radius;
  `material=` / `grains_per_particle=` give the sand look of the MPM renderer).
- This is the split Isaac Lab 3.0's Newton renderer makes (`render_context.py`: triangle particles are not also
  drawn as spheres).
- Inactive particles (`ParticleFlags.ACTIVE` off) are hidden every frame.
- MPM (`metalsim/physics/mpm`) keeps its own particle arrays. Pass them to `set_particles` directly, as the particle
  renderer's own demo does.

## 2. Coverage (measured)

Method:
- `scripts/diagnostics/newton_render_bridge.py`: each MJCF is imported into Newton (`add_mjcf`) and replicated 2 m
  apart.
- 3 random states per world are written into Newton's `joint_q` (`mujoco_qpos_to_joint_q`, world offsets added) and
  posed with `eval_fk`.
- Each state is rendered through the bridge and compared with the same MJCF at the same qpos through the MuJoCo path
  (`render_host` / `trace_host`). Resolution 160×120. Values are the worst over states and envs.
- Coverage is per body (body-mode segmentation, ids matched by body name): the fraction of the MuJoCo-path silhouette
  that the bridge labels with the same body.

| scene (worlds, camera) | twin | T0 min body coverage | T0 fg IoU | T0 depth p99 | RT hit agreement | RT depth p99 | RGB MAD (/255) |
|---|---|---|---|---|---|---|---|
| primitives + mesh + 2-link arm (4, world cam) | Newton | 1.000 | 1.000 | 0.07 mm | 1.000 | 0.007 mm | 0.08 |
| | MJCF | 1.000 | 1.000 | 0.07 mm | 1.000 | 0.006 mm | 0.0001 |
| SO-101 (2, overview) | Newton | 1.000 | 1.000 | 0.19 mm | 1.000 | 0.006 mm | 44.8 (appearance) |
| | MJCF | 1.000 | 1.000 | 0.19 mm | 1.000 | 0.005 mm | 0.003 |
| SO-101 (2, wrist camera on a body, physical intrinsics) | Newton | 1.000 | 1.000 | 0.17 mm | 0.99995 | 0.10 mm | 63.0 (appearance) |
| | MJCF | 1.000 | 1.000 | 0.17 mm | 1.000 | 0.10 mm | 0.007 |
| Menagerie G1 (2, overview) | Newton | 1.000 | 1.000 | 0.97 mm | 1.000 | 0.004 mm | 79.6 (appearance) |
| | MJCF | 1.000 | 1.000 | 0.97 mm | 1.000 | 0.003 mm | 0.002 |
| Menagerie Panda (2, overview) | Newton | 0.986 | 1.000 | 0.66 mm | 1.000 | 0.005 mm | 93.8 (appearance) |
| | MJCF | 0.986 | 1.000 | 0.66 mm | 1.000 | 0.005 mm | 2.24 |

Notes on the table:
- **Geometry** (silhouettes, depth, which body is where) matches to float32 rounding in every case.
- **Panda 0.986.** Two adjacent links have coincident surfaces. With poses that differ only by float32 rounding, a
  few pixels flip between the two links (z-fighting). The same happens with the MJCF twin, whose tables are
  identical to the MuJoCo path's. Foreground IoU is 1.0.
- **RGB with the Newton twin** shows Newton's own appearance: `shape_color` as the Newton importer resolves it
  (palette colours where the MJCF uses material textures), no skybox, no textures. That is what the Newton viewer
  shows. Use the MJCF twin for the MJCF's appearance.

Tests (`tests/test_render_newton_bridge.py`, 15 tests, about 12 s):
- per-body coverage ≥ 0.99 and depth error < 1 mm for both twins and batched worlds with different states;
- SO-101 meshes from a world camera and a body-mounted physical camera;
- spaced worlds equal co-located worlds;
- heterogeneous worlds: each env sees only its own world and the ground;
- ray-traced depth camera and lidar (on an MJCF site imported by Newton as a `SITE` shape), tier 1 and tier 2;
- a lidar on a Newton `add_site` site;
- `IsaacAnnotators` 3-D boxes equal the MuJoCo path's;
- graph replay equals eager;
- the pose kernels recorded inside a SolverXPBD step graph;
- SolverMuJoCo stepping;
- cloth and soft-body surfaces on the dynamic-mesh path vs the same triangles as a static MuJoCo mesh: coverage ≥ 0.99,
  IoU ≥ 0.98, depth < 1 mm, raster and ray tracer;
- free particles on the particle path vs MuJoCo spheres: IoU ≥ 0.95, icosphere grain vs UV sphere.

`tests/test_newton_bridge_poses.py` (5 tests) needs no Metal renderer. It checks per-body world-space bounds of the
twin's drawn tessellation against MuJoCo's to 1e-4 m, for both twins and the primitive and SO-101 scenes. It also
checks the SolverMuJoCo-stepped state.

Environments:
- **`.venv` (Newton 1.7):** everything passes. The two SolverMuJoCo tests are skipped: Newton 1.7 there requires
  mujoco ~= 3.12, the environment has 3.14, and `SolverMuJoCo`'s contact-conversion kernel fails to compile.
- **`.venv-newton152` (Newton 1.5.2):** has no pyobjc, so no renderer can run there. The geometry-level file passes,
  including SolverMuJoCo stepping 50 steps on Metal, with the bridge reading its `body_q` (5 passed).
- The bridge itself supports both Newton lines; `shape_opacity` and `tri_color` are optional.
- **Cloth and particle tests** need workstream A's dynamic-mesh path and the particle renderer. They are skipped on
  this branch until those land. They were verified on throwaway merges of `render-fix-deformables` (cloth: passed,
  with the rest of the bridge suites: 15 passed, 2 SolverMuJoCo skips) and of `metalsim-particle-render` (particles:
  passed). Those two WIP branches
  conflict with each other in `tier0.py` / `tier2.py` / `raytrace.py`, so they were not tested together.

## 3. Cost (measured)

`scripts/diagnostics/newton_render_bridge_speed.py`, SO-101 (19 drawn geoms), M4 Max, tier 0 at 64×64 with RGB, depth
and segmentation, GPU queue `timing` job.

| worlds | `update` (pose kernels, graph replay + signal) | tier-0 render | twin + tables build |
|---|---|---|---|
| 16 | 0.037 ms | 2.4 ms | 0.23 s |
| 256 | 0.037 ms | 35.9 ms | 0.40 s |
| 1024 | 0.033 ms | 142 ms | 1.41 s |

The bridge costs under 2 % of the render at 16 worlds and about 0.02 % at 1024. The render itself is the same as for a
MuJoCo Warp scene of the same model. The device-utilisation probe read 100 % with no process at or above 1 % before and
after the run, an ioreg artefact. The queue held the GPU exclusively.

## 4. Differences from Isaac Lab 3.0

Isaac Lab 3.0 renders Newton scenes in two ways (source: `upstream/IsaacLab3`, Isaac Lab 3.0.0):
- `isaac_rtx` (the default `CameraCfg.renderer_cfg`): `NewtonManager.pre_render` writes `body_q` and `particle_q`
  into Fabric for Kit's RTX.
- `newton_warp`: Newton's `SensorTiledCamera`, a Warp BVH ray caster over the shapes with `ShapeFlags.VISIBLE`.

| aspect | Isaac Lab 3.0 (Newton backend) | MetalSim bridge |
|---|---|---|
| state hand-off | RTX: `body_q` → Fabric world matrices on the host-driven `pre_render`. Warp renderer: reads `State` directly | GPU kernels from `State` into the twin's pose buffers, graph-captured, event-ordered |
| shapes drawn | `VISIBLE` shapes. Colliders of bodies without a visual are made visible (primitives only). With a `NewtonRaycastSensor`, the shared BVH also holds collision shapes (inferred from the shared BVH mask) | `VISIBLE` (default), `"collision"` or `"all"`. Same primitive-collider fallback, extended to a world's static shapes when none is visible (an MJCF floor is a collider only) |
| per-env visibility | own world + world -1 (BVH groups; RTX scene partitions) | the same, with a per-env twin layout |
| planes | infinite, BVH-clipped at ±1000 m and by the far clip | MuJoCo's rule: infinite planes drawn at `plane_extent` (10 × 10 m), RENDER_COVERAGE gap #4, shared with the MuJoCo path |
| heightfields | meshes | MuJoCo hfields (exact top surface, GPU tile culling in tier 0) |
| SDF shapes | not supported in the Warp renderer | Newton has no SDF-only shape type; SDF-carrying meshes draw as their mesh |
| Gaussians | partly (Warp renderer) | not drawn (and out of scope: no splats) |
| cloth / soft bodies | `tri_indices` over `particle_q`, one mesh per world; the mesh refit breaks CUDA-graph capture | the same triangles through the dynamic-mesh path (raster, refit BLAS, tier 2); the gather kernel is graph-capturable |
| particles | spheres at `particle_radius` (Warp); RTX via Fabric points | particle renderer grains (round, one per particle by default) |
| colours / materials | `shape_color` (from USD `displayColor` / OmniPBR diffuse), mesh base-colour textures; Lambert + hemispheric ambient + one default light, shadows off (Warp). Full MDL in RTX | Newton twin: `shape_color` / opacity, MetalSim PBR shading, MJCF-style lights with shadows. MJCF twin: the MJCF's materials and textures. Newton mesh textures, UVs, roughness and metallic carried (2026-10-12) |
| cameras | one per world, world-frame poses, Isaac "world" / ROS conventions; Warp renderer takes only a vertical FOV from `fy` | MuJoCo pinhole (-Z forward), fixed in the env frame or body-mounted; `fovy` or physical intrinsics with principal offset |
| depth | Warp: Euclidean ray distance converted to image-plane depth; miss = 0 (RTX: inf) | z-depth; background 0 in the raw buffer, inf through `replicator/isaac.py` (as RTX) |
| segmentation | Isaac Lab remaps shape ids to label-based instance / semantic ids; particles and cloth are UNLABELLED | slot / geom / body ids of the twin; `src.shape_of_geom()` maps geoms to Newton shape indices per env; annotators use twin body names; cloth = extra object id (`ngeom + 1 + j`), particles = grain id |
| graph capture | conditional CUDA graph, except with cloth | pose and particle kernels capturable in or out of the solver graph; the renderers encode their own Metal command buffers |

## 5. Limits and follow-ups

- **Per-world scale and colour are read every frame** (twin "newton", 2026-10-12, RENDER_COVERAGE §3c): the pose kernel
  folds each world's `Model.shape_scale` (relative to the compiled, world-0 scale) into the geom frames, and
  `render_fields.geom_rgba` carries each world's `shape_color` / `shape_opacity` to the renderers' model sync. A capsule
  keeps its compiled cap shape (radius and length exact); a non-uniform scale of a recentred mesh shades with
  approximate normals. Runtime changes to `Model.shape_transform` are still not seen (build a new source). MJCF twins
  keep the MJCF's appearance.
- **Appearance of the Newton twin:** colour, opacity, and for meshes the UVs (with `texture_transform`), texture,
  roughness and metallic (an MjSpec material and texture). Everything is still drawn opaque (RENDER_COVERAGE #6).
- **MJCF twin limits:**
  - Cameras in `track` / `targetbody` modes are drawn as fixed in their body frame (warning).
  - Extra `Camera` / `Light` objects are refused; add them to the MJCF.
  - USD-imported Newton models have no MJCF; use the Newton twin. It reads the shape flags `add_usd` sets; a USD
    scene is not among the tests.
- **SolverMuJoCo** cannot run in `.venv`, a Newton 1.7 / MuJoCo Warp 3.14 mismatch outside this module. The
  MuJoCo-via-Newton path is verified at the geometry level in `.venv-newton152` and through rendering with
  `eval_fk` / SolverXPBD states in `.venv`.
- **Heterogeneous worlds** cost O(worlds × total shapes) instances.
