# Particles in the renderer (render grains)

Simulated particles (Newton's implicit MPM sand, any particle buffer) are drawn by metalsim's own renderers as
**render grains**: each particle carries `grains_per_particle` small irregular grains, every grain an instance of one
of a few grain meshes. Tier 2 (path tracer) traces them in the same instance acceleration structure as the scene;
tier 0 (raster) draws the same instances, so batched camera observations see the same material. Research and design
choices: `research_notes/Metal MPM speed and scalable determinism/particle_rendering.md`.

## Use

```python
from metalsim.render.tier2 import Tier2Renderer
from metalsim.render.particles import GrainMaterial

r = Tier2Renderer(model, n_envs, width=1920, height=1080, material_model="omnipbr", denoise="oidn", ...)
r.set_particles(positions, radius=0.005, grains_per_particle=27)     # numpy (n_envs, P, 3) / (P, 3), or a Warp array
out = r.render_host(datas, passes=32)                                # or r.render(sim) on the GPU path
r.set_particles(new_positions)                                       # next frame: positions only
```

- `positions`: numpy `(n_envs, P, 3)` or `(P, 3)` [m] (uploaded), or a **Warp array on the Metal device** (e.g.
  Newton's `state.particle_q`), bound directly: the grain instances are written from it on the GPU every frame, no
  host copy. With Newton, call `wp.synchronize_device()` (or order the queues) before rendering, as for any buffer
  written by another queue. Non-finite positions are culled (mask 0), so envs can hold fewer particles by padding with
  NaN.
- `radius`: the particle radius (half the particle spacing; Newton's `model.particle_radius`). A particle's grains are
  spread over its cube of half side `radius` (Newton's `sample_render_grains` sampling).
- `grains_per_particle` (K), `grain_radius` (default `radius / K^(1/3)`: grains about touching), `radius_jitter`
  (relative, default 0.3), `n_meshes` (4), `grain_shape` (`grain_meshes` options: `irregularity`, `elongation`,
  `subdiv`), `seed`, `cull_z` (particles below it get no grains), `ids` (stable particle ids, `(n_envs, P)` or `(P,)`
  uint32: grains hash the id instead of the index, so a caller can drop or reorder particles between frames, e.g. draw
  only particles near the surface, without grains changing shape, place or colour).
- `material`: `GrainMaterial(palette=[(r, g, b, weight), ...], brightness_jitter, roughness, specular, metallic,
  model)`. The default is dry sand: 74 % sand colour, 14 % lighter quartz, 8 % darker, 4 % dark mineral grains, +-18 %
  brightness per grain, roughness 0.85. `GrainMaterial.uniform(rgb)` for one colour.
- Newton's own advected grains (`SolverImplicitMPM.sample_render_grains` / `update_render_grains`) can be passed as
  positions with `grains_per_particle=1`.
- `Tier0Renderer.set_particles` takes the same arguments.

Outputs: grains appear in RGB, depth and normals like any geometry; segmentation id `n_slots + 1` (slot mode), geom
`ngeom` + 1 / body `nbody` + 1 in tier 0's geom / body modes.

## How it works

| step | where | what |
|---|---|---|
| grain meshes | `particles.grain_meshes` | icospheres (80 triangles) with smooth random lobes and axis ratios, rescaled to unit volume-equivalent radius; one primitive acceleration structure each |
| grain instances | `shaders/particles.metal` `grain_instances` | one thread per grain: particle position + hashed offset in the particle cube, hashed rotation (Shoemake), scale and mesh, env offset; hashes of (particle index or id, grain index, seed) only, so a grain keeps its shape, place and colour frame to frame and two runs are bitwise equal; also writes each grain's identity key (the colour hash both tiers read) |
| structure | `particles.InstanceStructure` | scene refit (`RayTracer._encode_refit`) into the first descriptors, grains after them, one instance-structure build per frame, all in the frame's command buffer |
| path tracer | `shaders/pathtrace.metal` (`FLAG_GRAINS`) | instance ids >= the first grain instance are grains: mesh data from the grain buffers, material from the palette by grain index; rays leave a grain with a ray offset of 1 % of the grain radius (scene surfaces keep 1 mm, so scenes without particles render bit for bit as before) |
| tier 0 | `shaders/particles_tier0.metal` (compiled appended to `tier0.metal`), `particles.GrainRaster` | the same instances, one instanced draw in the shadow and geometry passes (all grain meshes share one topology; the vertex shader picks each grain's mesh), tier 0's lighting and the same palette pick |

## Validation (`tests/test_render_particles.py`)

| test | checks |
|---|---|
| grain meshes | closed, unit volume-equivalent radius, unit normals |
| single grain | spherical grain from above under a head-on directional light: coverage = projected disk (2 %), mean radiance = rho E / pi x <cos> (2 %) |
| white furnace | albedo-1 grain pile under a uniform sky returns the sky radiance (3 %): energy conserved through grain interreflections |
| instance scaling | instance count = scene + envs x P x K; structure size linear in grains |
| batched envs | per-env particles, NaN-culled env empty, identical envs identical |
| determinism | two renderers, same seed: bitwise equal images |
| stable ids | a compacted particle set with ids renders as the full set with the dropped particles culled |
| GPU input | Warp array input renders the same image as numpy |
| tier 0 | grain segmentation agrees with tier 2 (IoU > 0.9), palette colour, per-env |

RESULTS_PLACEHOLDER

## Demo: ANYmal C on MPM sand

`scripts/particle_demo/`: `sim_anymal.py` records Newton 1.5.2's `example_mpm_anymal` (its model, solvers and
walking policy; a 1.2 x 4.6 m bed, 10 cm deep, and a 0.4 m/s forward command) and `render_anymal.py` path-traces it
with the Menagerie ANYmal C meshes and textures mapped onto Newton's bodies by name, a Poly Haven HDRI whose sun is
split into an analytic sun disk (`hdr.extract_sun`: no fireflies from BSDF-sampled rays hitting a 0.3-degree
HDRI sun), decoration grains around the bed on a sand-textured ground plane, OIDN, ACES. `watchdog.py` kills a
stalled or overrunning job.

DEMO_PLACEHOLDER

## Limits and next steps

- Grains are 2-3 mm (K = 27 at 1 cm particle spacing), so close-ups read as coarse sand; finer grains need more
  grains per particle near the surface only (surface culling with compaction: the indirect instance descriptor, or a
  deterministic prefix scan). Masked grains still occupy the structure.
- Memory: ~270 B per grain (structure + descriptor); 22 M grains = 6 GB.
- Tier 1 (hybrid) draws grains with tier-0 shading and no ray-traced shadows on them; the grains are not in tier 1's
  ray-tracing structure.
- Mesh textures through the atlas are vertically mirrored (pre-existing, `docs/RENDER_COVERAGE.md` item 11); the demo
  flips the Menagerie UVs until the atlas fix lands.
