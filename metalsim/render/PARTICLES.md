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
| structure | `particles.InstanceStructure` | default `"two_level"`: the grains are their own instance structures (chunks of at most 2^24 - 1 grains, UserID descriptors with user id 0x80000000 \| grain index), each chunk one identity instance of the scene's top-level structure; `"flat"`: scene instances then one top-level instance per grain. Rebuilt every frame in the frame's command buffer |
| path tracer | `shaders/pathtrace.metal` (`FLAG_GRAINS`) | instance ids >= the first grain instance are grains: mesh data from the grain buffers, material from the palette by grain index; rays leave a grain with a ray offset of 1 % of the grain radius (scene surfaces keep 1 mm, so scenes without particles render bit for bit as before) |
| tier 0 | `shaders/particles_tier0.metal` (compiled appended to `tier0.metal`), `particles.GrainRaster` | the same instances, one instanced draw in the shadow and geometry passes (all grain meshes share one topology; the vertex shader picks each grain's mesh), tier 0's lighting and the same palette pick |

## Acceleration-structure layout and Metal's limits

Metal supports 2^24 instances, 2^28 primitives and 2^24 geometries per acceleration structure; past that a structure
needs `MTLAccelerationStructureUsageExtendedLimits` (up to 2^30) and the `extended_limits` tag on every intersector
that traverses it (`metalsim/render/as_limits.py`). Before 2026-10-13 the grains were one flat top-level structure
of 25.1 M instances built without the flag, outside the supported range. On macOS 26, with the demo's four ground
planes, that build never finished: still at 100 % utilisation after 900 s (`runs/stall_bisect/`), the same build with
one plane took 1 s. The GPU work outlived the killed process, and the Mac rebooted after each such run. The 14-inch
(macOS 27) built and rendered it, so it rendered by luck.

Now:
- `as_limits.check` refuses any structure past the standard limits unless the caller has an extended-limits path,
  and past 2^30 always; `RayTracer` checks every mesh and its instance structure, `particles.py` its grain meshes
  and grain structures.
- Two-level (default) splits the grains into chunks of at most 2^24 - 1, so every structure stays under the standard
  limits whatever the grain count. Metal reports only the innermost instance of a nested hit (`instance_count` stays 1
  and `instance_id[0]` / `user_instance_id[0]` are the innermost instance's: `scripts/particle_demo/multilevel_probe2.py`
  on macOS 26), so grains are told apart by their UserID descriptors' user id (0x80000000 | grain index; the scene's
  Default descriptors read 0).
- Flat past 2^24 instances is built with extended limits and traced with the `extended_limits` intersector.

Measured on this Mac (macOS 26), 4 ground planes, 6.28 M grains: two-level build 121 ms (rebuild 77 ms) vs flat 117 ms
(79 ms); build + one 256x144 trace pass 82 ms vs 84 ms. A 480x270, 64 spp, OIDN frame differs from flat in 5.7 % of
pixels by at most 11/255 (mean relative HDR difference 3.5e-4: path-tracer and denoiser noise from ulp-level
differences in the nested transforms). This Mac refuses builds past 6.3 M grains in `stall_bisect.sh` without
`ALLOW_LARGE=1`.

Correctness validation on the 14-inch M4 Max (macOS 27.0.1, origin/main 94b75be, 2026-10-13), the demo scene with 4
ground planes, each step alone with the GPU back to idle (no ownerless utilisation, recoveryCount 0) between steps. The
times are from that validation run, not benchmark figures (timing measurements are taken on the 16-inch):

| grains | structures (instances) | build (rebuild), validation run | 1080p frame, 128 spp, OIDN, validation run | Metal allocated |
|---|---|---|---|---|
| 6.28 M | top 55, grains 6,280,443 | 175 ms (129 ms) | 8.3 s | 5.5 GB |
| 12.56 M | top 55, grains 12,560,859 | 420 ms (348 ms) | 9.5 s | 7.4 GB |
| 25.12 M | top 56, grains 16,777,215 + 8,344,476 | 861 ms (711 ms) | 11.5 s | 11.1 GB |

No structure used extended limits, and every build and frame completed. The full 240-frame clip (8 s, 30 fps, 1080p,
128 spp, OIDN) also rendered to the end. Its frames match the 2026-10-12 flat-layout clip within about 1/255 mean (sky,
ground-plane and sand means equal; 2-3.5 % of pixels differ by more than 10/255, texture detail after the atlas fix).

**Benchmark on the 16-inch M4 Max** (macOS 26.7.1, the machine where the flat 25 M build stalled; origin/main 44dd48c,
Warp 95f6f121, MuJoCo Warp f1b35e2; 2026-10-13). The same scene with 4 ground planes, two-level grains. Each step ran
alone as an exclusive timing job through `scripts/gpu_run.sh` under a watchdog (60 s / 120 s without output). Before
and after every step: device, renderer and tiler utilisation 0-2 %, recoveryCount 0, no process stuck exiting.

| grains | structures (instances) | upload + build (rebuild) | build + one 256x144 1-spp pass | 1080p frame, 128 spp, OIDN | Metal allocated |
|---|---|---|---|---|---|
| 6.28 M | top 55, grains 6,280,443 | 116 ms (77 ms) | 81 ms | - | 2.9 GB |
| 12.56 M | top 55, grains 12,560,859 | 296 ms (226 ms) | 233 ms | - | 5.0 GB |
| 25.12 M | top 56, grains 16,777,215 + 8,344,476 | 654 ms (508 ms) | 511 ms | 5.1 s | 10.5 GB |

The 25.12 M build that never finished in the flat layout (900 s, 2026-10-12) completes in 654 ms. No structure used
extended limits. The 1080p frame time includes loading the frame's particles, the structure rebuild, 32 passes of
4 spp, OIDN and the PNG. Logs: `runs/sand25m_16in/` in the working tree (not committed).

## Validation (`tests/test_render_particles.py`)

| test | checks |
|---|---|
| grain meshes | closed, unit volume-equivalent radius, unit normals |
| single grain | spherical grain from above under a head-on directional light: coverage = projected disk (2 %), mean radiance = rho E / pi x <cos> (2 %) |
| white furnace | albedo-1 grain pile under a uniform sky returns the sky radiance (3 %): energy conserved through grain interreflections |
| instance scaling | instance count = scene + envs x P x K; structure size linear in grains |
| batched envs | per-env particles, NaN-culled env empty, identical envs identical |
| determinism | two renderers, same seed: bitwise equal images |
| two-level vs flat | the same colour (1e-4), depth and segmentation with one grain structure and with 11 chunks |
| structure limits | extended-limits usage above 2^24 instances and not at it; refusals without an extended path / past 2^30 (CPU) |
| stable ids | a compacted particle set with ids renders as the full set with the dropped particles culled |
| GPU input | Warp array input renders the same image as numpy |
| tier 0 | grain segmentation agrees with tier 2 (IoU > 0.9), palette colour, per-env |

Measured (M4 Max, 2026-10-12): instance scaling in a 64 x 64 frame, 2 envs: 2,000 grains 0.4 MiB / 3 ms,
160,000 grains 34 MiB / 19 ms, 1.6 M grains 338 MiB / 105 ms per frame (structure build included). At demo scale:
25.1 M grains, a 5.2 GB instance structure; no instance-count limit was hit (Apple documents none).

## Demo: ANYmal C on MPM sand

`scripts/particle_demo/`: `sim_anymal.py` records Newton 1.5.2's `example_mpm_anymal` (its model, solvers and
walking policy; a 1.2 x 4.6 m bed, 10 cm deep, and a 0.4 m/s forward command) and `render_anymal.py` path-traces it
with the Menagerie ANYmal C meshes and textures mapped onto Newton's bodies by name, a Poly Haven HDRI whose sun is
split into an analytic sun disk (`hdr.extract_sun`: no fireflies from BSDF-sampled rays hitting a 0.3-degree
HDRI sun), decoration grains around the bed on a sand-textured ground plane, OIDN, ACES. `watchdog.py` kills a
stalled or overrunning job.

Result (2026-10-12, 14-inch M4 Max): 669,372 sand particles (46 ms per simulated frame) plus 261,061 decoration
particles, 27 grains each = 25.1 M grain instances; 1920 x 1080, 128 spp (32 passes x 4), 5 bounces, OIDN: 10.4 s per
frame median (12.1 s max), the 240-frame clip in 43 minutes; stills at 256 spp 16-21 s. Credits: Menagerie
`anybotics_anymal_c` (BSD-3), Poly Haven `goegap` HDRI and `coast_sand_01` texture (CC0), Newton 1.5.2
`example_mpm_anymal` and its walking policy (Apache-2.0).

## Limits and next steps

- Grains are 2-3 mm (K = 27 at 1 cm particle spacing), so close-ups read as coarse sand; finer grains need more
  grains per particle near the surface only (surface culling with compaction: the indirect instance descriptor, or a
  deterministic prefix scan). Masked grains still occupy the structure.
- Memory: ~270 B per grain (structure + descriptor); 22 M grains = 6 GB.
- Tier 1 (hybrid) draws grains with tier-0 shading and no ray-traced shadows on them; the grains are not in tier 1's
  ray-tracing structure.
- The 2026-10-12 demo clip was rendered before the atlas fix (RENDER_COVERAGE item 11) with `--flip-mesh-v 1`; on
  the current renderer the default (0) is right.
