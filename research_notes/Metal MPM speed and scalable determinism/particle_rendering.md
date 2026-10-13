# Rendering MPM particles (granular media) in metalsim's own renderer

Question: how should metalsim's tier-2 path tracer (Metal hardware ray tracing) and tier-0 rasterizer draw the
particles of Newton's implicit MPM (sand, ~0.5-1 M particles per scene) so that a path-traced frame reads as sand,
and batched camera observations see the same material?

## Findings

### Granular rendering practice
- Production and research renderers draw granular media as **instanced exemplar grains**: a few grain shapes,
  randomly mixed and oriented, path traced explicitly where the grains are resolved, with coarser models
  (volumetric / diffusion) only for long multiple-scattering paths in very large aggregates. Meng et al., "Multi-scale
  modeling and rendering of granular materials", SIGGRAPH 2015 — [paper](https://cs.dartmouth.edu/wjarosz/publications/meng15granular.pdf),
  [slides](https://cs.dartmouth.edu/wjarosz/publications/meng15granular-slides.pdf); Müller et al. 2016 follow-up
  (grains as spheres with precomputed scattering) — [project page](https://cg.ivd.kit.edu/granular_media.php).
  Their explicit-path-tracing baseline is the method used here; the multi-scale speedup (12x on tens of millions of
  specular grains) targets scenes far beyond ours (opaque diffuse grains, 5 bounces).
- Newton 1.5.2 ships the simulation-side half: `SolverImplicitMPM.sample_render_grains` / `update_render_grains`
  (`newton/_src/solvers/implicit_mpm/render_grains.py`, `examples/mpm/example_mpm_grain_rendering.py`) upsample each
  particle into points sampled uniformly in its cube (side 2 x particle radius), advected each step by the particle's
  velocity gradient plus the grid velocity and projected back into the particle's deformed ellipsoid; Newton's viewer
  draws them as points of radius voxel / (3 x points per particle).

### Metal ray tracing options
- **Bounding-box primitives with an intersection function** (analytic spheres): one primitive structure of N boxes;
  needs an intersection function table and linked functions in the compute pipeline, a different intersector type
  (`intersector<instancing, triangle_data>` would gain `max_levels` / function-table arguments), and every hit runs
  shader code. Spheres also look like ball bearings, not sand.
- **Instances of small triangle meshes** in the existing instance acceleration structure: no shader-model change
  (the tracer already uses `intersector<triangle_data, instancing>`), hardware triangle tests, irregular shapes for
  free, per-instance transforms written by a compute kernel, one rebuild per frame. Chosen.
- Instance-count limit: Apple's documentation states none for `MTLInstanceAccelerationStructureDescriptor.instanceCount`
  (search 2026-10-12 found no documented cap; the only explicit maximum is `maxInstanceCount` of the indirect
  descriptor, which we do not use). **Measured** on the M4 Max (64 GB): 22.4 M grain instances + 136 scene instances
  build and trace correctly; structure 4.6 GB (~206 B per instance) plus 64 B per instance descriptor; a 1920 x 1080
  frame at 128 spp with OIDN in 8 s including the rebuild (`metalsim/render/PARTICLES.md` has the table).
- Masked-out instances (mask 0) are skipped by traversal but still occupy the structure, so culling by mask saves
  shading, not memory. True compaction would need the indirect instance descriptor (GPU-written count, per-instance
  `MTLResourceID`) or a deterministic prefix scan; not needed at these sizes.

## Design (implemented)
- `metalsim/render/particles.py` + `shaders/particles.metal`: each particle carries K render grains (Newton's cube
  sampling), each an instance of one of 4 irregular grain meshes (perturbed icospheres of unit volume-equivalent
  radius), with offset, rotation (Shoemake), scale and mesh **hashed from (particle index, grain index, seed)**: no
  state, no host round trip, a grain keeps its identity frame to frame, and two runs build identical instances.
  Positions can be a Warp array on the Metal device (bound directly).
- Path tracer: grain instances follow the scene instances in the same structure; a grain hit takes its material from
  a palette (per-grain albedo variation: mostly the sand colour, some lighter / darker / dark mineral grains, plus a
  brightness jitter) and leaves the surface with a mm-scale ray offset (scene hits keep 1 mm / 0.5 mm, so scenes
  without particles render bit for bit as before).
- Tier 0: the same instance kernel feeds one instanced draw (all grain meshes share a topology), shaded by tier 0's
  lighting with the same palette pick; segmentation id `n_slots + 1` in both tiers.
- Newton's advected grains can be fed straight in (`grains_per_particle=1`, positions = the grain array); the
  hashed-offset grains move rigidly with their particle, which is visually equivalent for sand at ~1 cm particle
  spacing (the demo) and needs nothing from the solver.

## Open
- Sub-pixel grains (real sand is 0.1-1 mm; ours are 2-3 mm, so close-ups read as coarse sand): needs more grains per
  particle near the surface, i.e. surface culling with compaction (above).
- Multi-scale transport (Meng 2015) only matters for bright, translucent or specular grains; not for opaque sand.
