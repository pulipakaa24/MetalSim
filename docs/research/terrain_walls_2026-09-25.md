# Vertical walls on Isaac's rough terrain: collision surface options for MuJoCo Warp

2026-09-25. Scope: make the collision surface of Isaac-Velocity-Rough-G1-v0's terrain faithful to Isaac's on the
vertical-wall sub-terrains (pyramid stairs, inverted pyramid stairs, random-grid boxes), where our 0.1 m heightfield
turned walls into 0.1 m ramps (PARITY §1.2 terrain row, §1.5 "Rough terrain, like for like", GAPS "Vertical-wall
terrain"). **measured** = run here (M4 Max; GPU work through `scripts/gpu_run.sh`); **reported** = from the cited
source; **estimated** = our judgement.

Sources: [IL-mesh] https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py
(L100-146, L219-244 stairs; L305-366 random grid), [IL-gen] .../terrains/terrain_generator.py (L165-167, L382),
[IL-imp] .../terrains/terrain_importer.py (L226-253), [IL-utils] .../terrains/utils.py (L60-103),
[IL-hf] .../terrains/height_field/utils.py (L60-66, L130-150), [IL-rough] .../terrains/config/rough.py (L12-41);
local copies in `assets/isaac/terrains/`. [MJ-hf] https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-hfield,
[MJ-coll] https://mujoco.readthedocs.io/en/latest/computation/index.html#collision, [MJW] `upstream/mujoco_warp`
(v3.14.0 + MetalSim patches; upstream main 7a060f6 for the table), [MJW-326] https://github.com/google-deepmind/mujoco_warp/issues/326,
[MJW-1083] .../issues/1083, [mjlab] https://github.com/mujocolab/mjlab (main c2e1e06), [MP] https://github.com/google-deepmind/mujoco_playground
(4057c14), [LG] https://github.com/leggedrobotics/legged_gym (8fa29ac).

## 1. Research plan (what had to be known before building anything)

1. What exactly Isaac collides with on these cells (geometry and collider type), so "faithful" has a definition.
2. What MuJoCo can represent exactly (hfield, box, mesh, SDF), and what each costs in MuJoCo Warp (narrowphase
   path, broadphase scaling with static geoms, contact capacity, the fork's hfield plane-contact patch).
3. What the MuJoCo-based locomotion stacks (mjlab, MuJoCo Playground) do for stairs, and what legged_gym did
   (Isaac Lab's terrain code descends from it).
4. Then measure every option on the same four axes: off-grid height error against Isaac's own mesh, G1 rough env-step
   throughput at 4096 envs, contact behaviour at a step edge against MuJoCo C, and the decisive test: cross-simulator
   transfer of Isaac's checkpoints on the wall cells.

## 2. Isaac Lab: the terrain is one triangle mesh built from boxes (reported)

* Stairs (`pyramid_stairs_terrain`, `inverted_pyramid_stairs_terrain`) are `trimesh.creation.box` rings, four boxes
  per step plus the centre platform and a border [IL-mesh]. `random_grid_terrain` tiles the 8 vertices / 12 faces of
  a template box with torch and raises only the top vertices by the noise, one `trimesh.Trimesh` [IL-mesh]; the
  geometry is the same union of axis-aligned boxes.
* Each sub-terrain's meshes are concatenated, then all sub-terrains and the border are concatenated into a single
  terrain mesh [IL-gen], imported by `create_prim_from_mesh` as one USD `Mesh` prim with only
  `UsdPhysics.CollisionAPI` applied [IL-utils]: no `MeshCollisionAPI`, so the approximation is USD's default "none",
  which PhysX treats as a triangle-mesh collider (an inference from the USD defaults; Isaac Lab does not state it).
  No convex decomposition, no SDF anywhere in `terrains/`.
* Height-field sub-terrains (random rough, slopes) go through `convert_height_field_to_mesh` with
  `slope_threshold=0.75`: vertices whose height jump exceeds the threshold are moved one cell, which turns steep
  pixels into vertical walls [IL-hf]; this is legged_gym's `convert_heightfield_to_trimesh` ported [LG]
  (`mesh_type='trimesh'` default, "slopes above this threshold will be corrected to vertical surfaces").
* Consequence: Isaac's walls are exactly vertical, riser tops are sharp box edges. Our port already reproduces the
  boxes bit-for-bit (`metalsim/learn/isaac_terrain.py`, `tests/test_terrain.py::test_isaac_sub_terrain_exact`).

## 3. MuJoCo and MuJoCo Warp (reported, checked in the fork's source)

* **Heightfield**: a union of triangular prisms down to the base; collisions select the prism sub-grid under the
  geom's bounding box and run the convex collider per prism, at most 50 contacts per pair (mjMAXCONPAIR) [MJ-hf].
  A prism's top between grid points is a plane through three grid heights, so a wall is a ramp one cell wide at any
  resolution. Both MuJoCo C and MuJoCo Warp split each cell along (c, r)-(c+1, r+1), Isaac's diagonal (verified
  here: our analytic surface equals `mj_ray` to 1e-13 m).
* **Meshes collide as their convex hulls** (qhull); a non-convex object must be decomposed into convex geoms
  [MJ-coll]. So "one mesh per stair cell" is not an option in MuJoCo: the exact convex decomposition of Isaac's stairs
  is Isaac's own boxes. The only non-convex exception is SDF geoms (plugins; MuJoCo Warp has an SDF path in
  `collision_sdf.py` with octree volumes), with their own cost and accuracy limits; not pursued because the boxes are
  already an exact decomposition.
* **MuJoCo Warp narrowphase** [MJW `collision_driver.py` table, `collision_convex.py`]: HFIELD-MESH goes through
  `ccd_hfield_kernel` (one thread per pair walking the prism sub-grid serially). In the MetalSim fork
  (`HFIELD_PLANE_CONTACTS`, PARITY §1.6) each prism gives a contact against its top-triangle plane: a steep ramp
  triangle gives a contact normal tilted by the ramp angle, and a foot that crosses the ramp strip is pushed up onto
  the upper tread. BOX-MESH and MESH-MESH are CONVEX with MULTICCD (up to 4 contacts per pair), the same algorithm
  family as MuJoCo C. HFIELD pairs get one contact per prism in the fork (upstream: at most 1 per pair under MULTICCD,
  [MJW] `io.py` L637-668 warning).
* **Broadphase cost** [MJW `io.py` L533-620]: all geom pairs are pre-filtered on the host (contype/conaffinity,
  same weld body, parent-child); static world geoms never pair with each other, so the terrain adds
  (robot colliders x terrain geoms) pairs. NXN is used below 250 k filtered pairs and launches
  `dim = (nworld, pairs)`: cost grows linearly with the number of terrain geoms per world. [MJW-326] reports NXN on
  filtered pairs beating SAP for a terrain scene. G1 here has 3 colliders (two foot meshes, torso), so 14,080 boxes
  give 42,243 pairs, NXN.
* **Batched model fields** (`geom_pos`, `geom_size`, `geom_aabb`) can be per world, `hfield_data` cannot [MJW
  `types.py`]; a per-world local window of boxes is possible but not needed (section 5).

## 4. What mjlab and MuJoCo Playground do (reported)

* **mjlab** (MuJoCo Warp, Isaac Lab's terrain generator ported): stairs, inverted stairs and random grids are native
  **box geoms** (`BoxPyramidStairsTerrainCfg`, `BoxInvertedPyramidStairsTerrainCfg`, `BoxRandomGridTerrainCfg` in
  `src/mjlab/terrains/primitive_terrains.py`), slopes and random rough are **hfields**
  (`heightfield_terrains.py`). All sub-terrains share one static `terrain` body. `BoxRandomGridTerrainCfg` has an
  optional `merge_similar_heights` (height quantisation 0.05 m, greedy rectangles) to cut the geom count, off by
  default (it changes the heights). Its G1 rough config raises `nconmax` to 70 and `ccd_iterations` to 500.
* **MuJoCo Playground**: rough terrain is a single 256 x 256 PNG heightfield for G1 / Go1 / Berkeley Humanoid /
  T1; the Go1 stairs scene is hand-written box geoms on a plane.
* So the established MuJoCo Warp practice for Isaac-style stairs is option (a) below.

## 5. Options and measurements

Filled in from `runs/terrain_walls/` (see the results section below).
