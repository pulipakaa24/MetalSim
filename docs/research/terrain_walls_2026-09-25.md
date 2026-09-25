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

## 5. Options built (all in `metalsim/learn/terrain.py`, `isaac_rough_terrain(collision=...)`, task flag
`G1VelocityTask(terrain_collision=...)`, CLI `--terrain_collision`)

* **hfield**: the 0.1 m heightfield of the exact grid (the surface before this work).
* **(a) boxes**: Isaac's trimesh sub-terrains as the boxes Isaac builds them from (14,080 box geoms: 1,160 per
  stairs type, 11,760 for the random grids), over the same heightfield lowered 5 cm under the lowest box top inside
  every box-built cell (the boxes tile each cell; the cell boundary lines stay at 0 under the border boxes). The
  heightfield keeps random rough and slopes.
* **(a') boxes_local**: the same boxes, but each world holds only the boxes whose footprint meets a 2.4 m square
  around its pelvis, in 96 box slots whose `geom_size` / `geom_aabb` / `geom_rbound` are per world and whose
  `geom_xpos` is written directly (`BoxWindow`, a Warp kernel run before each control step and after each substep).
  Feet stay within ~0.9 m of the pelvis, so the window holds every box a foot can touch; at most 50 boxes needed at
  any position (4,000 random positions: selection identical to a numpy reference; overflow counter 0 in every run).
  Geometry = (a); broadphase cost independent of the box count.
* **(b) meshes**: the boxes as 8-vertex / 12-triangle mesh geoms, Isaac's own `trimesh.creation.box` triangles.
  A whole-cell triangle mesh is not possible (MuJoCo collides meshes as convex hulls, section 3); the exact convex
  decomposition of Isaac's stair and grid meshes is its boxes, so (b) is (a)'s geometry through the mesh collider.
* **(c) hfield_fine**: one heightfield of Isaac's exact top surface at 0.05 or 0.025 m (from Isaac's boxes and
  Isaac's height-field triangles after the slope-threshold moves; at 0.025 m every wall of ROUGH_TERRAINS_CFG lies on
  a grid line, so the ramp is 2.5 cm wide).
* **boxes_fine**: (a) over a 0.025 m exact heightfield (also shrinks the random-rough slope-threshold walls).
* **Height scan** (`HeightScanner(surface=...)`, task `scan_surface`): "grid" = triangle interpolation of the exact
  0.1 m grid (exact at grid points; a wall between two grid points reads as a 0.1 m ramp); **"exact"** = on
  box-built cells Isaac's top surface as 0.025 m cell values (every wall is a cell boundary there, so a ray reads the
  exact top on either side of a wall; a ray exactly on a wall line reads the cell on its +x/+y side), elsewhere the
  grid interpolation (exact for Isaac's height-field meshes except at their slope-threshold walls).

## 6. Results (measured)

Throughput: `scripts/diagnostics/terrain_walls_bench.py`, Isaac's benchmark protocol (G1 rough, `rough_isaac`
rewards, 2.5 ms x 8 substeps, random actions, one captured graph per env step), synchronized rate over 110 steps,
best of 3, 4096 envs unless stated, all in one GPU slot on the MuJoCo Warp fork at a8e6485 exported clean
(`runs/terrain_walls/bench_pinned.jsonl`, `bench_pinned.log`, `bench_fine.log`; another agent was editing the fork's
working tree during this work). Off-grid error: `scripts/diagnostics/terrain_wall_fidelity.py`, 40,000 uniform points
per sub-terrain type (all 10 rows), our collision top vs Isaac's `wp.mesh_query_ray` on Isaac's own terrain mesh
(Isaac's generator run here), mean / p99 / max in mm; the analytic surface equals MuJoCo's `mj_ray` on the compiled
model to 1e-13 m (`runs/terrain_walls/fidelity.json`, `fidelity_boxes_fine.json`). Step edge:
`scripts/diagnostics/terrain_step_edge.py` (G1 foot collider, 5 kg, stubbing a 0.11 m riser of the inverted-stairs
pit at 1 m/s, and landing across its edge; MuJoCo Warp vs MuJoCo C, 0.4 s; `runs/terrain_walls/step_edge.jsonl`).
Transfer: `scripts/diagnostics/g1_rough_transfer.py --reps 8` (the protocol of PARITY §1.5 with 8 starts per cell
type: rep 0 = the recorded protocol start, 7 jittered by up to +-5 cm), levels 3 and 6, Isaac's checkpoints 500 /
1000 / 1499 and ours 500 / 1000 / 1500, fall = pelvis < 0.3 m above the local ground at step 400
(`runs/terrain_walls/transfer_reps.jsonl`, `scripts/diagnostics/terrain_walls_reps_summary.py`).

### 6.1 Geometry and cost

| option | off-grid error, stairs / inverted stairs / boxes cells (mean / p99 / max, mm) | random rough | all cells | env-steps/s at 4096 (vs hfield 29.5 K) | step edge: Warp vs C |
|---|---|---|---|---|---|
| hfield 0.1 m (before) | 13 / 168 / 227; 13 / 167 / 227; 10 / 122 / 327 | 5 / 64 / 98 | 7 / 131 / 327 | 29,547 | fails: Warp never makes a wall contact (min normal z 0.67); C's toe rides 3.6 cm up the ramp; trajectories 0.21 m apart |
| (a) boxes, all 14,080 | 0 (max 5 um) | 5 / 64 / 98 | 0.9 / 34 / 98 | out of Metal memory at 4096; 14,325 at 1024 (hfield 18,460, -22 %), 18,124 at 2048 (hfield 24,882, -27 %) | matches: wall normals in both, toe stops at the riser (+7 mm Warp, +5 mm C, soft contact), no climb; 2 mm max apart, 0.6 mm at 0.4 s |
| **(a') boxes_local** (window) | 0 (same boxes) | 5 / 64 / 98 | 0.9 / 34 / 98 | **27,772** (-6.0 %); 27,826 with the exact scan | as (a) (`test_step_edge_contacts_match_mujoco_c[boxes_local]`, Warp window vs C on all boxes) |
| (b) meshes | 0 | 5 / 64 / 98 | 0.9 / 34 / 98 | out of memory at 4096; 14,664 at 1024 | identical to (a) (same contact set to 1e-9 in C) |
| (c) hfield 0.05 m | 6 / 142 / 225; 6 / 144 / 224; 4 / 72 / 314 | 3 / 53 / 96 | 3 / 91 / 314 | 27,750 (-6.1 %, njmax 256) | fails: min normal z 0.41; 11 cm apart |
| (c) hfield 0.025 m | 3 / 111 / 220; 3 / 111 / 222; 4 / 105 / 374 | 1 / 40 / 93 | 2 / 70 / 374 | 25,806 (-12.7 %, njmax 256) | fails: min normal z 0.22; 9 cm apart |
| boxes_fine (a + 0.025 m) | 0 | 1 / 40 / 93 | 0.2 / 1.8 / 93 | out of memory at 4096 (14,080 geoms) | as (a) |

A finer heightfield does not remove the wall error where it matters: p99 stays 7-14 cm on the wall cells because a
steep ramp is still a ramp, and with the plane-contact kernel a foot that enters the ramp strip is pushed up and
over rather than stopped (the normal of a 2.5 cm-wide, 11 cm-high ramp is 77 degrees from vertical, not 90). Isaac's
walls are boxes; only boxes (or their meshes) reproduce them. Box count is the cost problem, not box collision: all
14,080 boxes add 3 x 14,080 = 42,243 NXN pairs per world, `geom_xpos` / `geom_xmat` of (4096, 14,084) and the
host-side 99 M-entry geom-pair table, and the 4096-world step runs out of Metal memory; the window keeps 96 slots.

Height scan (`tests/test_terrain.py::test_exact_height_scan_matches_isaac_raycast`, 600 random torso poses, 67,507
rays on box-built cells, vs Isaac's ray cast): "exact" max 4.8 um; "grid" mean 17 mm, p99 177 mm, max 321 mm. The
scan costs nothing measurable (hfield 29,547 grid vs 29,560 exact).

### 6.2 The decisive test: transfer of Isaac's checkpoints (falls / starts, mean x travelled)

| surface | scan | policies | pyramid stairs | inverted stairs | boxes | random rough | wall cells |
|---|---|---|---|---|---|---|---|
| hfield 0.1 m (before) | grid | Isaac's | 0 / 48, 3.76 m | **34 / 48, 0.82 m** | **13 / 48, 2.35 m** | 0 / 48, 3.14 m | 47 / 96 (49 %) |
| hfield 0.1 m (before) | grid | ours | 0 / 48, 3.39 m | 0 / 48, 2.37 m | 1 / 48, 3.02 m | 0 / 48, 3.21 m | 1 / 96 (1 %) |
| hfield 0.1 m | exact | Isaac's | 0 / 48, 3.69 m | 36 / 48, 0.81 m | 17 / 48, 2.07 m | 0 / 48, 3.12 m | 53 / 96 (55 %) |
| (c) hfield 0.05 m | grid | Isaac's | 1 / 48, 3.74 m | 39 / 48, 0.71 m | 21 / 48, 2.19 m | 0 / 48, 3.14 m | 60 / 96 (62 %) |
| (c) hfield 0.025 m | grid | Isaac's | 1 / 48, 3.84 m | 40 / 48, 0.78 m | 24 / 48, 2.00 m | 0 / 48, 3.07 m | 64 / 96 (67 %) |
| (a) boxes | grid | Isaac's | 0 / 48, 2.81 m | 22 / 48, 2.03 m | 0 / 48, 3.24 m | 0 / 48, 3.15 m | 22 / 96 (23 %) |
| (b) meshes | grid | Isaac's | 1 / 48, 2.77 m | 16 / 48, 2.18 m | 4 / 48, 3.04 m | 0 / 48, 3.12 m | 20 / 96 (21 %) |
| (a') boxes_local | grid | Isaac's | 2 / 48, 2.78 m | 17 / 48, 2.25 m | 2 / 48, 3.09 m | 0 / 48, 3.15 m | 19 / 96 (20 %) |
| **(a') boxes_local** | **exact** | **Isaac's** | 1 / 48, 2.43 m | **12 / 48, 2.84 m** | **2 / 48, 3.22 m** | 0 / 48, 3.13 m | **14 / 96 (15 %)** |
| **(a') boxes_local** | **exact** | ours | 0 / 48, 2.23 m | 2 / 48, 2.75 m | 2 / 48, 3.25 m | 0 / 48, 3.22 m | 4 / 96 (4 %) |
| boxes_fine 0.025 m | exact | Isaac's | 1 / 48, 2.45 m | 12 / 48, 2.85 m | 2 / 48, 3.12 m | 0 / 48, 3.05 m | 14 / 96 (15 %) |
| boxes_fine 0.025 m | exact | ours | 1 / 48, 2.15 m | 2 / 48, 2.89 m | 1 / 48, 3.14 m | 0 / 48, 3.24 m | 3 / 96 (3 %) |

Protocol starts only (rep 0; the 24-start numbers PARITY §1.5 quotes): hfield 7 / 24 falls for Isaac's checkpoints
(5 inverted stairs, 2 boxes; this run 6 / 24); boxes 3 / 24; boxes + exact scan 2 / 24 (0 inverted stairs, 2
boxes). In Isaac Sim itself Isaac's checkpoints fall 0 / 24.

Per Isaac checkpoint, wall cells (inverted stairs + boxes, 32 starts each), hfield/grid -> boxes_local/exact:
it500 14 + 0 -> 8 + 0; it1000 15 + 10 -> 4 + 2; **it1499 5 + 3 -> 0 + 0**.

Reading: exact wall geometry is what moves the result. It cuts Isaac's falls on the wall cells from 49 % to 15-23 %
(boxes cells 13 -> 0-4 of 48; inverted stairs 34 -> 12-22), and Isaac's distance on inverted stairs from 0.8 m to
2.0-2.8 m. The exact scan helps further on inverted stairs (17 -> 12). Every heightfield variant is worse than the
0.1 m one (a steeper ramp with plane contacts launches the stepping foot). Isaac's final policy no longer falls
anywhere on the wall cells; the earlier checkpoints still fall on inverted stairs in 12 of 32 starts. That remainder
is not terrain geometry (exact to 5 um here). Outcomes near the pit's first riser are chaotic: the full-box and
windowed-box runs, with the same geometry, flip single protocol starts, which is why the 8-start rates are the
numbers to use. The remaining physics differences are listed in PARITY §1.5 (PhysX 5 ms x 4 vs MuJoCo Warp 2.5 ms x 8,
soft contacts, PhysX friction/restitution on a triangle mesh); in the recorded trajectory Isaac's it500 moves through
the pit at a quarter of our speed (0.21 vs 0.56 m after 100 steps) and yaws away from the riser.
Our own policies (trained on the ramped heightfield) stay at 1-4 % falls on the new surface.

## 7. Decision

Default for the G1 rough task on MuJoCo Warp: **(a') boxes_local with the exact height scan** (fidelity: exact
walls, exact scan, the largest transfer improvement; 27.8 K env-steps/s, 6 % below the heightfield's 29.5 K). Kept
behind flags: `terrain_collision="hfield"` (the previous surface), `"boxes"`, `"meshes"`, `"hfield_fine"` (with
`fine_res`), `"boxes_fine"`; `scan_surface="grid"`. Newton keeps the heightfield (no per-world box window). Not
pursued: SDF / octree non-convex meshes (the boxes are already Isaac's exact decomposition), mjlab's height merging
(changes Isaac's heights), a per-world hfield (MuJoCo Warp does not batch `hfield_data`).

Caveats: renderers that draw the task model see only the lowered heightfield on box cells in boxes_local (the boxes
live in `hfield["boxes"]` and per-world slots); `hfield_fine` at 4096 envs needs njmax 256 (the task asks for 512
because `put_data` checks MuJoCo C's initial contact set, and 512 runs out of Metal memory at 4096); MuJoCo C
comparisons of a boxes_local task should build the model with `terrain_collision="boxes"` (as the step-edge test does).
