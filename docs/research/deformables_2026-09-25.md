# Deformables on Metal: MuJoCo flex through MuJoCo Warp, Isaac Lab's deformable coverage, alternatives

2026-09-25. Scope: cloth, cables and soft volumes on the Apple GPU through MuJoCo Warp's flex path, as a step
toward Isaac Lab's `DeformableObject`. Everything marked **measured** was run here (M4 Max; CPU runs on the Warp CPU
device, Metal runs through `scripts/gpu_run.sh` at kind `low`); **reported** = from the cited source; **estimated** =
our judgement.

Sources: [MX] https://mujoco.readthedocs.io/en/latest/XMLreference.html (flexcomp, flex, deformable),
[MM] https://mujoco.readthedocs.io/en/latest/modeling.html (Deformable objects), [MC314]
https://github.com/google-deepmind/mujoco/blob/3.14.0/src/engine/engine_collision_driver.c and
engine_collision_primitive.c, [MCmain] the same files on `main` (2026-09-23 commits "Integrate flex collisions into
threaded mj_narrowphase", "Enable threading across mj_collideTree"), [MJW] upstream/mujoco_warp (v3.14.0 + MetalSim
patches), [ILA] https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html (DeformableObject,
DeformableObjectCfg, DeformableObjectData), [ILT] https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_deformable_object.html,
[ILM] https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sim/spawners/materials/physics_materials_cfg.html,
[ILS] https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html, [IL2004]
https://github.com/isaac-sim/IsaacLab/issues/2004 (cloth/fluid proposal), [IL5285]
https://github.com/isaac-sim/IsaacLab/issues/5285 (Newton deformable proposal), [IL3] Isaac Lab v3.0.0-beta / Beta 2
release notes (https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta, discussion #6249), [ILP] Isaac Lab
paper arXiv 2511.04831 (HTML v1), [OP] https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_beta.html.

## 1. MuJoCo's flex model

* A **flex** is "a collection of MuJoCo bodies that are connected with massless stretchable elements": 1D capsules
  (cables), 2D triangles (cloth, shells), 3D tetrahedra (soft volumes), each with a collision radius [MM].
  `flexcomp` generates the bodies (one per vertex, 3 slide joints each with `dof="full"`) and the flex [MX].
* `flexcomp type`: grid, box, cylinder, ellipsoid, square, disc, circle, mesh, gmsh, direct [MX]. `dof`: full,
  radial, trilinear (8-node hexahedral interpolation cell), quadratic (27-node); interpolated dofs "require fewer
  degrees of freedom than full flexes and can result in significantly faster simulation times" [MM].
* Stretch: either **edge equality constraints** (soft constraints on every edge: implicit, "permitting large
  timesteps") or edge springs (`<edge stiffness damping>`), or a **continuum model** (`<elasticity young poisson
  damping thickness elastic2d>`): piecewise-linear FEM with Saint Venant–Kirchhoff hyperelasticity; 2D shells add
  bending [MM]. The two cannot be combined on one flex (MuJoCo: "flex constraints and elasticity (young) cannot both
  be present", measured). For dim 2, `elastic2d` must be set ("both"/"stretch"/"bend") or `young` has no effect
  (measured: identical trajectories for young 3e4…1e6 with the default).
* Collisions: flex vs geom, self-collision (`selfcollide` none/narrow/bvh/sap/auto), `internal`, `activelayers`,
  midphase BVH per flex [MM]. Cables can also be built with the `cable` composite (plugin `mujoco.elasticity.cable`,
  twist and bending) [MM]; MuJoCo Warp rejects body plugins, so for Warp a cable is a dim-1 flex (no bending/twist
  energy; bending needs extra constraints).
* MuJoCo C caps contacts per collision pair at `mjMAXCONPAIR` = 50. For flexes [MC314] keeps, per (body, flex) or
  (flex, flex) pair, the deepest contact and then farthest-point samples 49 more (`filterFlexContacts`), including
  coincident duplicates (vertex contacts shared by adjacent triangles). 3.14's implementation swaps each chosen
  contact to the front of the array but not its `selected`/`min_dist` bookkeeping, so its choice is not a plain
  farthest-point sampling (measured: a plain FPS reproduces 0 of 83 capped plane-contact sets, an emulation with the
  swap 83/83). [MCmain] (2026-09-23) replaced it with `filterPreContacts`: plain FPS over pairs sorted by (group,
  pair type, geom, element). A 10×10 cloth lying on a plane therefore gets 50 vertex contacts, not 100, in MuJoCo C
  (both versions, measured), and penetrates (−7 mm at radius 5 mm, measured).

## 2. MuJoCo Warp's flex implementation (v3.14.0) and what differs from MuJoCo C

Kernels (all Warp, all worlds per launch): `smooth.flex` (vertex positions, edge lengths and sparse edge Jacobians,
trilinear nodes, face kinematics); `passive._flex_elasticity/_flex_bending/_flex_passive_interp/_flex_passive_bend_interp`;
`constraint._equality_flex/_equality_flexstrain` and `_efc_contact_*_flex` (contact rows with interpolated
Jacobians); `collision_flex`: per-flex AABBs, plane-vertex candidates, 1D vertex-geom candidates, 2D/3D
element-geom candidates (primitives or GJK/EPA), SAP self and flex-flex narrowphase, candidate filtering
(block-parallel FPS for self/flex-flex groups), contact write; `bvh`/`ray` for rendering and rays.

Unsupported (raises): `dof="quadratic"`, flex–SDF, flex–heightfield, flex internal collisions, flex equality with
sleeping, body/actuator/sensor plugins (so no `cable` composite); warning: bending damping on interpolated shells
[MJW io.py].

Differences from MuJoCo C found and fixed here (MuJoCo Warp fork branch `metalsim-flex`,
github.com/pulipakaa24/mujoco_warp, commits b2e9ea5, a277152, bbe19bb, 6234396, fa2ff1d; every change behind a module flag, upstream behaviour
one flag away):

| # | MuJoCo Warp v3.14 | MuJoCo C | fix (flag) | measured effect (one step from MuJoCo C's own states) |
|---|---|---|---|---|
| 1 | geom-flex contacts never capped (test asserted 100 of 100 plane contacts) | ≤ 50 per (body, flex) pair by FPS | per-(world, body, flex) group, serial exact selection (`ENABLE_GEOM_FLEX_FPS`) | cloth on plane: settled mean height bias 2.3 mm → 0.03 mm |
| 2 | FPS stopped at zero distance, dropping coincident duplicates | keeps them | allowed | box: 9 → 50 contacts kept |
| 3 | FPS tie-breaks by element id | first in C's pair order | canonical order key (plane-vertex by vertex, element by element or, for 3.14, by flex-BVH DFS rank = `Model.flex_elemorder`) | plane 83/83 sets equal |
| 4 | — | 3.14 swap quirk vs main plain FPS | `FLEX_FPS_MODE` = `c314` / `main` / `parallel`, default by installed mujoco | c314 vs C 3.14 and main vs C main both exact on plane and box |
| 5 | box–triangle ≤ 2 contacts per element, no margin tests | all 3 vertices + 8 corners with margin tests | `BOX_TRIANGLE_ALL_CONTACTS` | raw candidates 286 → 384 = C's 384 |
| 6 | element contacts of any flex after the first used the **global** element index where the Jacobian reads the local one | local index | pass local ids (bug fix) | cloth as second flex: fell through the box (0.76 m error) → 3.8e-5 m/s |
| 7 | cables (dim 1) collide as vertex spheres | capsule elements (raw sphere/capsule/box, CCD otherwise) | `CABLE_CAPSULE_ELEMENTS`; dim-1 Jacobian weights (were zero rows) | cable on box 0.38 → 2.6e-5 m/s |
| 8 | mesh–element contact point at the midpoint of the un-inflated penetration | radius-inflated CCD object | shift by half the radius (`FLEX_RADIUS_CONTACT_POS`) | positions to ≤ 3e-4 m |
| 9 | float32 depth/distance near-ties broken arbitrarily | float64 exact ties broken by order | tie tolerances (`FPS_DEPTH_TIE` 1e-8 m, `FPS_DIST_REL_TIE` 1e-6; looser values merged distances C separates: plane 83/83 → 77/83) | sphere obstacle median 3.3e-5 m/s |
| 10 | volume (dim 3) flexes: every tetrahedron collides with geoms | only active-layer elements (the flex BVH holds only those) | `FLEX_ACTIVE_LAYERS_ONLY` | trilinear soft cube on box: raw candidates 54 → 34 = C, one-step median 3.1e-4 → 6.5e-5 m/s |
| 11 | mesh–flex normal snapped to the nearest mesh face within 5 mm | EPA penetration direction | face normal only within 0.01 rad of EPA (`MESH_FLEX_FACE_NORMAL`) | cloth over a mesh box edge: normals were 0.1–0.9 rad off |

Fidelity now (CPU device, float32, against MuJoCo C float64 3.14.0 and main 3.14.1 built here; cloth 10×10 +
cable 12, CG 100/50; `scratch/deformable/fidelity.py`; **measured**):

| scene (10×10 cloth, 12-vertex cable) | vs C 3.14 (`c314`): sets equal, one-step Δqvel median / max (m/s) | free-trajectory max vertex error @100 / @200 / @400 steps | C vs C′ @200 / @400 | vs C main (`main`): sets, Δqvel median / max |
|---|---|---|---|---|
| cloth + cable onto a plane | 83/83, 1.9e-5 / 4.8e-5 | 1.2e-2 / 5.4e-3 / 7.2e-3 m | 1.3e-7 / 1.3e-7 | 82/83, 2.2e-5 / 9.8e-2 |
| cloth onto a box, cable onto the floor | 78/78, 2.1e-5 / 4.6e-5 | 5.1e-5 / 1.1e-2 / 3.6e-2 | 6.2e-7 / 1.7e-2 | 78/78, 2.0e-5 / 4.9e-5 |
| cable onto a box | 84/84, 5.4e-6 / 2.6e-5 | 3.7e-6 / 3.3e-4 / 1.1e-2 | 1.5e-7 / 4.4e-4 | 84/84, 5.3e-6 / 2.6e-5 |
| cloth as the second flex, onto a box | 88/88, 2.1e-5 / 3.9e-5 | 1.3e-2 / 3.0e-2 / 2.6e-2 | 5.1e-4 / 7.8e-3 | 88/88, 2.0e-5 / 4.9e-5 |
| cable across the cloth on a box (flex-flex) | 80/88, 2.3e-5 / 8.5e-3 | 4.7e-3 / 1.4e-2 / 1.4e-2 | 1.5e-6 / 1.5e-6 | 69/88, 2.7e-5 / 1.6e-2 |
| sphere, capsule, cylinder obstacles | 33/81, 3.3e-5 / 6.0e-2 | 1.7e-4 / 9.8e-3 / 2.7e-2 | 1.6e-2 / 1.7e-2 | 21/81, 4.8e-3 / 6.0e-2 |
| cloth + cable onto a convex mesh box | 15/88, 1.7e-2 / 2.3e-1 | 1.3e-2 / 1.5e-2 / 3.9e-3 | 2.5e-7 / 1.5e-7 | 17/88, 1.4e-2 / 2.1e-1 (before fix 11) |
| 4×4×4 trilinear soft cube (young 5e3, 1 ms) onto a box | 48/67, 6.5e-5 / 1.1e-2 | 8.2e-8 / 4.6e-4 / 5.0e-4 | 2.7e-4 / 3.1e-4 | – |

Before the fixes (same harness, upstream contact model): one-step Δqvel median 0.05–0.10 m/s, max 0.4–0.6 m/s on
the plane and box scenes, contact sets 0–1 of 78–83; cable across a box 0.38 m/s; cloth as the second flex fell
through the box. `runs/deformable/fidelity_cpu_final.log`, `…_summary.txt`.

Reading: per step the model now equals MuJoCo C to float32 level wherever contact generation is deterministic
(planes, boxes, capsules, spheres); the free trajectories still separate by millimetres to centimetres after
contact because the capped 50-of-N selection is discontinuous, so a float32 difference in a depth or distance flips
a choice. Remaining genuine differences: (a) EPA witness points on face–face mesh contacts (MuJoCo Warp's GJK/EPA
port in float32 lands elsewhere on the face; the contact point on a flat face contact is not unique), (b) flex-flex
contacts shallower than ~10 µm (float32 CCD at the activation boundary), (c) coincident candidates from adjacent
triangles whose float32 positions differ by rounding.

Why the three MuJoCo Warp flex tests failed on Metal: _to be filled from the Metal run (§6)_.

## 3. What Isaac Lab exposes for deformables

* `DeformableObject` / `DeformableObjectCfg` (prim path, spawn, `init_state` pos/rot, `collision_group`,
  `debug_vis`) with `DeformableObjectData`: `nodal_pos_w`, `nodal_vel_w`, `nodal_state_w`,
  `nodal_kinematic_target`, `default_nodal_state_w`, `sim_element_quat_w`, `collision_element_quat_w`,
  `sim_element_deform_gradient_w`, `sim_element_stress_w`, `root_pos_w`, `root_vel_w`; writers
  `write_nodal_state_to_sim`, `write_nodal_pos_to_sim`, `write_nodal_velocity_to_sim`,
  `write_nodal_kinematic_target_to_sim`, `transform_nodal_pos` [ILA].
* PhysX backend: FEM **volume** soft bodies only, "only supported in GPU simulation", two tetrahedral meshes
  (simulation mesh deformed by the solver, collision mesh matching its surface), partial kinematic control of nodes
  [ILT]. `DeformableBodyMaterialCfg`: `density` (None = solver decides), `dynamic_friction` 0.25, `youngs_modulus`
  5e7 Pa, `poissons_ratio` 0.45 (0…0.5; Isaac Sim limits it to 0.4999), `elasticity_damping` 0.005,
  `damping_scale` 1.0 [ILM]. Newer schemas (Omni Physics deformable beta) add `OmniPhysicsSurfaceDeformableSimAPI`
  / `VolumeDeformableSimAPI`; surface deformables collide through their simulation mesh only [ILS, OP].
* Cloth: "cloth and fluid simulation isn't officially supported through the Isaac Lab API yet" [IL2004]. Isaac
  Lab 3.0 adds, on the Newton backend, VBD (Vertex Block Descent) for cloth and soft bodies, a Newton
  `DeformableObject` (particle `q`/`qd`, kinematic constraints, nodal forces) and a `CoupledSolver` alternating a rigid
  solver (Featherstone or MuJoCo) with VBD per substep, one- or two-way, normal + Coulomb friction [IL5285, IL3].
* Published throughput for deformables: none. The Isaac Lab paper's performance section covers rigid and
  articulated scenes only; deformables are described qualitatively (FEM soft bodies, cloth solver exchanging
  impulses with Featherstone) [ILP].

## 4. Alternatives on Metal if the MuJoCo Warp flex path were not usable

| option | status here | fidelity to MuJoCo C | cost |
|---|---|---|---|
| MuJoCo Warp flex (this work) | runs, fixed to MuJoCo C's contact model | per-step equal (table above) | measured §6 |
| Newton VBD | fails to compile on Metal (GAPS ledger), Newton path archived | different model (VBD) | not measured |
| Newton XPBD particles / Warp's old `warp.sim` cloth | `warp.sim` removed from Warp (only `warp/examples/benchmarks/benchmark_cloth_*`, a spring cloth); Newton XPBD runs on Metal for rigid bodies | different model | not pursued (archived engine) |
| MetalSim-native XPBD cloth/cable (`metalsim.physics.deformable.XPBDSim`) | runs: same flex topology, graph-coloured distance + cross-edge bending constraints, geom contacts with friction (plane/sphere/capsule/box; meshes as OBBs), two-way coupling via `xfrc_applied`, one captured graph | **not MuJoCo C's model**: different constitutive law and contacts (§6 numbers) | measured §6 |

## 5. Plan and effort (estimated)

1. Done: research; isolated environment (`.venv-flex`, worktree `upstream/mujoco_warp-flex` on `metalsim-flex`);
   flex contact fidelity fixes 1–9 (≈ 1 day).
2. Metal: run MuJoCo Warp's flex tests on `metal:0`, read the three failures, fix in the Metal codegen / Warp fork
   (branch `metalsim-flex` of upstream/warp-innate) or MuJoCo Warp (0.5–2 days depending on cause).
3. Demo + tests on Metal: `metalsim/physics/deformable.py`, `tests/test_deformable.py` — MuJoCo C parity (free fall,
   contact trajectory, in-contact one-step), energy over 2 s, graph replay = eager; throughput at 64–4096 envs
   (0.5 day, done on CPU, Metal pending the GPU queue).
4. Cheaper faithful path (fidelity-first rule): profile the flex step on Metal; candidates: contact budget per
   world (raw candidates dominate), serial FPS vs parallel, sparse Jacobian row sizes, CG iterations (1–3 days).
5. Isaac Lab parity items (later): `DeformableObject`-style API (nodal state tensors, kinematic targets = flex
   vertex bodies with `mocap`/equality pins), tetrahedral FEM volumes (`flexcomp type="mesh"/"gmsh" dim=3` with
   `elasticity`), surface cloth with bending (`elastic2d="both"`), quadratic dofs and flex–heightfield in MuJoCo Warp
   (2–5 days each).

## 6. Metal results and throughput

_To be filled from the queued Metal runs._

## 7. Decisions

| decision | options (numbers) | chosen, why | how to switch |
|---|---|---|---|
| Contact model for flexes in MuJoCo Warp | upstream uncapped / MuJoCo C cap+selection | MuJoCo C (fidelity first): one-step error 0.4–0.6 m/s → 2e-5 m/s on plane/box scenes | `collision_flex.ENABLE_GEOM_FLEX_FPS=False` |
| Selection semantics | C 3.14 quirk / C main plain FPS / upstream parallel FPS | follow the installed mujoco (3.14 → `c314`) so the reference used by every test is matched exactly | `collision_flex.FLEX_FPS_MODE` |
| Cable contacts | vertex spheres / capsule elements | capsule elements (MuJoCo C); vertex spheres gave 0.38 m/s one-step error on a box | `CABLE_CAPSULE_ELEMENTS=False` |
| Box–triangle | first 2 / all (≤ 11) | all (C) | `BOX_TRIANGLE_ALL_CONTACTS=False` |
| Deformable engine for the demo | MuJoCo Warp flex / MetalSim XPBD | MuJoCo Warp flex (MuJoCo C's model); XPBD kept as `XPBDSim` with its measured deviation and cost | `XPBDSim` / `--backend xpbd` |
| Stretch model in the demo | edge equality / continuum `young` | edge equality: stable at 2 ms; continuum cloth needed `elastic2d` and went unstable at young 1e6 (NaN at 0.05 s) | `ClothCfg` / XML |
