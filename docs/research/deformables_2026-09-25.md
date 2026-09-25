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
cable 12, CG 100/50; `scripts/diagnostics/deformable/fidelity.py` (MuJoCo C main: a python built from mujoco main, `MUJOCO_MAIN_PYTHON`); **measured**):

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

Why MuJoCo Warp's flex tests failed on Metal (measured, `runs/deformable/metal_run1.log`): the two flex failures
(`FlexConstraintTest.test_constraint_parity2/3`, the 3×3 cloth with edge equality, 1 and 2 worlds; the third of
the "3 failures" is the upstream `io_test::test_put_data_nefc_zero_dense`, which also fails on CPU) are neither a
compile error, a missing atomic nor precision: the 16 `efc_pos` values equal MuJoCo C's to 1.2e-8 **after sorting**,
but rows 4–7 and 8–11 come out swapped. `_equality_flex` takes each edge's row from an atomic counter, so the row
order is the device's thread order (on CUDA it happens to be edge order; Metal dispatches the 3D launch's threads in
another order), and the test compares rows in order. The rope (4 rows) and the trilinear volume (strain rows) pass.
Fix (fork commit 9c6e09f): a reserve kernel takes one block of rows per world and every edge writes at its
(equality, edge) offset, which is MuJoCo C's order on every device (and makes the Newton solver's block order
deterministic). Everything else in the flex suite passed on Metal (233 of 237 in that run; the other two failures
were the mesh-normal change of fix 11 on Metal, §6).

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
| MetalSim-native XPBD cloth/cable (`metalsim.physics.deformable.XPBDSim`) | runs: same flex topology, graph-coloured distance + cross-edge bending constraints, geom contacts with friction (plane/sphere/capsule/box; meshes as OBBs), two-way coupling via `xfrc_applied`, one captured graph | **not MuJoCo C's model** (measured, box scene, same start as §2): max vertex error 5.4 cm already at step 50 (it removes the initial 2 mm constraint violation at once where MuJoCo's soft edge constraints relax over solref's 20 ms), 5.5–15 cm at rest for 10–40 substeps, cloth mean height +0.9 to +2.0 cm; rejected on fidelity | measured §6 |

## 5. Plan and effort (estimated at the start; status in brackets)

1. Done: research; isolated environment (`.venv-flex`, worktree `upstream/mujoco_warp-flex` on `metalsim-flex`);
   flex contact fidelity fixes 1–11 (≈ 1 day) [done].
2. Metal: run MuJoCo Warp's flex tests on `metal:0`, read the three failures, fix in the Metal codegen / Warp fork
   (branch `metalsim-flex` of upstream/warp-innate) or MuJoCo Warp (0.5–2 days depending on cause) [done: row
   order + capture + mesh normal; 237/237 on Metal].
3. Demo + tests on Metal: `metalsim/physics/deformable.py`, `tests/test_deformable.py` — MuJoCo C parity (free fall,
   contact trajectory, in-contact one-step), energy over 2 s, graph replay = eager; throughput at 64–4096 envs
   (0.5 day) [done, 10/10 on Metal].
4. Cheaper faithful path (fidelity-first rule): profile the flex step on Metal; candidates: contact budget per
   world (raw candidates dominate), serial FPS vs parallel, sparse Jacobian row sizes, CG iterations (1–3 days).
5. Isaac Lab parity items (later): `DeformableObject`-style API (nodal state tensors, kinematic targets = flex
   vertex bodies with `mocap`/equality pins), tetrahedral FEM volumes (`flexcomp type="mesh"/"gmsh" dim=3` with
   `elasticity`), surface cloth with bending (`elastic2d="both"`), quadratic dofs and flex–heightfield in MuJoCo Warp
   (2–5 days each).

## 6. Metal results and throughput (measured, `runs/deformable/metal_run*.log`, `metal_final.log`)

Environment: `.venv-flex` (git-ignored) with the MuJoCo Warp worktree `upstream/mujoco_warp-flex` (branch
`metalsim-flex`, fork commits b2e9ea5…dfa5d30) and the Warp worktree `upstream/warp-innate-flex` (branch
`metalsim-flex`, 9dcb140), both pushed to the `fork` remotes. All GPU work ran through `scripts/gpu_run.sh` at kind `low`.

Correctness on `metal:0`:

* MuJoCo Warp flex tests: 237 passed, 0 failed (before: `test_constraint_parity2/3` failed, §2). Full MuJoCo Warp
  suite: 1450 passed, 2 failed, 39 skipped — the two are not flex: `io_test::test_put_data_nefc_zero_dense`
  (known upstream) and `collision_driver_test::test_hfield_maxconpair`, which fails identically on the CPU and on the
  unmodified `metalsim` branch (the fork's heightfield patch).
* Two Metal-only problems met on the way: (1) graph capture of a flex step failed because Warp's Metal
  `segmented_sort_pairs` validated its segments with a synchronous `.numpy()` (Warp fork 9dcb140); (2) on Metal the
  GJK direction of a touching tetrahedron apex (distance 1e-6 m) was 0.127 rad off the face normal (CPU 1.7e-3):
  the EPA/GJK normal is now used only for witness separations above 0.1 mm (dfa5d30).
* `tests/test_deformable.py`: 10/10 on Metal — free fall equals MuJoCo C (< 1e-5 m, 60 steps, 4 worlds), contact
  trajectory equals MuJoCo C to 1 mm over 100 steps (landing on the box), one-step in-contact parity (contact sets
  equal apart from < 10 µm flex-flex contacts; velocity error median < 1e-4 m/s, max < 0.05), energy never above
  its initial value over 2 s in 64 randomized worlds with no contact/constraint overflow, graph replay = eager
  (free fall < 1e-6 m; through contact graph-vs-eager 3.3e-2 m vs eager-vs-eager 3.1e-2 m after 150 steps, i.e.
  the atomic contact order, not the graph), XPBD Metal = CPU, XPBD energy/strain/penetration, XPBD drape vs MuJoCo
  Warp, G1 scenes finite.
* Fidelity harness on Metal (c314 mode vs MuJoCo C 3.14): identical to the CPU table for plane, box, cable, second
  flex, flex-flex, primitives and the soft cube (e.g. box 78/78 sets, one-step Δqvel median 2.2e-5 / max 4.8e-5 m/s);
  **the convex-mesh scene is worse on Metal** (Δqvel median 3.7e-2, max 0.52 m/s; mean height −3 cm after 0.8 s),
  because Metal's float32 GJK/EPA witness points on mesh face contacts differ from the CPU's (open item).

Throughput (physics step only, graph replay, 10 steps per synchronization, `python -m metalsim.physics.deformable`):

| scene | DOFs | 64 | 256 | 1024 | 2048 | 4096 envs |
|---|---|---|---|---|---|---|
| MuJoCo Warp flex: 10×10 cloth + 12-vertex cable onto a box (dt 2 ms, CG 20/10) | 336 | 6.0K | 17.5K | 36.5K | 49.2K | **54.1K** env-steps/s |
| same, Warp's host sorts (`--flex-flags FLEX_DEVICE_SORT=0`) | 336 | – | 9.1K | 11.4K | – | – |
| MuJoCo Warp flex: 20×20 cloth + 24 cable | 1272 | – | 8.6K | 10.0K | – | 9.6K |
| MuJoCo Warp flex: Isaac's G1 + 12×12 cloth + 16 cable (dt 2.5 ms, CG 30/10) | 523 | – | 15.2K | 19.1K | 20.0K | (memory) |
| MetalSim XPBD, same box scene topology, 10 substeps (**not** MuJoCo C's model) | – | 129K | 491K | 1.36M | 1.93M | 2.35M |

Cost of the faithful path at 1024 worlds (graph replay, box scene): full step 26.7 ms; contacts off 9.7 ms;
constraints off 9.4 ms; CG 5 iterations 20.7 ms. Eager stage timings: flex collision ~9 ms (element–geom detection
~4 ms, contact filter ~4 ms, flex-flex SAP + CCD ~4 ms, excluding the per-call workspace allocation that the graph
does once), contact solve the rest. The device-side sort/scan (fork 361f11f) was worth 2–3.6× (host sorts: a
segmented sort of 4096×173 SAP keys took 172 ms on the CPU per step). For scale: the rigid G1 alone runs at 68.6K
physics steps/s at 4096 envs (STATUS); Isaac Lab publishes no deformable throughput.

## 7. Decisions

| decision | options (numbers) | chosen, why | how to switch |
|---|---|---|---|
| Contact model for flexes in MuJoCo Warp | upstream uncapped / MuJoCo C cap+selection | MuJoCo C (fidelity first): one-step error 0.4–0.6 m/s → 2e-5 m/s on plane/box scenes | `collision_flex.ENABLE_GEOM_FLEX_FPS=False` |
| Selection semantics | C 3.14 quirk / C main plain FPS / upstream parallel FPS | follow the installed mujoco (3.14 → `c314`) so the reference used by every test is matched exactly | `collision_flex.FLEX_FPS_MODE` |
| Cable contacts | vertex spheres / capsule elements | capsule elements (MuJoCo C); vertex spheres gave 0.38 m/s one-step error on a box | `CABLE_CAPSULE_ELEMENTS=False` |
| Box–triangle | first 2 / all (≤ 11) | all (C) | `BOX_TRIANGLE_ALL_CONTACTS=False` |
| Deformable engine | MuJoCo Warp flex (54.1K env-steps/s at 4096, per-step equal to MuJoCo C) / MetalSim XPBD (2.35M, 43× faster, but 5–15 cm vertex and +0.9–2 cm mean-height deviation from MuJoCo C) | MuJoCo Warp flex: fidelity first; XPBD kept in `metalsim.physics.deformable.XPBDSim` with its tests and numbers | `XPBDSim(...)`, `--backend xpbd` |
| Sorts in the flex filter/SAP on Metal | Warp's host utilities (11.4K env-steps/s at 1024) / device bitonic + scan (36.5–41.5K) | device (same selections, verified by the fidelity harness) | `collision_flex.FLEX_DEVICE_SORT=False` |
| Mesh–flex normal | face normal (upstream) / EPA direction / EPA only for separations > 0.1 mm | the last: C's direction at real penetrations, robust at touching contacts on Metal | `MESH_FLEX_FACE_NORMAL`, `MESH_FLEX_NORMAL_MIN_SEP` |
| Flex equality row order | atomic counter (device-dependent) / reserved block in C's order | reserved block (deterministic; fixes the Metal test failures) | – (bug fix) |
| Stretch model in the demo | edge equality / continuum `young` | edge equality: stable at 2 ms; continuum cloth needed `elastic2d` and went unstable at young 1e6 (NaN at 0.05 s) | `ClothCfg` / XML |

## 8. What remains against Isaac Lab's deformable features

| Isaac Lab | MetalSim now | gap / next step |
|---|---|---|
| `DeformableObject` (PhysX FEM volume, tetrahedral sim + collision meshes) | flex volumes through MuJoCo Warp: `SoftCfg` soft cube (`dof="trilinear"` or `"full"`, `<elasticity young poisson damping>`), per-step equal to MuJoCo C for the trilinear cube (§2) | arbitrary tetrahedral meshes (`flexcomp type="mesh"/"gmsh"`) untested; full-dof volumes at young 5e3 are unstable at 1–2 ms in MuJoCo C itself; `dof="quadratic"` unsupported in MuJoCo Warp |
| `DeformableBodyMaterialCfg` (youngs_modulus 5e7, poissons_ratio 0.45, damping, friction, density) | `young`, `poisson`, `damping`, friction, mass per flex | per-world material randomization (MuJoCo Warp batches model fields; flex stiffness arrays not yet exposed per world) |
| nodal state: `nodal_pos_w`, `nodal_vel_w`, `nodal_state_w`, `write_nodal_state_to_sim` | `DeformableSim.nodal_pos()`, `nodal_vel()` (zero-copy MPS views), `set_qpos`, `randomize` | batched on-device writes (no host round trip) |
| `write_nodal_kinematic_target_to_sim` (partial kinematic control) | – | flex `pin`s or equality/mocap-driven vertex bodies |
| `sim_element_deform_gradient_w`, `sim_element_stress_w`, element quaternions | – | from MuJoCo Warp's elasticity kernels (per-element F is computed there) |
| cloth (Isaac Lab: not in the PhysX API; Newton VBD in 3.0) | 2D flex cloth (edge equality; or continuum with `elastic2d`), per-step equal to MuJoCo C | bending with `elastic2d="bend"` untested at scale; self-collision off in the demo (`selfcollide` supported by MuJoCo Warp, untested for fidelity) |
| cables / ropes (Newton VBD cable) | 1D flex with capsule-element contacts (fixed here) | no bending/twist energy without the `cable` plugin (unsupported by MuJoCo Warp) |
| coupling with articulations (PhysX two-way; Newton `CoupledSolver`) | one MuJoCo solve for robot + flex (two-way by construction); G1 + cloth + cable at 20.0K env-steps/s (2048 envs) | 4096 G1 worlds exceed memory with the default CCD workspace |
| throughput | 54.1K env-steps/s at 4096 for a 112-vertex cloth+cable scene (faithful), 2.35M for XPBD (not faithful) | Isaac publishes none; next cuts in §5 item 4; Metal EPA precision on mesh face contacts (§6) |

## 9. PhysX's deformable formulations from the source, and which MetalSim backend shares them

Primary sources (all read, not summarised from docs): PhysX SDK 5.6.1 (tag `107.3-physx-5.6.1`, the PhysX in Isaac Sim
5.1 / omni.physx 107.3) and PhysX main (5.11, `da950a3`), github.com/NVIDIA-Omniverse/PhysX (BSD-3, GPU code included);
Isaac Lab v2.3.2 and v3.0.0-EA (`ae37b02`); Newton `90e2332` (upstream/newton).

**Particle cloth (PBD).** `PxParticleClothBuffer` / `PxParticleSpring` (5.6.1 `include/PxParticleBuffer.h`), solved by
`ps_solveSpringsLaunch` (`source/gpusimulationcontroller/src/CUDA/particlesystem.cu` ~L4137–4300): every cloth
spring is a distance constraint with stiffness `k` and damping `d` applied as an implicit spring,
`a = dt(dt k + d)`, `x = 1/(1 + a w)`, position and velocity updated together, clamped to the full error — i.e. a
compliant (XPBD-type) distance constraint with compliance `1/k`. Springs are processed in partitions (graph colouring:
parallel Gauss–Seidel). Under TGS (Isaac's default) `PxgPBDParticleSystemCore::solveTGS` steps the particles
(`stepParticleSystems`, L541) and solves springs/contacts (`solveParticleCollision` → `solveSprings`, L373) once per
position iteration, so the 16 position iterations are 16 substeps ("small steps" XPBD). There are only distance springs:
omni.physx's `PhysxAutoParticleClothAPI` generates stretch (mesh edges), shear and bend springs (`springStretchStiffness`,
`springShearStiffness`, `springBendStiffness`, `springDamping`; `omni.physx.scripts.particleUtils.add_physx_particle_cloth`),
no volume constraint (`pressure` is only for inflatables). Contacts treat particles as spheres: `restOffset` (particle
radius for solids, `solidRestOffset`) and `contactOffset` (contact generation distance); self-collision filtering uses
2.01 × solidRestOffset (particlesystem.cu L1192). The particle cloth API is **gone on PhysX main** (5.11: the spring
solve is commented out in `PxgParticleSystemCore.h` L140, no `PxParticleSpring`); it is deprecated in omni.physx 107.3
("DEPRECATED: Will be replaced by new deformable implementation").

**FEM cloth (`PxDeformableSurface`, Isaac Sim 6 surface deformables).** `FEMCloth.cu`, `FEMClothUtil.cuh` (main): XPBD
"fixed corotated" membrane (L417) and XPBD "Discrete Shells" bending (L523); material `PxDeformableSurfaceMaterial`
(Young's modulus, Poisson, thickness, bending stiffness/damping).

**FEM soft body (`PxSoftBody` → `PxDeformableVolume`).** `softBodyGM.cu` (5.6.1): the simulation mesh is a hexahedral
grid ("GM"), each cell split into tetrahedra; per-tet XPBD constraints — default `eCO_ROTATIONAL` (ARAP deviatoric term
with `alphaTilde = 1/(dt² · 2 μ V)` plus a volume term) or `eNEO_HOOKEAN` (`tetrahedronsSolveInnerNeoHookean`,
`alpha = invDt² / E`), damping `elasticityDamping × dampingScale` inside the constraint (`PxDeformableVolumeMaterial.h`
L39–46: "eCO_ROTATIONAL: Default model. Well suited for high stiffness"). Solved in partitions (parallel Gauss–Seidel,
Jacobi fallback above `SB_PARTITION_LIMIT`, L1454–1503) once per TGS iteration; rotations extracted per tet
(Müller's rotation extraction, `deformableUtils.cuh` L43) once per step.

**Isaac Lab v2.3.2 mapping.** `DeformableBodyMaterialCfg` → `spawn_deformable_body_material`
(`sim/spawners/materials/physics_materials.py` L110–128) applies `PhysxSchema.PhysxDeformableBodyMaterialAPI` and sets
`density`, `dynamicFriction`, `youngsModulus`, `poissonsRatio`, `elasticityDamping`, `dampingScale` one to one.
`DeformableBodyPropertiesCfg` → `modify_deformable_body_properties` (`sim/schemas/schemas.py` L930–975): mesh and solver
options through `deformable_utils.add_physx_deformable_body(...)` (`simulation_hexahedral_resolution`,
collision simplification, `solver_position_iteration_count`, `vertex_velocity_damping`, sleep/settling, `self_collision`,
`self_collision_filter_distance`), `rest_offset` / `contact_offset` on `PhysxCollisionAPI`, the rest on
`PhysxDeformableAPI`. `DeformableObject` reads and writes nodes through `physics_sim_view.create_soft_body_view`
(`assets/deformable_object/deformable_object.py` L327). **Isaac Lab v2.3.2 exposes no cloth object**: only volume
soft bodies. The 5.1 recording therefore made its cloth with omni.physx `particleUtils` directly (ParticleClothDemo
values), and its rope as a thin volume soft body.

**Isaac Lab 3.0-EA.** `DeformableObject` covers both backends: PhysX (`isaaclab_physx`) surface deformables
(`PhysxSurfaceDeformableBodyMaterialCfg`: density 1000, friction 0.25, Young's modulus 1e6, Poisson 0.45,
`surface_thickness` 0.01, stretch/shear/bend stiffness, `bend_damping`, `elasticity_damping` 0.005) and volume
deformables (`PhysxDeformableBodyMaterialCfg`); Newton VBD (`isaaclab_newton`): volumes with `k_mu`, `k_lambda`, `k_damp`
(stable Neo-Hookean, Newton `particle_vbd_kernels.py` L178–218), cloth with `tri_ke`, `tri_ka`, `tri_kd` (Neo-Hookean
membrane, L532–578) and `edge_ke`, `edge_kd` (dihedral-angle bending, L694–798); a Newton-only `CableObject` (VBD rod:
`CableMaterialCfg` thickness, density, stretch/bend/shear/twist moduli). `scripts/demos/deformables.py` is the reference
usage. Recording script prepared: `metalsim/parity/isaac_side/record_deformables_il3.py` (both backends, same three
protocols); not run (the 3.0 environment is being built by another agent).

**Which MetalSim backend shares the formulation:**

| object | PhysX (5.1, recorded) | MuJoCo Warp flex | MetalSim XPBD | physical mappings / fitted |
|---|---|---|---|---|
| cloth | PBD particle cloth: compliant distance springs (stretch/shear/bend), 16 TGS substeps, sphere contacts | soft edge-equality constraints (or explicit edge springs / StVK membrane + bending), solved in the global constraint solve; capsule-triangle contacts | **same family**: compliant distance constraints on edges and bending pairs, small steps, coloured Gauss–Seidel, sphere-vertex contacts | XPBD: compliance = 1/k, damping = d per spring, substeps = position iterations, radius = rest offset, mass, friction (all physical); flex: solref (−2k/m, −2d/m) from the spring (physical in the small-deformation limit), contact solref fitted, no bending in equality mode |
| cable/rope | thin FEM volume (co-rotational XPBD tets) with kinematic end nodes | thin flex volume (StVK tets), pinned vertices: **same family** (FEM volume), different constitutive law and explicit integration | 1D chain with distance + bending constraints: different family (rod, not a volume) | flex: Young's modulus, Poisson ratio, density, pin (physical); elastic damping fitted (PhysX's 0.005 is unstable in explicit flex elasticity); XPBD: bending compliance and damping fitted |
| soft volume | FEM co-rotational XPBD on a hexahedral-cell tet mesh | flex volume (StVK, explicit): FEM family | not implemented | flex: E, ν, density, friction (physical); damping and contact softness fitted |

Newton VBD (Isaac Lab 3.0) is a block-descent implicit solver on the same kinds of energies (Neo-Hookean membranes and
volumes, dihedral bending): closer to flex's energy-based FEM than to PBD springs; the 3.0 recordings will say whether
XPBD (for PhysX PBD cloth) or flex (for FEM/VBD) is the closer default per object type.

## 10. PhysX (Isaac Sim 5.1) deformable protocol: residuals, fits, throughput (measured)

Reference recording (`runs/deformable/isaac51/`, recorded before the plan moved to Isaac Lab 3.0; Isaac Sim 5.1 /
Isaac Lab 2.3.2, L4, `metalsim/parity/isaac_side/record_deformables.py`, parameters in `meta.json`): (a) 1 m cloth,
21×21 PhysX particle cloth, ParticleClothDemo parameters (stretch 1e4, shear 100, bend 200, spring damping 0.2,
0.02 kg/particle = 8.82 kg, rest offset 0.025 m, 16 iterations, friction 0.6, self-collision on, no drag/lift),
released at z = 0.5 m over a 0.4 m box; (b) 0.5 × 0.02 × 0.02 m FEM rod (E 1e5, ν 0.4, density 1000, elasticity
damping 0.005, 44 nodes, 4 pinned) released from horizontal; (c) 0.2 m FEM cube (same material, 8 kg, 1331 nodes)
dropped from z = 0.5 m. 200 Hz, 5 s. Bulk metrics and fits: `scripts/diagnostics/deformable/physx_protocol.py`
(fit logs `runs/deformable/physx_fit51/`); per-vertex comparison is meaningless across these discretisations
(441 particles vs a 21×21 flex grid; 44/1331 hex-grid nodes vs 44/125 flex vertices or an 11-node chain), so only
bulk quantities are compared.

| object / metric | PhysX | XPBD, physical mapping | XPBD fitted | flex, physical mapping | flex fitted |
|---|---|---|---|---|---|
| cloth: rest height on box (m) | 0.427 | 0.425 | 0.425 | 0.429 (1 ms; diverges at 5 ms) | same |
| cloth: rest mean height (m) | 0.267 | 0.270 | 0.272 | 0.291 | same |
| cloth: xy extent (m) | 0.791 | 0.771 | 0.775 | 0.766 | same |
| cloth: settling time (s) | 1.07 | 1.23 | 1.14 | 2.62 | same |
| cloth: KE peak time (s) | 0.260 | 0.270 | 0.275 | 0.245 | same |
| cloth: height-map RMSE vs PhysX (m) / silhouette IoU | – | 0.039 / 0.72 | 0.033 / 0.73 | 0.038 / 0.71 | same |
| rope: time to vertical (s) | 0.37 | 0.34 | 0.34 | 0.39 | 0.39 |
| rope: swing period (s) | 1.026 | 1.323 | 1.323 | – (NaN: PhysX damping 0.005 is unstable) | 1.006 |
| rope: log decrement | 0.59 | 0.003 | 0.003 | – | 0.09 |
| rope: first back-swing x (m) | −0.305 | −0.472 | −0.472 | – | −0.456 |
| rope: rest tip drop (m) | 0.477 | 0.456 | 0.456 | – | 0.429 |
| cube: impact centroid minimum (m) | 0.079 | not implemented | – | – (unstable) | 0.067 |
| cube: bounce peak (m) | 0.135 | – | – | – | 0.262 |
| cube: rest centroid (m) | 0.098 | – | – | – | 0.100 |
| cube: settling time (s) | 0.55 | – | – | – | 1.97 |

Fitted parameters: XPBD cloth: velocity damping 0.5 1/s on top of the physical mapping (bend compliance 5e-3 vs 5e-2
made no difference to these metrics); XPBD rope: none helped (grid over bending compliance 1e-3..1e-1 and damping 0/0.2:
best is the default); flex cloth: the physical solref (−2k/m, −2d/m) at 1 ms beat positive solrefs (loss 61 vs 118–456);
flex rope: elastic damping 3e-5 (≤ 1e-4 is stable at 0.5 ms); flex cube: elastic damping 1e-3, contact solref
0.005 1 at 0.5 ms. The explicit flex elasticity cannot reach PhysX's damping (rope log decrement 0.09 vs 0.59, cube
bounce 0.26 vs 0.135 m): damping above 1e-4 (rod) / 1e-3 (cube) diverges at 0.5 ms.

Throughput of the fitted settings on Metal (graph replay, one 5 ms frame per env-step, `bench_protocol.py`,
`runs/deformable/physx_fit51/bench_fitted.log`):

| scene | 256 | 1024 | 4096 envs |
|---|---|---|---|
| cloth, XPBD (441 vertices, 16 substeps) | 144K | 316K | 312K env-steps/s |
| cloth, flex (441 vertices, 1 ms × 5) | 1.8K | 1.9K | 1.9K |
| rope, flex (44-vertex FEM rod, 0.5 ms × 10) | 4.9K | 9.2K | 11.7K |
| rope, XPBD (11-node chain, 16 substeps) | 147K | 731K | 2.62M |
| cube, flex (125-vertex FEM, 0.5 ms × 10) | 2.7K | 3.6K | 3.9K |

**Recommendation (fidelity first, against this PhysX 5.1 reference):** cloth → **XPBD** (PhysX's own formulation;
closest on every cloth metric, 160× flex's throughput); cable/rope → **flex FEM rod** (period within 2 %; XPBD's
chain is 29 % slow because it has no bending stiffness to speak of; neither reaches PhysX's damping); soft volume →
**flex** (only candidate; bounce and settling too lively). The Isaac Lab 3.0 recordings (PhysX FEM cloth, Newton VBD)
will replace this reference; for the Newton backend the primary comparison is the same Newton VBD solver on Metal.

| decision | options (numbers) | chosen, why | how to switch |
|---|---|---|---|
| cloth backend | XPBD: height-map RMSE 3.3 cm, settle 1.14 vs 1.07 s, 312K env-steps/s / flex: 3.8 cm, 2.62 s, 1.9K | XPBD (same formulation as PhysX particle cloth; closer and faster) | `XPBDSim` default for cloth; `DeformableSim` (flex) behind the backend switch |
| cable backend | flex rod: period 1.006 vs 1.026 s, 11.7K / XPBD chain: 1.323 s, 2.6M | flex (fidelity first) | XPBD chain via `XPBDSim` |
| soft-volume backend | flex only (XPBD volume not implemented) | flex | – |
| flex cloth stretch | physical solref (−2k/m, −2d/m) at 1 ms / positive solref at 5 ms | physical at 1 ms (loss 61 vs 118–456) | `edge_solref` |
