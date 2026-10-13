# Isaac Lab 3.0-EA particle / MPM reference: what PhysX and Newton expose, and how to record from each on an L4

Scope date: 2026-10-09. Evidence levels used below:
- **[SRC]**: verified by reading source in the local checkouts: Newton `v1.5.2` (`upstream/newton-1.5.2`, cf5378db, 2026-09-09, the version Isaac Lab 3.0-EA pins); Newton main-era fork (`upstream/newton`, 90e23324, 2026-09-24, whose CHANGELOG includes upstream 1.6.0 of 2026-09-10); Isaac Lab `v3.0.0-EA` (`upstream/IsaacLab3`, ae37b02, 2026-09-16); PhysX repo tag `ovphysx-0.6.3` (`upstream/PhysX`, da950a3, 2026-09-16, top CHANGELOG entry v5.11.0); omni.physx 5.6.1 Kit extensions (`upstream/PhysX-omni-5.6.1`).
- **[DOC]**: claims from official documentation or release notes.
- **[INF]**: my inference. Not verified.

Line numbers refer to the local checkouts at the tags above. GitHub URLs point to the same tags.

## Q1. PhysX backend (Isaac Sim 6.x, PhysX 5.6 to 5.11): PBD particles, schemas, Python APIs, MPM history

### Takeaway
The PBD particle system (`PxPBDParticleSystem`: fluids, granular, diffuse particles, isosurface/smoothing/anisotropy post-processing) **still exists and is maintained** in current PhysX (5.9 to 5.11 add fixes and new features for it). PhysX 5.9.0 **removed** particle cloth, and with it particle rigids, particle attachments/filters and particle volumes (inflatables). PhysX had experimental FLIP and MPM particle systems. They were deprecated earlier and **removed in v5.4.0**, so PhysX has had no MPM solver since then. Isaac Lab 3.0-EA does **not** wire PhysX particles into any asset or task. Its MPM is Newton-only.

### Cited Findings
- [SRC] PhysX v5.4.0 CHANGELOG, "Removed": "Experimental FLIP and MPM Particle Systems (e.g. PxFLIPParticleSystem, PxFLIPMaterial, PxMPMParticleSystem, PxMPMMaterial)" (CHANGELOG.md:1147 and :1252, under `# v5.4.0-106.0` at :1089). The same release made `PxParticleSystem` a typedef of `PxPBDParticleSystem` and `PxParticleMaterial` a typedef of `PxPBDMaterial` (":1257-1258"). It also deprecated particle cloth, particle rigids, particle attachments and particle volumes (:1139, :1262-1276). — [PhysX CHANGELOG](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/CHANGELOG.md)
- [SRC] PhysX v5.9.0-110.1, "Particles": "Removed the deprecated particle cloth feature. Use PxDeformableSurface as a replacement" (:332). The removal includes `SnippetPBDCloth` and `SnippetPBDInflatable` (:340). It also removed particle-based rigids (:342), particle attachment/filter features (:350) and particle volumes (`PxParticleVolume*`, :358). PhysX 5.9 removed `PxParticleSolverType`, so PBD is the only particle solver type (:370). It deprecated `setMaxVelocity` in favour of `setMaxLinearVelocity` (:375). — [PhysX CHANGELOG](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/CHANGELOG.md)
- [SRC] PBD particles are still actively changed after 5.9. In v5.10.0, mass-splitting was extended to PBD particle contacts with rigid and deformable bodies (:168, :177), and `PxPBDParticleSystem::setMaxDepenetrationVelocity` now takes effect (:178). In v5.11.0, velocity iterations now solve particles "only for velocity, not position errors; previously these iterations applied only position corrections" (:102), and `minVelocityIters` is the second argument of `setSolverIterationCounts` (default 1, :104). Lowering `setSolverIterationCounts` now takes effect (:97). The particle–deformable rest distance no longer double-counts the deformable rest offset (:98). **Parity consequence:** a PBD fluid recorded on Isaac Sim 5.1 (PhysX 5.6.1) and one recorded on Isaac Sim 6.x (PhysX ≥ 5.9) are not guaranteed to match. — [PhysX CHANGELOG](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/CHANGELOG.md)
- [SRC] Current PhysX `physx/include` still ships `PxPBDParticleSystem.h`, `PxPBDMaterial.h`, `PxParticleBuffer.h`, `PxParticleGpu.h`, `PxParticleMaterial.h`, `PxParticleNeighborhoodProvider.h`, `PxParticleSystem.h`, `PxParticleSystemFlag.h`, `PxIsosurfaceExtraction.h`, `PxAnisotropy.h` and `PxSmoothing.h`. The only particle snippet left is `snippetisosurface`. 5.6.1 also had `snippetpbdcloth` and `snippetpbdinflatable`. — [PhysX include dir](https://github.com/NVIDIA-Omniverse/PhysX/tree/main/physx/include)
- [SRC] `PxPBDMaterial` setters in current PhysX: `setFriction`, `setDamping`, `setAdhesion`, `setGravityScale`, `setAdhesionRadiusScale`, `setViscosity`, `setVorticityConfinement`, `setSurfaceTension`, `setCohesion`, `setCFLCoefficient`, `setParticleFrictionScale` and `setParticleAdhesionScale`. `setLift` and `setDrag` are `PX_DEPRECATED` (PxPBDMaterial.h:34-273). — [PxPBDMaterial.h](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/include/PxPBDMaterial.h)
- [SRC] USD schemas (PhysX repo `schemas/physx/source/physxSchema`, schema version 25.11.1). They still contain `PhysxParticleSystem` (typed prim), `PhysxParticleAPI`, `PhysxParticleSetAPI`, `PhysxParticleSamplingAPI`, `PhysxPBDMaterialAPI`, `PhysxParticleIsosurfaceAPI`, `PhysxParticleSmoothingAPI`, `PhysxParticleAnisotropyAPI` and `PhysxDiffuseParticlesAPI`. All are tagged "WARNING: This is a draft API; the design is not fixed". The defaults in `generatedSchema.usda` are as follows:
  - `PhysxParticleSystem`: `particleContactOffset = 0.05`, `contactOffset/restOffset/solidRestOffset/fluidRestOffset = -inf` (auto), `solverPositionIterationCount = 16`, `maxNeighborhood = 96`, `neighborhoodScale = 1.01`, `maxVelocity = inf`, `maxDepenetrationVelocity = inf`, `enableCCD = 0`, `globalSelfCollisionEnabled = 1`, `nonParticleCollisionEnabled = 1`, `wind`.
  - `PhysxPBDMaterialAPI`: `friction = 0.2`, `damping 0`, `viscosity 0`, `surfaceTension 0`, `cohesion 0`, `vorticityConfinement 0`, `adhesion 0`, `adhesionOffsetScale 0`, `particleFrictionScale 1`, `particleAdhesionScale 1`, `cflCoefficient 1`, `gravityScale 1`, `density 0`.
  - `PhysxParticleAPI`: `particleGroup 0`, `selfCollision 1`, `particleEnabled 1`.
  - `PhysxParticleSetAPI`: `physxParticle:fluid = 1`, `physxParticle:simulationPoints`.
  - `PhysxParticleSamplingAPI`: `maxSamples = 50000`.

  MIGRATIONGUIDE.md (March 2026) removes `PhysxParticleClothAPI` and `PhysxAutoParticleClothAPI`, and drops `physxPBDMaterial:lift` and `:drag`. — [physxSchema](https://github.com/NVIDIA-Omniverse/PhysX/tree/main/schemas/physx/source/physxSchema)
- [DOC] omni_physics "Particles" dev guide (latest, updated 2026-07-23) says: "The particle cloth feature has been removed", "CPU simulation of particles is not supported, so you must have GPU simulation enabled", and "The particles schema is not finalized". Particles in different particle systems cannot collide with each other. Materials are per particle system, not per object. — [omni_physics particles](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html)
- [SRC] The omni.physx Python helper `omni.physx.scripts.particleUtils` (5.6.1 Kit source) provides `add_physx_particle_system`, `add_pbd_particle_material`, `AddPBDMaterialWater`, `AddPBDMaterialViscous`, `add_physx_particleset_points`, `add_physx_particleset_pointinstancer`, `configure_particle_set`, `add_physx_particle_anisotropy`, `add_physx_particle_smoothing`, `add_physx_particle_isosurface`, `add_physx_diffuse_particles`, `create_particles_grid` and `poisson_sample_mesh`, plus the cloth helpers that 5.9 removes. — [particleUtils.py (omni 5.6.1)](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/omni/extensions/runtime/source/omni.physx/python/scripts/particleUtils.py)
- [DOC] Isaac Sim 6.1 docs: `isaacsim.core.prims` ("Extension deprecated since Isaac Sim 6.0.0 in favor of ... isaacsim.core.experimental.prims") still lists `ParticleSystem`/`SingleParticleSystem`, which wrap PhysX particle systems, and a "Deprecated cloth prim class" `ClothPrim`. The documented particle-system methods are solver and offset setters (`set_wind`, `set_max_velocity`, contact and rest offsets, CCD). The page says solver parameters cannot change while the scene is playing. No position getter appeared in the part of the page I read. — [isaacsim.core.prims 6.1.0](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/deprecated/isaacsim.core.prims/docs/index.html)
- [SRC] In Isaac Lab 3.0-EA, the only PhysX-particle code is a type check in `isaaclab/sim/utils/prims.py:885` (`has_particle_system = prim.GetTypeName() == "PhysxParticleSystem"`, used to allow binding a physics material). There is no PhysX particle spawner, asset class or task. `docs/source/concepts/deformables.rst:46-47` says: "Particle-based materials such as fluids and granular media are not deformables in this sense. They use a separate asset and solver; see using_mpm" (this is the Newton MPM path). — [IsaacLab v3.0.0-EA](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA)

### Inferences
- [INF] To get a PhysX PBD fluid or granular reference on Isaac Sim 6.1, bypass Isaac Lab and author USD directly with `particleUtils` (`add_physx_particle_system` + `add_pbd_particle_material` + `add_physx_particleset_points`), as the user's 5.1 recorder `metalsim/parity/isaac_side/record_deformables.py:32,74-81` already does for cloth. Enable GPU dynamics (particles are GPU-only). The 6.x signatures of `add_pbd_particle_material` may no longer accept `drag`/`lift`, because those attributes are gone from the schema.
- [INF] Readback options for PhysX particle sets on 6.1, in order of preference: (1) a tensor view, if `omni.physics.tensors` in 6.1 exposes a particle-set or particle-buffer view (5.1 had `create_particle_cloth_view`, which the existing recorder used at :118, but cloth is gone and I did not verify a set view); (2) reading the `points` attribute of the Points/PointInstancer prim after each step, with omni.physx's USD writeback for particles enabled; (3) a fallback that reads `PxParticleBuffer::getPositionInvMasses()` through ovphysx, the kit-less PhysX wheel. Run a smoke test on the VM to confirm which path works before the full recording.
- [INF] For comparison against our Metal MPM, PhysX PBD is a different model class (position-based SPH-like constraints, not continuum plasticity). It is a cross-model sanity reference only. Newton ImplicitMPM is the like-for-like reference.

### Gaps
- I did not verify which exact PhysX version Isaac Sim 6.1 ships. The local research note says Isaac Sim 6.x uses PhysX ≥ 5.9, and the PhysX repo at the Isaac Lab 3.0-EA date is at 5.11.0 (ovphysx 0.6.3). Check `omni.physx` extension metadata on the VM.
- I did not confirm the 6.1 signatures of `particleUtils`, its USD writeback setting name, or whether `omni.physics.tensors` has a particle-set position view. I had no Isaac Sim 6.1 install to read.
- The Isaac Sim 6.1 docs page for the deprecated prims was only partly read (100k of 345k characters).

## Q2. Newton ImplicitMPM (1.5.2 as pinned, and main / 1.6.0)

### Takeaway
`newton.solvers.SolverImplicitMPM` is a **fully implicit, unconditionally time-step-stable MPM**. It follows Daviet & Bertails-Descoubes, "A semi-implicit material point method for the continuum simulation of granular materials" (SIGGRAPH 2016, DOI 10.1145/2897824.2925877; the attribution of authors and title to that DOI is from my memory). The solver is built on `warp.fem`. It solves a coupled rheology + frictional-contact complementarity problem on a grid with a GPU-friendly iterative solver (Gauss-Seidel colouring, Jacobi, CG/CR/GMRES). The material model is a single family: Drucker-Prager-type pressure-dependent yield with a compressive cap, tensile yield, deviatoric yield stress (Bingham-like), plastic viscosity, dilatancy and hardening/softening, with optional elasticity. Sand, snow, mud and viscous fluid are parameter choices within it. Rigid coupling is two-way through collider impulses, either hand-written (examples) or through `SolverCoupledProxy` with MuJoCo-Warp, Kamino, XPBD or VBD.

### Cited Findings
- [SRC] Class docstring (`newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:739-797`, v1.5.2): "Implicit MPM solver for granular and elasto-plastic materials. Implements an implicit Material Point Method (MPM) algorithm roughly following [1], extended with a GPU-friendly rheology solver supporting pressure-dependent yield (Drucker-Prager), viscosity, dilatancy, and isotropic hardening/softening. This variant is particularly well-suited for very stiff materials and the fully inelastic limit. It is less versatile than traditional explicit MPM but offers unconditional stability with respect to the time step. [1] https://doi.org/10.1145/2897824.2925877". The class inherits `SolverBase` and `CouplingInterface`. — [solver_implicit_mpm.py v1.5.2](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)
- [SRC] Module size in 1.5.2: about 10.9 k lines (`solver_implicit_mpm.py` 3,731; `solve_rheology.py` 2,058; `rheology_solver_kernels.py` 1,489; `implicit_mpm_solver_kernels.py` 1,321; `implicit_mpm_model.py` 919; `rasterized_collisions.py` 876; `contact_solver_kernels.py` 280; `render_grains.py` 254). It imports `warp.fem as fem` and `warp.sparse as wps` (:15-16). — same link
- [SRC] Step pipeline (`_step_impl`, :2510-2572; `step`, :1639-1667). `step` calls `_particles_to_cells` (a `fem.PicQuadrature` over the particles), rebuilds the scratchpad (function spaces), and then runs these stages:
  1. rasterize colliders;
  2. unconstrained velocity (P2G mass and momentum, gravity) with inverse mass;
  3. collider rigidity operator (two-way rigid coupling);
  4. elasticity compliance system;
  5. plasticity system (strain matrix, yield parameters);
  6. warm start;
  7. `_solve_rheology` (the coupled rheology/contact solve);
  8. save warm start;
  9. `_update_particles` (G2P + advection);
  10. save data.

  `control` and `contacts` are unused ("collisions are handled internally"). `state_out` may equal `state_in`. There is **one implicit solve per `step` call** and no internal substepping. — same link
- [SRC] Advection (`implicit_mpm_solver_kernels.py:336-358`): particle velocity is the grid velocity interpolated at the particle, clamped to `model.particle_max_velocity` (CFL clamp). Then `pos += dt * v` and the velocity gradient is stored for APIC (`particle_qd_grad`). — [implicit_mpm_solver_kernels.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py)
- [SRC] `SolverImplicitMPM.Config` (:799-924) with defaults:
  - Numerics: `max_iterations=250`; `tolerance=1e-4`; `solver="auto"`, where auto means `"gs"` for a Q1 velocity basis and `"gs-batched"` for B2/B3. The other options are `gs-soa`, `jacobi`, `cg`, `cr` and `gmres`, and an ordered tuple warm-starts left to right, e.g. `("cg","gauss-seidel")`. `warmstart_mode="auto"` (alternatives: none, particles, grid, smoothed). `collider_velocity_mode="forward"` or `"backward"`.
  - Grid: `voxel_size=0.1`; `grid_type="sparse"`, `"dense"` or `"fixed"`; `grid_padding=0`; `max_active_cell_count=-1`; `max_leaf/lower/upper_node_count=-1`.
  - Transfer and quadrature: `transfer_scheme="apic"` (or `"pic"`); `integration_scheme="pic"` (or `"gimp"`).
  - Material and background: `critical_fraction=0.0`; `air_drag=1.0`.
  - Bases: `collider_basis="S2"`; `strain_basis="P0"`; `velocity_basis="Q1"` (or `"B2"`, `"B3"`).
  - Worlds: `separate_worlds=False` (marked experimental).

  — same link
- [SRC] Grid: a sparse grid is a NanoVDB-backed `fem.Nanogrid`, rebuilt each step or rebuilt in place within a fixed capacity. Dense is an AABB `fem.Grid3D` (`_allocate_grid`, :2056-2215). A sparse grid is "rebuildable" (graph-capturable) when `max_active_cell_count > 0`, `grid_padding == 0` and `velocity_basis == "Q1"` (docstring :764-768). Outer CUDA-graph capture requires all of these: CUDA, a memory pool, conditional-graph support, `enable_timers=False`, a positive active-cell capacity, and a fixed or rebuildable sparse grid (:770-779). — same link
- [SRC] Constitutive parameters are per-particle custom attributes in the `mpm` namespace, registered by `SolverImplicitMPM.register_custom_attributes(builder)` (:926-1128). Each is listed below as `name` default: meaning.
  - `young_modulus` 1e15 Pa: rigid-plastic limit by default.
  - `poisson_ratio` 0.3.
  - `damping` 0 s: elastic damping relaxation time.
  - `friction` 0.5: Drucker-Prager friction coefficient μ (tan of the friction angle).
  - `yield_pressure` 1e15 Pa: compressive cap.
  - `tensile_yield_ratio` 0.
  - `yield_stress` 0 Pa: deviatoric (cohesive/Bingham) yield stress.
  - `hardening` 0, `hardening_rate` 1, `softening_rate` 1.
  - `dilatancy` 0.
  - `viscosity` 0 Pa·s.

  State attributes are `particle_qd_grad` (APIC), `particle_elastic_strain` (mat33), `particle_Jp` (plastic volume ratio), `particle_stress` (Cauchy, Pa) and `particle_transform` (rendering). — same link
- [SRC] Yield surface encoding (`rheology_solver_kernels.py:40-99`): `YieldParamVec` = [p_max·√(3/2), p_min·√(3/2), s_max, …]. `shear_yield_stress` uses μ = yield_params[3]/p_max on the clamped normal stress plus s. Development toggles include `_USE_CAM_CLAY = False` ("Use Modified Cam-Clay flow rule instead of the piecewise-linear anisotropic one"), `_INCLUDE_LEFTOVER_STRAIN = False` and `_ISOTROPIC_LOCAL_LHS = False`. So the **default flow rule is piecewise-linear and anisotropic**, not Cam-Clay. — [rheology_solver_kernels.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/rheology_solver_kernels.py)
- [SRC] Changelog history (Newton 1.5.2 `CHANGELOG.md`):
  - Solver added 2025-06-24 in commit e9dfe8a3, "Add implicit MPM solver and Anymal-on-sand example (#180)". Multi-material support followed (#750, 2025-09-24).
  - 1.1.0 (2026-04-13): added `velocity_basis` and GIMP. Added `mpm:viscosity`, `dilatancy`, `hardening_rate` and `softening_rate`. Added the beam-twist, snow-ball and viscous examples. The hardening law changed to `sinh(-h·log Jp)`.
  - 1.2.0 (2026-05-12): added gs-soa, gs-batched, cr and gmres. The default solver became `"auto"` and the default collider basis became `"S2"`.
  - 1.4.0 (2026-07-16): proxy coupling with MPM collider/transfer proxies. Inactive particles are excluded from transfers ("keep those particles active and set their mass to 0" for kinematic obstacles). Added `collider_particle_ids`.
  - 1.5.0 (2026-08-11): graph-capturable rebuildable sparse grids and opt-in isolated multi-world MPM.

  — [Newton CHANGELOG v1.5.2](https://github.com/newton-physics/newton/blob/v1.5.2/CHANGELOG.md)
- [SRC] Newton 1.6.0 (2026-09-10) and later main, not in the 1.5.2 pin:
  - `newton.geometry.ParticleSurface` / `extract_particle_surface()`, with a new `mpm_water_dam_break` example.
  - USD import of MPM particle sets from `UsdGeom.Points` with `NewtonParticleAPI` + `NewtonMPMSceneAPI` / `NewtonMPMMaterialAPI` (and `NewtonPointsDeformableSimAPI`), plus `SolverImplicitMPM.Config.create_from_usd()`; requires `newton-usd-schemas` ≥ 0.5.0. The asset is `newton/examples/assets/mpm_sand.usda` and the test is `tests/test_import_usd_mpm.py`.
  - Long solver-name aliases (`conjugate-gradient`, …).

  — [Newton CHANGELOG main](https://github.com/newton-physics/newton/blob/main/CHANGELOG.md)
- [SRC] Newton's solver feature table (`docs/solvers/index.rst:89-96`) lists `SolverImplicitMPM` as "Implicit", with particles ✅ and rigid bodies, articulations, cloth and soft bodies ❌. :206 says ImplicitMPM does not support joints. `docs/concepts/coupling.rst:348`: "Implicit MPM supports proxy body and proxy particle rewind/harvest hooks". — [Newton docs/solvers](https://github.com/newton-physics/newton/blob/v1.5.2/docs/solvers/index.rst)
- [SRC] Colliders: by default, every model shape with `ShapeFlags.COLLIDE_PARTICLES` is a collider. `setup_collider(...)` (keyword-only since 1.4) takes `collider_meshes`, `collider_body_ids`, `collider_margins`, `collider_friction`, `collider_adhesion`, `collider_projection_threshold`, `collider_particle_ids` (deformable mesh colliders), `model`, `body_com`, `body_mass`, `body_inv_inertia`, `body_q` and `collider_world_ids` (:1262-1331). Kinematic bodies are zero-effective-mass colliders. Other helpers: `collect_collider_impulses(state)`, which returns (impulses, positions, collider ids) (:1914); `project_outside(state_in, state_out, dt, gap=None)` (:1943); `update_particle_frames` and `update_render_grains` (rendering); `reset(...)` (:1545); `check_sparse_grid_rebuild_status()` (:1338). — solver link above
- [SRC] Two-way rigid coupling, hand-written (`examples/mpm/example_mpm_twoway_coupling.py`). The MuJoCo rigid solver (`SolverMuJoCo(use_mujoco_contacts=False, njmax=100)`) runs 4 substeps at 100 fps, applying the sand impulses collected on the previous frame as `body_f`. Then the sand step removes the previously applied forces from the body velocities (`subtract_body_force`), calls `mpm_solver.step(..., dt=frame_dt)` **once per frame**, and calls `collect_collider_impulses`. The scheme is lagged and staggered. The sand runs on a separate `sand_model` and calls `setup_collider(model=self.model)`. — [example_mpm_twoway_coupling.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/examples/mpm/example_mpm_twoway_coupling.py)
- [SRC] Generic coupling: `SolverCoupledProxy` examples in `newton/examples/multiphysics/`:
  - `example_mujoco_mpm_coupled_solver.py`: rigid boxes in MuJoCo or Kamino fall into an MPM bed; "MPM grid impulses are harvested back into rigid-body wrenches".
  - `example_vbd_mpm_coupled_solver.py`: a VBD tet cube on an MPM heap, with the soft-body surface as a prescribed-velocity deformable mesh collider.
  - `example_xpbd_mpm_coupled_solver.py`: XPBD particles as transfer-active proxy particles, which "participate in P2G/G2P momentum exchange, but ... do not contribute material stress".

  — [newton/examples/multiphysics](https://github.com/newton-physics/newton/tree/v1.5.2/newton/examples/multiphysics)
- [SRC] Shipped MPM examples (v1.5.2, `newton/examples/mpm/`; run as `python -m newton.examples <name>`). Defaults are from each argparse:
  - `example_mpm_granular.py`: a sand block in `[-1,-1,1.5]..[1,1,3.5]` dropped onto a collider (`cube`/`wedge`/`concave`/`none`) and a ground plane with μ 0.5. 60 fps, 1 substep, voxel 0.1, 3 particles per cell per axis (`add_particle_grid` with jitter = 2·radius), density 1000, E 1e15, ν 0.3, μ 0.68, yield_pressure 1e12, tensile 0, yield_stress 0, max_iterations 250, tolerance 1e-4, strain basis P0, sparse grid. It calls `project_outside` after each step. **[INF] Particle count** is about (3·20+1)²·(3·20+1) ≈ 227 k.
  - `example_mpm_anymal.py`: ANYmal C with a pretrained policy on MPM sand. MuJoCo runs at 50 fps with 4 substeps. MPM runs once per frame, with voxel 0.03, 3 ppc, tolerance 1e-6, max_iterations 50 and a sparse grid.
  - `example_mpm_twoway_coupling.py`: voxel 0.05, fixed grid, padding 50, `max_active_cell_count = 1<<15`, P0, 50 iterations, sand μ 0.75.
  - `example_mpm_multi_material.py`: sand, snow and mud. Snow: yield_pressure 2e4, tensile 0.2, μ 0.1, hardening 10, dilatancy 1. Mud: yield_pressure 1e10, yield_stress 300, tensile 1, μ 0, viscosity 100. Voxel 0.05, 60 fps, 250 iterations, tolerance 1e-6.
  - `example_mpm_snow_ball.py`: 60 fps, density 400, E 1.4e6, ν 0.3, μ 0.5, damping 0.01, yield_pressure 1.4e6, tensile 0.2, hardening 5, dilatancy 1, solver `("cg","gauss-seidel")`, 150 iterations, `add_particles` at 2 per cell per axis.
  - `example_mpm_viscous.py`: viscous fluid through a funnel mesh collider. 240 fps, voxel 0.005, viscosity 50, tensile 1.0, μ 0.
  - `example_mpm_beam_twist.py`: an elastic beam twisted. 240 fps, voxel 0.25, E 5e6, ν 0.45, damping 0.001, solver `cr`, strain basis P1d.
  - `example_mpm_grain_rendering.py`: grain rendering.
  - Main/1.6.0 adds `example_mpm_water_dam_break.py`.

  — [newton/examples/mpm](https://github.com/newton-physics/newton/tree/v1.5.2/newton/examples/mpm)
- [SRC] There is a benchmark (`asv/benchmarks/simulation/bench_implicit_mpm.py`) but no published numbers in the repo. It times a 14³ grid of particles (2,744), E 1e4, voxel 0.05, fixed grid, PIC transfer, Jacobi solver. — [bench_implicit_mpm.py](https://github.com/newton-physics/newton/blob/v1.5.2/asv/benchmarks/simulation/bench_implicit_mpm.py)

### Inferences
- [INF] "Implicit" here means the velocity/stress problem at the end of the step is solved as one coupled complementarity system (Daviet-style). There is no explicit stress update and no CFL-bound time step. It is neither MLS-MPM nor explicit APIC-MPM, although transfers default to APIC. A Metal reimplementation must reproduce the rheology solve (and its tolerance and iteration cap) to match trajectories. Matching only the P2G/G2P transfers is not enough.
- [INF] Trajectories are not bitwise-deterministic: Gauss-Seidel colouring, atomics in P2G, and the sparse-grid build order all vary. Compare statistically: heap profile, run-out distance, centre-of-mass trajectory, angle of repose. Expect differences of the order of the solver tolerance (1e-4 relative residual by default).
- [INF] A clean reference is a pure MPM scene with no coupling, for example `example_mpm_granular` with `--collider none` or `cube`. Run it through Newton 1.5.2 directly with `--grid-type fixed`, a fixed seed (the particle jitter is random) and `particle_q` saved at step 0. This isolates the solver from the coupling.

### Gaps
- I did not read the paper itself. The algorithm description comes from the docstring and code structure. I did not trace in detail the exact relation between `friction` and a friction angle (other than the μ = tan φ convention, [INF]), or the snow/hardening mapping.
- I found no published Newton MPM performance numbers (particles/s, ms per step) in the repo, CHANGELOG or docs.

## Q3. Isaac Lab 3.0-EA's integration of MPM (and PhysX particles)

### Takeaway
Yes. Isaac Lab 3.0-EA ships a first-class but **experimental** Newton MPM path. It has a solver config (`MPMSolverCfg` → `NewtonMPMManager`), an asset (`MPMObject`/`MPMObjectCfg`/`MPMObjectData`), spawners (`MPMGridCfg`, `MPMPointsCfg`, `MPMParticleMaterialCfg`), four demos, two contrib RL tasks with MuJoCo-Warp ↔ MPM proxy coupling (`IsaacContrib-Franka-Pour`, `IsaacContrib-UR10-Particle-Push`) and a docs page. There is no PhysX-particle equivalent.

### Cited Findings
- [DOC] The v3.0.0-EA release (published 2026-09-16) says it is "built for Isaac Sim 6.1, Python 3.12, PyTorch 2.11, NVIDIA Warp 1.16, and Newton 1.5.2". It lists "Newton physics with multi-solver capabilities - MuJoCo-Warp, VBD, MPM and coupled-solver workflows, with expanded support for deformables, cables, and particle systems" and "Newton VBD and rigid/deformable/MPM coupling". GA is "targeted toward the end of October 2026" on `release/3.0.0`. It also says "`--headless` is removed ... Use `--viz none` to explicitly disable all visualizers". — [IsaacLab v3.0.0-EA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
- [SRC] `docs/source/concepts/using_mpm.rst` (153 lines) says "MPM support and rigid-MPM coupling are experimental." It gives this minimal setup:
  - `SimulationCfg(dt=1/100, physics=NewtonCfg(solver_cfg=MPMSolverCfg(voxel_size=0.02, max_iterations=100, tolerance=1e-4), num_substeps=2))`
  - `MPMObjectCfg(prim_path="{ENV_REGEX_NS}/Media", spawn=MPMGridCfg(lower=..., upper=..., voxel_size=0.02, particles_per_cell=2.0, particle_placement="cell_center"))`

  Each Newton substep uses `SimulationCfg.dt / NewtonCfg.num_substeps`. `project_outside_colliders` is an optional post-step correction and is "not supported" for coupled entries. The page also covers tuning `CouplerProxyCfg`/`CouplerEntryCfg.substeps`/`CouplerProxyMappingCfg.mass_scale`. — [using_mpm.rst](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/docs/source/concepts/using_mpm.rst)
- [SRC] `isaaclab_newton/physics/mpm_manager_cfg.py:20-126`: `MPMSolverCfg(NewtonSolverCfg)`, with `class_type = "{DIR}.mpm_manager:NewtonMPMManager"` and `solver_type = "implicit_mpm"`. It mirrors every Newton `Config` field with the same defaults, and adds `project_outside_colliders: bool = False`. `mpm_manager.py:219-233`: `_step_solver` calls `solver.step(state_0, state_1, control, contacts, substep_dt)` and then optionally `project_outside`. `:88-102` registers the MPM custom attributes on the builder. — [mpm_manager.py](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab_newton/isaaclab_newton/physics/mpm_manager.py)
- [SRC] `isaaclab_newton/sim/spawners/mpm/mpm_cfg.py` defines three configs:
  - `MPMParticleMaterialCfg`: `density=1000`, `young_modulus=1e15`, `poisson_ratio=0.3`, `viscosity=0`, `friction=0.68`, `damping=0`, `yield_pressure=1e12`, `tensile_yield_ratio=0`, `yield_stress=0`, `hardening=0`, `dilatancy=0`. These match `example_mpm_granular`, not Newton's registry default of μ 0.5 and yield_pressure 1e15.
  - `MPMGridCfg`: `lower`, `upper`, `voxel_size`, `particles_per_cell=1.0`, `particle_placement="boundary"|"cell_center"`, `jitter=0`, `mass`, `radius`.
  - `MPMPointsCfg`: `positions`, `velocities`, `mass=1.0`, `radius=0.01`.

  The spawn function is `spawn_mpm_particles`. — [mpm_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab_newton/isaaclab_newton/sim/spawners/mpm/mpm_cfg.py)
- [SRC] `isaaclab_newton/assets/mpm_object/`: `MPMObjectCfg(DeformableObjectCfg)` and `MPMObject(BaseDeformableObject)`. `MPMObjectData` provides `particle_pos_w`, `particle_vel_w` and `particle_state_w`, plus `nodal_*` aliases and `root_pos_w`/`root_vel_w`. These are `ProxyArray`s gathered from `state.particle_q`/`particle_qd` by a Warp kernel and shaped `(num_instances, particles_per_object, 3)` (`mpm_object_data.py:65-107`). Writers include `write_nodal_state/pos/velocity/kinematic_target_to_sim_index/mask`. — [mpm_object](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA/source/isaaclab_newton/isaaclab_newton/assets/mpm_object)
- [SRC] Demos in `scripts/demos/mpm/`:
  - `newton_mpm_granular.py`: "Isaac Lab port of Newton's granular implicit-MPM example". It takes `--max_steps`, `--collider cube|wedge|concave|none` and `--voxel_size` (default 0.1).
  - `newton_mpm_twoway_coupling.py`.
  - `snowball_smash.py`.
  - `teapot_fill.py`, which `setup/demos.rst:84-88` describes as "Fill a Utah teapot with MPM water particles", with preset `data-demo-physics="newton_mpm"`.

  The isaaclab_newton CHANGELOG adds MPM in 0.16.0 (2026-06-12) and fixes "headless execution of the Newton MPM standalone demos" in 2.0.0 (2026-07-24). 5.1.0 (2026-08-14) adds graph-captured bounded-sparse snowball and teapot demos. — [scripts/demos/mpm](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA/scripts/demos/mpm); [isaaclab_newton CHANGELOG](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab_newton/docs/CHANGELOG.rst)
- [SRC] Contrib RL tasks:
  - `IsaacContrib-Franka-Pour` (`isaaclab_tasks/contrib/franka_pour/`). `sim.dt = 1/120` with decimation 4. `NewtonCfg(solver_cfg=CouplerProxyCfg(...), num_substeps=2, use_cuda_graph=True)`. The rigid entry is `MJWarpSolverCfg(use_mujoco_contacts=False, integrator="implicitfast", njmax=2560, nconmax=400)` with `substeps=3`. The media entry is `MPMSolverCfg(voxel_size=0.015, grid_type="sparse", strain_basis="P0", transfer_scheme="apic", max_iterations=24, tolerance=1e-4, collider_basis="pic27", solver="jacobi", separate_worlds=True)` with `substeps=2` and `in_place=True`. The proxy cups use `mass_scale=1000.0` and `mode="staggered"`, with coupler `iterations=1`. The material has density 1500, μ 0.7 and yield_pressure 1e12. The default fill is 735 particles. Train it with `--num_envs 2048`. The reset-dataset generator takes "about two minutes on an L40S-class GPU" (`pour_env_cfg.py:597-676`; using_mpm.rst).
  - `IsaacContrib-UR10-Particle-Push` (`contrib/ur10_particle_push/`): MJWarp + MPM, voxel 0.025, 2 ppc, collider margin 0.5·voxel.

  — [franka_pour/pour_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py); [ur10_particle_push](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA/source/isaaclab_tasks/isaaclab_tasks/contrib/ur10_particle_push)
- [SRC] `isaaclab_contrib/coupling/` (`CouplerProxyCfg`, `CouplerEntryCfg`, `CouplerProxyMappingCfg`) wraps Newton's `SolverCoupledProxy`. `physics_backends.rst:24` says: "Newton hosts MJWarp, Kamino, VBD, MPM, and other solver paths". — [IsaacLab docs/concepts](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA/docs/source/concepts)
- [DOC] Note on conflicting sources: a web search summary claimed "no Isaac Lab release note or doc page confirms MPM support". The v3.0.0-EA GitHub release text and the source tree above contradict it. That summary was looking at beta material. — [Isaac Lab coupled solvers doc](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/coupled_solvers.html)

### Inferences
- [INF] For a recorder, the cleanest Isaac Lab path is a `SimulationContext` + `InteractiveScene` holding one `MPMObjectCfg` with `MPMGridCfg(particle_placement="cell_center", jitter=0)`. That gives deterministic initial positions identical to what we generate on Metal. Static colliders are spawned as rigid `AssetBaseCfg` shapes, as in `newton_mpm_granular.py:125-136`. Read `obj.data.particle_pos_w` and `particle_vel_w` every step. Under the hood it is Newton 1.5.2's solver, so a direct Newton script gives the same physics with fewer moving parts. Record both if time allows; the Isaac Lab one proves "what Isaac users get".

### Gaps
- I did not verify that the hosted 3.0-EA docs page for using_mpm is identical to the source tree copy (it is presumably built from it).
- I did not read the `snowball_smash.py`/`teapot_fill.py` material values in full. teapot "water" is presumably a viscosity/tensile-ratio setting like `example_mpm_viscous` ([INF]).

## Q4. Practical recording on an NVIDIA L4 (24 GB, Linux, headless)

### Takeaway
Both paths run without a display on Linux CUDA. Newton MPM records trivially: `state.particle_q.numpy()` and `state.particle_qd.numpy()` after each `step`. The Isaac Lab path uses `MPMObject.data.particle_pos_w`. PhysX PBD particles need Isaac Sim (Kit) with GPU dynamics, and the readback route on 6.1 still has to be confirmed on the VM. No source gives L4-specific MPM issues or performance numbers.

### Cited Findings
- [SRC] Newton loop pattern, from every MPM example:
  - `builder = newton.ModelBuilder()`
  - `SolverImplicitMPM.register_custom_attributes(builder)`
  - `builder.add_particle_grid(pos, rot, vel, dim_x/y/z, cell_x/y/z, mass, jitter, radius_mean, custom_attributes={"mpm:friction": ...})` or `builder.add_particles(...)`
  - optional `add_ground_plane` / `add_shape_*` with `ShapeConfig(mu=...)`
  - `model = builder.finalize()`
  - optionally override per-particle values with `model.mpm.<attr>.fill_(...)`
  - `solver = SolverImplicitMPM(model, config=cfg)`
  - per step: `solver.step(s0, s1, None, None, dt)`, `solver.project_outside(s1, s1, dt)` (granular/multi-material), then swap.

  Positions and velocities are `state.particle_q` and `state.particle_qd` (`wp.array(vec3)`). The extra state is `state.mpm.particle_stress`, `particle_Jp` and `particle_elastic_strain`. — [example_mpm_granular.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/examples/mpm/example_mpm_granular.py)
- [SRC] Graph capture in the examples is done only when `wp.get_device().is_cuda` and `grid_type == "fixed"` (and an even substep count, for buffer swapping), e.g. `example_mpm_granular.py:113-114`. Isaac Lab additionally captures bounded rebuildable sparse grids (isaaclab_newton CHANGELOG 5.1.0). Sparse-grid overflow is reported asynchronously and must be checked with `check_sparse_grid_rebuild_status()` after replay. — solver docstring :764-779
- [SRC] Typical settings from the shipped examples:
  - Voxel size: 0.005 to 0.25 m, most often 0.03 to 0.1.
  - Particle density: 2 to 3 particles per cell per axis (8 to 27 per voxel).
  - Time step: one MPM step per 1/60 to 1/240 s frame. Coupled scenes use 1/100 to 1/120 s with 2 MPM substeps.
  - Solver effort: 24 to 250 rheology iterations; tolerance 1e-4 to 1e-6.
  - Particle counts: about 735 particles per env in Franka-Pour, and about 2.7 k in the asv bench. **[INF]** The granular example has about 227 k particles at its defaults.

  — sources above
- [DOC] The existing Isaac Lab 3.0 recorder in this repo (`metalsim/parity/isaac_side/record_deformables_il3.py:1-30, 87, 221-251`) already shows the pattern: `--physics isaacsim_physx|newton_vbd`, `SimulationContext(SimulationCfg(dt, device, physics=...))`, and `T(a) = (a.torch if hasattr(a,"torch") else a).detach().cpu().numpy()` on `ProxyArray`s such as `obj.data.nodal_pos_w` each step. An MPM recorder can copy it, swapping in `MPMObject.data.particle_pos_w` and `particle_vel_w`. Its usage string still says `--headless`, which 3.0-EA removed in favour of `--viz none` (release notes). — local file; [v3.0.0-EA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
- [DOC] PhysX particles are GPU-only ("CPU simulation of particles is not supported"). — [omni_physics particles](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html)
- [DOC] An NVIDIA forum answer for Isaac Sim 5.0 says the particle extension is not enabled by default in the UI. It suggests `carb.settings ... set("/physics/physxParticles", True)` and then `SingleParticleSystem`. This is a user forum post and is unverified for 6.1. — [NVIDIA forum: particle systems in isaacsim 5.0](https://forums.developer.nvidia.com/t/particle-systems-in-isaacsim-5-0/359442)

### Inferences
- [INF] Suggested reference set (three files per scene: `init.npz`, `traj.npz` with float32 `pos[T,N,3]` and `vel[T,N,3]`, and `meta.json` with every config value plus versions of Newton, Warp, CUDA driver and GPU):
  - (R1) A direct Newton 1.5.2 granular column collapse. Use `grid_type="fixed"` or `"sparse"`, a deterministic `cell_center` lattice (no jitter), a ground plane with μ 0.5, μ_material 0.68, voxel 0.05 and 3 ppc, about 30–60 k particles, dt 1/60 for 3 s. Record pos and vel every step, plus `particle_stress` and `particle_Jp` every N steps.
  - (R2) The same scene through Isaac Lab `MPMObject` with `physics=NewtonCfg(MPMSolverCfg)`.
  - (R3) Multi-material, snow and viscous variants using the example parameters.
  - (R4) The `example_mpm_twoway_coupling` or Franka-Pour reset states, for coupling.
  - (R5, optional) A PhysX PBD granular/fluid block via `particleUtils` on Isaac Sim 6.1, as a cross-model reference.

  At 24 GB, about 227 k particles at voxel 0.1 should fit easily; the memory risk is the dense grid on large domains. That is an estimate; I did not measure it.
- [INF] Newton 1.5.2 also runs on the Mac through the Metal Warp fork for VBD (local notes). Running ImplicitMPM there would give a same-code comparison on both GPUs before comparing our own Metal implementation. `warp.fem` / NanoVDB `Nanogrid` support on the Metal fork is unverified.
- [INF] The L4 (Ada, sm_89) supports CUDA conditional graphs on recent drivers. Confirm with a single uncaptured warm-up step and then capture, as the docstring requires ("Run an uncaptured warm-up step for rheology modes that build an inner graph lazily").

### Gaps
- I found no published Newton MPM or Isaac Lab MPM throughput numbers and no L4-specific known issues.
- I did not verify PhysX particle readback APIs on Isaac Sim 6.1 (tensor view vs USD writeback). Smoke-test this on the VM first.
- Determinism of Newton MPM across runs on the same GPU (atomics, GS colouring) has not been measured. Run the same reference twice on the L4 to establish a noise floor before comparing against Metal.
