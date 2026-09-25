# PhysX 5 deformable solvers: formulation, port feasibility to Warp/Metal, and the Newton VBD alternative

2026-09-25. Scope: what PhysX 5's GPU deformable solvers (PBD particle cloth, FEM deformable volume, FEM
deformable surface) actually compute, how far each is from the solvers MetalSim already has (MetalSim
`XPBDSim`, MuJoCo Warp flex) and from Newton's (VBD, XPBD), and what a faithful Warp/Metal port would take.
Companion: `newton_vbd_metal_2026-09-25.md` (Newton VBD and the coupled MuJoCo Warp + VBD solver on Metal,
the Isaac Lab 3.0 path). **Measured** = run here; **source** = read in the cited file; **estimated** = judgement.

Sources (all BSD-3 unless noted):
* [PX] `upstream/PhysX` = github.com/NVIDIA-Omniverse/PhysX `main` at da950a35 (2026-09-16, PhysX 5.11.0,
  ovphysx 0.6.3), shallow clone.
* [PX561] `upstream/PhysX-5.6.1` = tag `107.3-physx-5.6.1` (5ca9f472, PhysX 5.6.1.b07ddd12): the PhysX of
  Isaac Sim 5.1, which recorded every Isaac reference so far and is the last release with particle cloth.
* [OMNI] `upstream/PhysX-omni-5.6.1` = tag `107.3-omni-and-physx-5.6.1` (09ff24f3): the same PhysX plus the
  omni.physx Kit extensions (USD parsing, `particleUtils.py`), i.e. the code between Isaac Sim 5.1's USD and PhysX.
* File paths below are relative to `physx/source/` of [PX561] unless prefixed.
* Cross-check: `deformables_2026-09-25.md` §9 (the fitting agent, same sources) reaches the same formulations; this
  note adds the partition/copy/blend scheme, the exact spring and contact update formulas, the TGS schedule
  (iteration count raised for the whole island, bias coefficient, forward projection, finalisation), the hidden
  omni.physx cooker, code sizes and the port. §9 additionally notes the soft-body Jacobi fallback above
  `SB_PARTITION_LIMIT`.

## 0. Verdict (formulation)

| PhysX solver (who uses it) | formulation (source) | closest existing backend | algorithmically the same after |
|---|---|---|---|
| PBD particle cloth, `PxParticleClothBuffer` (Isaac Sim ≤ 5.1 "particle cloth"; the other agent's recorded cloth scene) | per TGS position iteration (= substep of dt/16): implicit spring-damper distance constraints on cooked springs (stretch/shear/bend classes), partitioned Gauss–Seidel through 8 combined partitions with per-particle copies blended by 1/(max copies + 1); particle–shape contacts from a once-per-step contact list with accumulated-impulse clamping and Coulomb friction, relaxation 0.5; self-collision through a hash grid, Jacobi-averaged with rest-pose filtering | MetalSim `XPBDSim` (small-step XPBD, coloured Gauss–Seidel distance constraints) and Newton `SolverXPBD` springs (same implicit spring-damper algebra, Jacobi) | new spring cooking (PhysX cooker's classes and bend pairs), PhysX's damping term and clamp, the copy/blend partition scheme, the once-per-step contact list with rest/contact offsets and accumulated impulses, velocity finalisation — a new backend, not a patch (§4, prototype `metalsim/physics/physx_cloth.py`) |
| FEM deformable volume, `PxDeformableVolume` (Isaac Lab's `DeformableObject` on PhysX; Isaac Sim 6's `DeformablePrim` volumes) | XPBD on tetrahedra of a voxel simulation mesh: co-rotational (default) = ARAP constraint ‖F − R‖ with compliance 1/(2μV) + volume constraint det F − 1 with compliance 1/(λV), XPBD damping from `elasticityDamping`; or Macklin-style Neo-Hookean; partitioned Gauss–Seidel with copies; embedded collision mesh; two-way rigid coupling inside every TGS iteration | Newton `SolverXPBD.solve_tetrahedra` (XPBD, but Neo-Hookean-style deviatoric + volume, Jacobi) | co-rotational ARAP + volume constraints, polar rotation per tet, voxel sim mesh + embedded collision mesh, PhysX partitioning, TGS schedule, rigid contact/impulse exchange per iteration |
| FEM deformable surface, `PxDeformableSurface` ("FEM cloth"; the only PhysX cloth from PhysX 5.9 / Isaac Sim 6.0) | XPBD fixed-corotated membrane (ARAP 2×2 + area constraint, compliance from E, ν, thickness) + XPBD discrete-shell bending on triangle pairs, partitioned with copies | none (MuJoCo flex `elastic2d` is StVK; Newton VBD is a different energy and solver) | a new backend |
| Newton VBD (Isaac Lab 3.0's Newton deformable path) | block-descent Newton per vertex on stable Neo-Hookean membranes (`particle_vbd_kernels.py:504`) + dihedral bending (`:666`) / stable Neo-Hookean tets (`:150`), penalty contacts; AVBD for rigid bodies | Newton itself: 1.5.2 runs unchanged on Metal, Newton main after one Warp fork fix (companion doc) | — |

Throughput is not the deciding factor for any of these: the formulations differ in what they converge to per
step (iteration counts are part of the model in position-based solvers), so fidelity to PhysX requires
reproducing the schedule, not only the constraint functions.

## 1. Versions and licence

* PhysX GPU source is public (since 5.6) and BSD-3 (`LICENSE.md`, "Copyright (c) 2008-2025, NVIDIA Corporation";
  identical text in [OMNI]). No GPL/Apache/MPL text anywhere under `physx/source` or `physx/include` (grep,
  source). [OMNI] ships two third-party licences (`omni/licenses/pyboost11-LICENSE.md`, `vhacd-LICENSE.md`),
  neither in the solver paths. Porting the algorithms (or code) into MetalSim is licence-compatible with
  attribution.
* **Particle cloth is gone from PhysX main.** [PX] `physx/CHANGELOG.md` v5.9.0-110.1, "Particles / Removed":
  "Removed the deprecated particle cloth feature. Use PxDeformableSurface as a replacement for cloth simulation"
  (with `PxParticleClothBuffer`, `PxParticleClothCooker`, `SnippetPBDCloth`); particle rigids, attachments and
  volumes removed as well. v5.9 also removed the `PxSoftBody` compatibility layer. So: Isaac Sim 5.1 (PhysX
  5.6.1) has particle cloth (deprecated); Isaac Sim 6.x (PhysX ≥ 5.9) has only FEM surface cloth and FEM volumes.
  The recorded PhysX cloth reference (the other agent's `record_deformables.py`, Isaac Sim 5.1) is particle
  cloth; any Isaac Lab 3.0 / Isaac Sim 6 PhysX cloth reference would be FEM surface cloth.
* Behaviour changes after 5.6.1 that matter for parity ([PX] CHANGELOG v5.11.0 Deformables/Particles): velocity
  iterations now solve deformables and particles for velocity only (previously position-only for volumes,
  ignored for surfaces); lowering solver iteration counts now takes effect; particle–deformable contacts no
  longer double-count the deformable rest offset. A port must therefore target one PhysX version; this survey
  targets 5.6.1 for particle cloth (the recordings) and notes 5.11 changes.

## 2. PBD particle cloth (PhysX 5.6.1)

### 2.1 Host side: how a cloth becomes springs

* API: `include/PxParticleBuffer.h` (`PxParticleSpring {ind0, ind1, length, stiffness, damping}`,
  `PxParticleCloth {startVertexIndex, numVertices, clothBlendScale, restVolume, pressure, ...}`),
  `include/extensions/PxParticleExt.h`, `include/PxPBDParticleSystem.h`, `include/PxPBDMaterial.h`.
* Spring generation (`physxextensions/src/ExtParticleClothCooker.cpp`, 464 lines, `cookConstraints`): all unique
  triangle edges; an edge that is the longest edge of both adjacent triangles is a quad diagonal → **shear**
  (`eTYPE_DIAGONAL_CONSTRAINT`) plus the other diagonal of the quad; other edges are **stretch**, classed
  vertical/horizontal by the angle to `verticalDirection` (45°); **bending**: for every vertex, pairs of
  neighbours that lie on a nearly straight line through it (|cos| > cos(`bendingConstraintMaxAngle`)) get a
  distance constraint between the two neighbours (i.e. i−1/i+1 along grid lines, not the flap-opposite vertices
  that MetalSim's `XPBDSim` uses). Rest lengths from the rest pose.
* **Isaac Sim's path is not in the open source.** `particleUtils.add_physx_particle_cloth`
  ([OMNI] `omni/extensions/runtime/source/omni.physx/python/scripts/particleUtils.py:515`) applies
  `PhysxAutoParticleClothAPI` with `springStretchStiffness/BendStiffness/ShearStiffness/Damping`; the springs
  are cooked by the omni.physx cooking service (`include/private/omni/physx/IPhysxCookingServicePrivate.h:131`,
  `ParticleClothMeshCookingParamsDeprecated`, with mesh welding on by default) whose implementation is **not** in
  the repository (`CookingDataAsync.cpp:3011-3100` only fills the parameters and stores the result). The result
  is written back to the prim (`storeParticleClothDataToUsdDeprecated`, `CookingDataAsync.cpp:3100-3135`:
  `physxParticle:springIndices/springStiffnesses/springDampings/springRestLengths`). **Action for the recording
  agent:** dump those four attributes (and the welded triangle indices) from the stage after `sim.reset()`; that
  removes the only unverifiable part of the cloth model. Until then the port uses `ExtParticleClothCooker` with all
  constraint types and assigns stretch/shear/bend stiffness by class (an assumption, flagged in the code).
* Partitioning (`physx/src/NpParticleBuffer.cpp:93-596`, `NpParticleClothPreProcessor`): greedy 32-bit
  partition masks (`PARTICLE_MAX_NUM_PARTITIONS_TEMP` 32) → partitions **combined** into
  `PARTICLE_MAX_NUM_PARTITIONS_FINAL` = 8 (partition p, p+8, p+16… merged; `combinePartitions`). Every spring
  endpoint gets its own copy slot; `remapOutput` chains the copy of a particle in one combined partition to its
  slot in a later one ("maintains the gauss-seidel part of the solver"), and chain ends go to per-particle
  accumulation slots. `clothBlendScale = 1/(maxCopies + 1)` over the cloth (`:578`, with the comment "AD Todo:
  figure out why blendScale is computed like this").

### 2.2 Step schedule (TGS, the Isaac default)

* One `PxScene::simulate(dt)` with the TGS solver: `stepDt = dt / numPositionIterations`
  (`gpusolver/src/PxgTGSCudaSolverCore.cpp:1269`), and `numPositionIterations` is the maximum over the island,
  which includes the particle system's `solverPositionIterationCount` (default 16, [OMNI] schema) and every
  rigid body/articulation (`PxgParticleSystemCore::getMaxIterationCount`). **A particle cloth with 16 iterations
  forces 16 TGS substeps on every rigid body in the scene.** `biasCoefficient = min(0.9, 2·sqrt(1/nPos))` = 0.5
  at 16 (`gpusolver/src/PxgContext.cpp:2004`).
* Before the solver: `ps_preIntegrateLaunch` (`gpusimulationcontroller/src/CUDA/particlesystem.cu:304`):
  v ← (v + g·dt)·(1 − min(1, damping·dt)); positions are projected forward by v·dt·0.5 (TGS,
  `PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS`, `gpucommon/src/CUDA/gridCal.cuh:114`) **only** for the hash grid
  and contact generation; the original positions are kept (`mOriginPos`).
* Contact generation once per step (narrow phase, `gpunarrowphase/src/CUDA/cudaParticleSystem.cu:378-500`,
  `particlePrimitiveCollision` `:268`, `contactPointBox` `gpunarrowphase/src/CUDA/particleCollision.cuh:42`):
  particle at its step-start position vs plane/sphere/capsule/box/convex-core; contact if distance ≤
  `cDistance` = particle-system contact offset + shape contact offset; stored error = distance − restDistance
  (particle rest offset + shape rest offset). Box: closest point, inside → nearest face. Triangle meshes and
  height fields use a separate one-way path (`particleSystemMeshMidphase.cu`, `particleSystemHFMidPhaseCG.cu`).
  Self-collision neighbour lists (`ps_selfCollisionLaunch` `:1145`): 27-cell hash grid, pairs within the
  particle contact distance, `maxNeighborhood` (96) cap, pairs whose rest distance ≤ 2.01·solidRestOffset
  filtered (`selfCollisionFilter`).
* Each of the 16 TGS position iterations (`PxgPBDParticleSystemCore::solveTGS`,
  `gpusimulationcontroller/src/PxgPBDParticleSystemCore.cpp:518`):
  1. `ps_stepParticlesLaunch` (`particlesystem.cu:120`): x ← x + v·stepDt (from the original position on the
     first iteration), deltaP accumulates the displacement.
  2. Springs (`solveSprings` `PxgPBDParticleSystemCore.cpp:952`): gather copies (`ps_updateRemapVertsLaunch`
     `:4043`), then one launch per combined partition (8) of `ps_solveSpringsLaunch` (`:4137`):
     with e = rest − l, b = h·k, d = h·c_d, a = h·b + d, x = 1/(1 + a·w_sum):
     Δλ = (x·b·e − x·a·(∇C·(v_i − v_j)))·h  (TGS: no λ accumulation), clamped to |Δλ| ≤ |e/w_sum|;
     Δx_i = +w_i·Δλ·n, Δx_j = −w_j·Δλ·n, and the copy's velocity += Δx/h. Negative stiffness = tether
     (compression allowed). Then `ps_averageVertsLaunch` (`:4312`): particle += clothBlendScale·Σ(copy − x).
     Then positions and velocities are updated (`ps_updateParticleLaunch` `:3010`: v += Δ/h, x += Δ).
  3. Self-collision (`ps_solveDensityLaunch` `:3404`, solid branch): per neighbour, mass-weighted push to
     2·solidRestOffset with an accumulated clamp, position friction (particle friction scale), adhesion;
     delta ×= coefficient (0.5); `ps_applyDeltaLaunch` (`:3689`) divides by max(count·relaxation, 1) (SOR-style
     Jacobi averaging).
  4. Rigid contacts (`ps_solvePCOutputParticleDeltaVTGSLaunch` `:1690`, `solvePCOutputDeltaVTGS` `:1473`):
     separation = error − n·Δx_particle + n·Δx_rigid (+ normal velocity·h); Δf = max(−f_acc, −sep·m_eff);
     friction bounded by μ·f_acc in the tangent plane of the accumulated displacement; the particle delta is
     scaled by min(0.7, biasCoefficient) = 0.5 and averaged over the particle's active contacts
     (`ps_accumulateDeltaVParticleLaunch` `:2828`). The rigid side gets the equal and opposite impulse in the same
     iteration (`ps_solvePCOutputRigidDeltaVTGSLaunch` `:2123`) — two-way coupling per TGS iteration.
  5. `updateParticles` (v += Δ/h, x += Δ).
* After the iterations (velocity iterations: 1 by default; 5.6.1 applies position corrections there, 5.11 velocity
  only): `ps_finalizeParticlesLaunch` (`:5731`): if |v| > |Δx_total/dt| then v ← 0.5·v + 0.5·Δx_total/dt
  (velocityScale = biasCoefficient; "we have the same code in FEMCloth"); `ps_integrateLaunch` (`:3087`):
  max-velocity clamp, lock flags, write back. Aerodynamic drag/lift (`ps_solveAerodynamics*`) and inflatable
  pressure are separate kernels (off in the recording: drag = lift = 0, pressure 0).

### 2.3 Size and dependencies

| piece | files (lines) | coupling to PhysX structures |
|---|---|---|
| spring solve + partition blend | `particlesystem.cu` kernels 4043–4435 (~400), `NpParticleBuffer.cpp` 93–596 (~500, host) | none beyond the cloth buffer; portable as is |
| integration, finalize, damping | `particlesystem.cu` 120–400, 3010–3220, 5731–5780 (~500) | particle sort order (hash-grid sorted arrays) |
| self-collision | `particlesystem.cu` 803–1237, 3404–3756 (~800) + grid sort (`PxgParticleSystemCore.cpp` 630–1080) | radix sort, cell start/end tables |
| particle–shape contacts | `cudaParticleSystem.cu` (1,714), `particleCollision.cuh` (224), mesh/HF midphases (~1,100), contact sort/prepare/solve in `particlesystem.cu` 1238–2276 (~1,000) | PhysX broadphase pairs, shape/transform caches, solver-body/articulation velocity readers (`PxgVelocityReader`), rigid delta accumulation (`rigidDeltaAccum.cu`) |
| host orchestration | `PxgPBDParticleSystemCore.cpp` (2,303), `PxgParticleSystemCore.cpp` (2,750) | streams/events, buffer management |

Total PBD particle path ≈ 5,900 kernel lines (`particlesystem.cu`) + 2,800 narrow phase + 5,000 host, most of it
fluid/diffuse/anisotropy/isosurface code that cloth does not use. The cloth-relevant kernel logic is
~2,000 lines; the rigid coupling depends on PhysX's solver-body layout and must be re-expressed against the rigid
engine's state (MuJoCo Warp `Data` here).

## 3. FEM deformable volume and surface (PhysX 5.6.1)

* Material (`include/PxDeformableVolumeMaterial.h:43`): `eCO_ROTATIONAL` (default, "well suited for high
  stiffness") or `eNEO_HOOKEAN` ("lower stiffness, robust to any tetrahedron shape"); Young's modulus, Poisson
  ratio, `elasticityDamping`, `dampingScale` (removed in 5.9: always 1), friction, density.
* Solver (`gpusimulationcontroller/src/CUDA/softBodyGM.cu`, 3,272 lines; `softBody.cu`, 2,128):
  `tetrahedronsSolveGM` (`softBodyGM.cu:926`): co-rotational = XPBD `ARAP_constraint` (`:722`, C = ‖F − R‖,
  R from a per-tet quaternion updated by `computeGMTetrahedronRotations` / `sb_gm_updateTetrahedraRotationsLaunch`
  `:418`) with α̃ = 1/(2μV·h²), then `volume_constraint` (`:756`, C = det F − 1, α̃ = 1/(λV·h²)), damping in
  XPBD form (1 + damping/h) (`compute_deltaLambdaXgradC` `:558`); Neo-Hookean = deviatoric XPBD constraint with
  α = invE/h² (`tetrahedronsSolveInnerNeoHookean` `:657`) + `volumeSolve` (`:621`). TGS: λ not accumulated
  (`isTGS ? 0 : stored`). Tets are solved by partition with per-vertex copies averaged
  (`sb_gm_cp_averageVertsLaunch` `:1509`), like the cloth springs. The simulation mesh is a voxel tet mesh
  (Isaac's `simulation_hexahedral_resolution`), the collision mesh a separate tet mesh embedded in it.
* Schedule: inside the same TGS loop (`PxgTGSCudaSolverCore.cpp:1399`, `PxgSoftBodyCore::solveTGS`
  `gpusimulationcontroller/src/PxgSoftBodyCore.cpp:2661`): step, FEM solve, rigid attachments, rigid–soft
  contacts (`sb_solveRigidSoftCollisionLaunchTGS`, rigid deltas output → two-way per TGS iteration),
  soft–soft / self contacts, particle and cloth interactions.
* FEM surface ("FEM cloth", `FEMCloth.cu` 1,356, `FEMClothUtil.cuh` 704, `FEMClothExternalSolve.cu` 1,682,
  narrow phase `femCloth*.cu` ~6,400): XPBD fixed-corotated membrane (`membraneEnergySolvePerTriangle`,
  `FEMClothUtil.cuh:436`: 2×2 ARAP with α̃ = 1/(2μ·area·thickness·h²) + area constraint with λ) and XPBD
  discrete-shell bending on shared-edge triangle pairs (`bendingEnergySolvePerTrianglePair` `:543`).
* Size: volume ≈ 5,400 kernel lines + 3,200 host (`PxgSoftBodyCore.cpp`) + narrow phase (`softbody*.cu`
  ≈ 4,600) + cooking (tet-mesh voxelisation, `physxcooking`/`PxDeformableVolumeExt`); surface ≈ 3,700 + 3,000
  host + 6,400 narrow phase. Both depend on PhysX's broadphase, contact managers, solver-body velocity readers
  and the TGS island loop exactly like the particle system.

## 4. Comparison with the existing backends

| aspect | PhysX PBD cloth | MetalSim `XPBDSim` | MuJoCo Warp flex | Newton `SolverXPBD` | Newton `SolverVBD` |
|---|---|---|---|---|---|
| stretch | implicit spring-damper on cooked stretch/shear springs | XPBD distance, compliance | edge equality (soft constraint, solref) or StVK | XPBD springs with compliance + damping (`xpbd/kernels.py:315`) | stable Neo-Hookean membrane energy, block Newton |
| bending | distance springs between straight-line neighbours | distance between flap-opposite vertices | `elastic2d` bending (flex) | dihedral angle constraint (`:381`) | dihedral energy |
| iteration scheme | 16 TGS substeps × 1 pass; 8 combined partitions with copies, blend 1/(maxCopies+1) | N substeps × 1 pass, full graph colouring (true GS) | one implicit solve per step (CG/Newton) | substeps × iterations, Jacobi + relaxation | substeps × iterations, coloured block GS |
| damping | per spring (c_d) + global velocity damping | global velocity damping | solref damping ratio | per spring k_d | k_d terms |
| contacts | once-per-step list at contact offset, accumulated normal impulse, rest offset, μ-bounded friction, relaxation 0.5, averaged | per-substep SDF projection with radius, position friction | MuJoCo soft contacts (capped, FPS) | per-iteration particle–shape contacts with restitution/friction | penalty (ke, kd) + friction |
| rigid coupling | impulse exchange every TGS iteration | `xfrc_applied` of the next MuJoCo step | same solver | same solver | AVBD rigid bodies in the same solver, or coupled solvers |

What it takes to make each algorithmically PhysX's cloth: `XPBDSim` — replace the constraint set, the solve
formula, the colouring and the contact model, i.e. a different backend (the prototype below); Newton XPBD —
same, plus Gauss–Seidel partitions instead of Jacobi; MuJoCo Warp flex and Newton VBD — different energies and
solvers, not reachable by configuration. None of the PhysX cloth features can be matched by parameter fitting
alone because the iteration count, partition blend and velocity finalisation are part of PhysX's effective
material response (a spring with k = 1e4 N/m, m = 0.02 kg at h = 5 ms/16 corrects only
k·h²·w_sum ≈ 0.098 of its error per substep, so the converged stiffness is set by the schedule).

## 5. Port plan (step 2)

### 5.1 PhysX-style PBD cloth on Warp/Metal (`metalsim/physics/physx_cloth.py`)

1. Constraint cooking in NumPy: `ExtParticleClothCooker` classes (stretch, shear with the alternate diagonal,
   straight-line bending), stiffness per class from the USD spring parameters; `NpParticleClothPreProcessor`
   partitioning (32-bit greedy, combine to 8, copy chains, accumulation slots, blend scale), reproduced
   index-for-index so that the springs meet in PhysX's order. Replace by the dumped USD springs when available.
2. Kernels (all worlds per launch, fixed sizes, graph-capturable): pre-integrate; forward projection for contact
   generation; contact generation vs plane/box/sphere/capsule with PhysX's offsets (once per step); per
   substep (= position iteration): step, gather copies, 8 partition launches, blend, update; self-collision
   (hash grid per world, later); contact solve with accumulated impulses and friction, averaged, relaxation 0.5;
   update; finalize and integrate.
3. Coupling to MuJoCo Warp rigid bodies. PhysX exchanges impulses in every TGS iteration; MuJoCo Warp takes one
   implicit step per call, so exact parity is impossible without substepping MuJoCo. Stages: (a) one-way: cloth
   reads geom poses from the MuJoCo Warp `Data` each step (static and kinematic obstacles; the recorded scenes
   are all static); (b) two-way lagged: per-step sum of contact impulses → `xfrc_applied` for the next MuJoCo
   step (what `XPBDSim` does); (c) per-substep exchange: MuJoCo Warp stepped at dt/16 with the cloth's reaction
   as `xfrc_applied` each substep (16× the rigid cost, the closest to PhysX TGS; measure before adopting). This
   mirrors Newton's `SolverCoupledProxy` "lagged" mode (Isaac Lab 3.0's choice, companion doc) at (b)/(c).
4. Validation (runs/parity/isaac/deformable/ when the recordings land): cloth scene (21×21, 1 m, over a 0.4 m
   static box, 200 Hz, 5 s): per-step particle RMS/max error vs PhysX, mean height, contact-band height on the
   box top, settling time, edge-strain distribution; controls: PhysX-vs-PhysX run-to-run spread (if the
   recorder is run twice) and our float32 perturbation spread (1e-7 m initial jitter), exactly as done for
   MuJoCo C (`deformables_2026-09-25.md` §2). Parameters taken from `meta.json`, never fitted, for the
   "same algorithm" claim; fitting is the other agent's route for the existing backends.
5. Effort (estimated): cooking + partitioning + springs + plane/box contacts + tests: 1.5 days (prototype, step
   3); self-collision grid: 1 day; mesh/heightfield one-way contacts: 1 day; two-way coupling (b) 0.5 day,
   (c) 1 day + measurement; validation against the recordings: 1 day once they exist. Total ≈ 5–6 days.

### 5.2 PhysX-style FEM volume

1. Tet meshing as PhysX does it (voxel simulation mesh at `simulation_hexahedral_resolution`, embedded collision
   mesh, `PxDeformableVolumeExt` cooking) — the largest unknown; the Isaac recordings should include node
   counts and rest positions so the port can take PhysX's own simulation mesh (the recorder already saves nodal
   positions per step; add the element indices).
2. Kernels: per-tet rotation update (polar/quaternion as `computeGMTetrahedronRotations`), XPBD ARAP + volume
   with damping, partitions with copies and averaging, TGS schedule, contacts against rigid shapes via the
   collision mesh, kinematic targets.
3. Effort (estimated): 6–9 days (kernels 3, meshing 2–4, contacts/coupling 1–2) + validation 1–2.

### 5.3 Risks

* Float-order divergence: PhysX's partition order is deterministic but CUDA's accumulation order is not; parity
  must be judged against run-to-run spread (as with MuJoCo C), not bit equality.
* Hidden cooking: omni.physx's spring cooking is closed source; without the USD dump the spring set is an
  assumption. Same for the voxel tet mesh of FEM volumes.
* Coupling: MuJoCo Warp cannot take impulses per TGS iteration without substepping; the rigid robot's own
  PhysX schedule is also 16 TGS iterations when a particle cloth is present (PhysX raises the island's iteration
  count), which our rigid path does not reproduce.
* Self-collision cost: a per-world hash grid over 441 particles is cheap, but 96-neighbour lists at 4096 worlds
  are 173 M entries; per-world grids must stay inside the world (no global sort).
* Version drift: particle cloth exists only in PhysX ≤ 5.8; Isaac Sim 6 users get FEM surface cloth. Porting
  particle cloth serves the Isaac Sim 5.1 reference, not Isaac Lab 3.0.

## 6. Prototype (step 3): `metalsim/physics/physx_cloth.py`

What it is: PhysX 5.6.1's particle-cloth algorithm for plane and box obstacles, batched over worlds, one Warp
launch per stage and world batch, fixed sizes, graph-captured on Metal. It covers everything in §2.2 except
self-collision, mesh/height-field contacts, two-way rigid coupling, aerodynamics and inflatables. Every kernel
cites the PhysX kernel it ports. Springs come from `cook_springs` (the `ExtParticleClothCooker` rules), or from an
explicit list (the USD springs once dumped). `partition_springs` reproduces `NpParticleClothPreProcessor`: 32-bit
greedy partitions, combined to 8, copy chains, accumulation slots, blend 1/(max copies + 1). On the recorder's
21×21 cloth this gives 2,438 springs (840 stretch, 800 shear, 798 bend), 878 accumulation copies and blend 1/3.
`mesh_from_flex` takes the topology from a MuJoCo flex, so it matches `XPBDSim` and MuJoCo Warp flex exactly.

Tests (`tests/test_physx_cloth.py`, 7/7 on Metal, `runs/physx_cloth/`). PhysX ships no numeric reference: the
snippets print nothing and the repository has no tests. So the tests check PhysX's rules, PhysX's semantics,
self-consistency, and `XPBDSim`:

| test | what | result |
|---|---|---|
| cooker = SnippetPBDCloth | the cooker rules on the snippet's own mesh give exactly its hand-written stretch + shear springs (and 46 straight-line bend springs on 7×5) | pass |
| partition invariants | 8 combined partitions; copy chains go forward and stay on one particle; bijection onto chain slots + accumulation slots; blend = 1/(max copies + 1) | pass |
| rest/contact offsets | recorder parameters (rest 0.025, contact 0.0375 m): after 3 s the floor-touching particles sit at 0.0250 m (none below), box-top median at 0.425 m, at rest | pass |
| contact gate | a contact exists iff the step-start distance ≤ particle + shape contact offset | pass |
| hanging cloth vs XPBDSim | same flex topology, pinned top row, 3 s: PhysX-style bottom 0.485 m, mean 0.9905 m, max stretch strain 2.9 %; XPBDSim (same compliances) 0.495 m / 0.998 m / 0.9 % | within 2 cm (PhysX's scheme is softer: its per-substep implicit springs correct only k·h²·w ≈ 10 % of the error and the blend averages copies with 1/3) |
| dropped cloth vs XPBDSim | centre on the box top in both; mean height 0.206 vs 0.237 m | 3.1 cm (bound 4 cm) |
| Metal = CPU, graph = eager | 40 steps through contact onset | graph bit-identical; Metal vs CPU 1.9e-4 m (float32 operation order) |

Throughput (recorder cloth scene, dt 5 ms, 16 position + 1 velocity iterations, graph replay;
`runs/physx_cloth/metal_run.log`):

| worlds | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| PhysX-style PBD cloth, env-steps/s | 36.1K | 63.8K | 57.7K | 41.3K |
| MetalSim `XPBDSim`, same topology, 16 substeps | – | – | 320K | 285K |
| Newton VBD cloth (10 substeps × 10 iterations; companion doc) | 1.8K | 3.5K | 3.6K | 3.7K |

The prototype does 5.5× XPBDSim's work per substep:
* 2,438 springs vs XPBDSim's 1,240 constraints;
* copy gather/scatter and blend;
* 8 partition launches plus 6 other launches per iteration, and a 17th velocity iteration.

At 4096 worlds its copy buffers are 4096 × 5,754 × 12 B = 283 MB per array (positions and velocities), which
explains the drop from 1024.

What is still needed before a parity claim against the recordings (the other agent's
`runs/parity/isaac/deformable/`): the USD springs from Isaac's cooker, self-collision (the recorder enables it),
and a check of the velocity-iteration count PhysX used. Next steps and estimates are in §5.1.

### 6.1 Decisions (archived options, DECISIONS.md style; not yet copied into `docs/DECISIONS.md`)

| date | decision | options considered (with numbers) | chosen and why | how to re-enable the others |
|---|---|---|---|---|
| 2026-09-25 | PhysX cloth prototype target | particle cloth (PhysX ≤ 5.8, Isaac Sim 5.1: the recorded reference) vs FEM surface cloth (PhysX ≥ 5.9, Isaac Sim 6 / Isaac Lab 3.0 PhysX backend) | particle cloth: the only PhysX cloth with a recording on the way | FEM surface: §3 and §5 (not started) |
| 2026-09-25 | Spring set | PhysX `ExtParticleClothCooker` rules (open source) vs omni.physx's cooker (closed; results stored on the USD prim) | cooker rules for now; USD springs when dumped (`PhysXClothSim(springs=...)`) | pass `springs=` |
| 2026-09-25 | Edge order in the cooker | PhysX hash-set bucket order vs sorted edges | sorted (same spring set, partition assignment can differ) | not implemented |
| 2026-09-25 | Rigid coupling of the prototype | one-way (obstacle poses read from arrays / MuJoCo Warp `geom_xpos`) vs lagged two-way vs per-substep two-way | one-way (all recorded obstacles are static) | §5.1 item 3 |
| 2026-09-25 | Deformable engine priority | PhysX-port prototype vs Newton VBD + coupled solvers (Isaac Lab 3.0's path; runs on Metal, companion doc) | VBD first (coordinator, 2026-09-25); the PhysX prototype kept as the PhysX-backend reference | `metalsim.physics.physx_cloth` |
