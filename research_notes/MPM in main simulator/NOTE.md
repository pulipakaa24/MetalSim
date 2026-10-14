# MPM in MetalSim's main simulator: research and design (2026-10-13)

Question: how to put MPM material (sand, snow, viscous, ...) into a scene whose robot is simulated by MetalSim's own
stack (`BatchSim`: MuJoCo Warp fork on Metal, `.venv` with MuJoCo 3.14 and the Newton 1.7 line), two-way coupled, at N
worlds, without losing the Isaac-parity results of the Newton 1.5.2 path (`.venv-newton152`, `metalsim/physics/mpm/MPM.md`).

Two research rounds feed this note: (1) Isaac Lab 3.0-EA / Newton 1.5.2 from source (local checkouts:
`upstream/IsaacLab3`, `upstream/newton-metalsim-1.5.2`, `.venv`'s Newton 1.7), (2) other systems (web + source:
Genesis `efc81f3b`, Newton main `dcd9106a`, ManiSkill2, Drake, MuJoCo docs). Citations are file:line in those trees
(`F` = `upstream/newton-metalsim-1.5.2/newton/_src`, `IL` = `upstream/IsaacLab3/source`).

## 1. How Isaac Lab 3.0-EA couples MPM to MuJoCo Warp (verified from source)

Isaac never calls `setup_collider` or computes wrenches itself: `NewtonCouplerManager` builds a `SolverCoupledProxy`
(`IL/isaaclab_contrib/.../coupling/coupler.py:383-400`) over `ModelView`s. Franka-Pour (`pour_env_cfg.py:597-677`):
entry `arm` = `SolverMuJoCo` (3 substeps; robot, table, cups, statics), entry `media` = `SolverImplicitMPM` (2
substeps, in place; particles + spill floor), one proxy mapping arm -> media for **both cups only** (the arm links and
fingers are not MPM colliders), `mass_scale=1000`, `mode="staggered"`, 1 iteration; sim dt 1/120 with 2 coupled steps.

* **Proxy bodies.** Selected source bodies appear in the media view with `BodyFlags.PROXY`
  (`F/solvers/coupled/solver_coupled.py:600-609`); other shapes lose `COLLIDE_PARTICLES` (`:805-838`). The MPM's default
  collider discovery then makes one collider (merged per-body mesh) per proxy body plus the static shapes
  (`implicit_mpm/implicit_mpm_model.py:432-907`).
* **Effective mass.** `SolverMuJoCo.coupling_eval_effective_mass_block` (`F/solvers/mujoco/solver_mujoco.py:4211-4259`,
  `kernels.py:321-382`): `m_eff = 1 / (invweight0[0] + 2/3 invweight0[1] |r|^2)` with r = 0, i.e. MuJoCo's
  `body_invweight0` (mean inverse inertia at **qpos0**: ignores configuration and grasp); inertia scaled by
  `max(trace(I^-1)/3 / inv_rot, 1)`; then x `mass_scale` (`solver_coupled_proxy.py:1262-1285`). Kinematic bodies get
  mass 0 (an infinitely stiff collider; `IL/.../physics/mpm_manager.py:115-121`).
* **Per coupled step** (`solver_coupled_proxy.py:1369-1569`): arm `body_f` = user `body_f` + last harvested wrenches,
  held constant over the arm's substeps (`solver_coupled.py:2692-2702`); arm steps; proxy sync - staggered: arm's
  **end** pose and end velocity into the media state, lagged: begin pose + end velocity and a velocity rewind
  `v -= dt M_proxy^-1 F` (`implicit_mpm/solver_implicit_mpm.py:1757-1783`); MPM steps in place (both substeps see the
  same synced collider pose; `collider_velocity_mode="forward"`: `v + w x r` at the contact point); harvest
  `coupling_harvest_proxy_wrenches` (`:1785-1830`): `F = impulse / dt_outer`, torque about the world COM, fixed-order
  lane sums on Metal (fork commit `6d746225`).
* **Inside the MPM solve** the proxy is a *compliant collider*: finite mass + inertia enter a Delassus operator
  `J M^-1 J^T` in the rheology iterations (`rasterized_collisions.py:516-567, 792-879`), so the sand sees the body's
  inertia implicitly within the step; feedback to MuJoCo is lagged by one coupled step.
* **Upstream quirks Isaac users get** (reproduced, not fixed, by default): only the last MPM substep's impulse field is
  harvested, divided by the outer dt (about half the mean force with 2 substeps; MPM.md 8i); `invweight0` is a qpos0
  value; with `mass_scale != 1` the lagged rewind removes 1/mass_scale of the integrated velocity change.
* **Feedback into MuJoCo**: `SolverMuJoCo._apply_mjc_control` copies `body_f` (world-frame wrench about the COM) into
  `xfrc_applied` (`solver_mujoco.py:4743-4762`), which MuJoCo applies at `xipos` through `J^T f`
  (`mj_xfrcAccumulate`). Kinematic bodies are zeroed.
* **Materials / emitters** (`IL/isaaclab_newton/.../sim/spawners/mpm/`): `MPMParticleMaterialCfg` (density 1000, E 1e15,
  nu 0.3, friction 0.68, yield pressure 1e12, rest 0) as `mpm:*` custom attributes; `MPMGridCfg` (box, voxel, particles
  per cell, `cell_center` / `boundary`, jitter) and `MPMPointsCfg`; one emission per world at the env pose, equal
  counts per world, contiguous by world (`mpm_object.py:53-112`). Solver defaults `MPMSolverCfg` (voxel 0.1, sparse
  grid, APIC, P0 strain, Q1 velocity, S2 colliders, 250 iterations, tolerance 1e-4, `separate_worlds=False`).

## 2. Other systems

| system | MPM | coupling to rigid | determinism / batching |
|---|---|---|---|
| Genesis (`couplers/legacy_coupler.py:128-465`) | explicit MLS-MPM | rigid steps first; grid nodes inside a geom SDF get a PlasticineLab-style velocity projection (softness, friction, restitution), `-dm v / dt` accumulated as a wrench applied in the **next** rigid substep: lagged explicit exchange, rigid = infinite mass inside MPM | dense grid per env; P2G and wrench accumulation by float atomics (not bitwise) |
| ManiSkill2 (arXiv 2302.04659) | Warp MLS-MPM | rigid SDFs; particle-level penalty forces per MPM substep, summed to the body and applied after the soft steps | per-env |
| Newton `SolverCoupledProxy` | implicit MPM | proxy with effective mass (above); design note (G. Daviet, 2026-04-23, linked from newton PR #2848): lagged iteration contracts iff `r = (1-s)/(1+qs)` has |r| < 1 with `s = M_proxy / M_eff`, `q = M_eff / M_other`; s in [0.5, 1] recommended | `separate_worlds`: one NanoVDB, per-world FEM environments |
| Newton examples | implicit MPM | `twoway_coupling`: body mass as collider mass, wrench lagged into `body_f`, velocity rewind before the MPM step; `anymal`: kinematic robot, one-way | |
| Drake (arXiv 2403.13783, 2503.05046) | convex MPM | monolithic convex contact with articulated rigid bodies; async GPU variant | |
| MuJoCo / MuJoCo Warp | none (flex is a Lagrangian mesh, no plasticity; mujoco_warp#232 deferred MPM to Newton) | `xfrc_applied` is the documented injection point | |
| PhysX | PBD particles; MPM / FLIP removed in 5.4 | | |

Stability (Newton design note; added-mass literature arXiv 1208.3511): a kinematic collider with lagged explicit
feedback is s -> infinity, r -> -M_mpm / M_link: it oscillates and diverges once the sand engaged per step outweighs a
light link (feet, fingers). The proxy with effective mass (s ~ 1) is the stable explicit scheme; monolithic solves
(Drake) are the unconditional ones but would mean rewriting MuJoCo Warp's constraint solver.

Determinism: MetalSim's fork already makes the implicit MPM bitwise repeatable on Metal at 2,048 worlds (MPM.md 7b, 8b,
8g) and the proxy harvest fixed-order; Warp's deterministic mode is CUDA/CUB-based upstream but implemented on Metal in
the fork.

## 3. Options for MetalSim

| | what | parity with Isaac / the Metal results | one environment | maintenance |
|---|---|---|---|---|
| (a1) | install Newton 1.5.2 next to 1.7 in `.venv` | - | impossible: same package name, and MuJoCo 3.11 vs 3.14 for the two MuJoCo Warp builds | - |
| (a2) | port the fork's MPM commits (and parity work) onto Newton 1.7's `implicit_mpm` | 1.7's MPM differs from 1.5.2's (reset from authored Jp, cylinder `barrel_radius`, new surface code, rewritten parts of `solve_rheology`): the L4 floors would have to be re-established against a solver Isaac Lab 3.0-EA does not ship | yes | high: every fork change twice; changes `.venv`'s Newton pin, which the Newton-engine G1 task also uses |
| **(b)** | **vendor the fork's `implicit_mpm` package (pinned `3fd1f052`, the commit `setup_newton152.sh` installs) into `metalsim/physics/mpm/implicit_mpm/`, generated by a script that only rewrites its six package-external imports; run it against `.venv`'s Newton 1.7 types; MetalSim coupler with `BatchSim` as the rigid entry** | kernels byte-identical to the fork: checkable bitwise against `.venv-newton152` on the same inputs | yes | low: regenerate on re-pin; a test checks the copy equals the pin |
| (c) | own explicit MPM (Genesis / ManiSkill lineage) | statistical at best (DECISIONS 2026-10-09 already rejected it) | yes | high |
| (d) | drive `.venv-newton152`'s solver from another process sharing GPU buffers | bitwise | no | fragile IPC, two Python processes per scene |

Compatibility facts behind (b) (checked by import and by comparison, CPU): every `newton` symbol the 1.5.2 package uses
exists in 1.7 with the same values (`BodyFlags.PROXY` 4, `ParticleFlags`, `ShapeFlags.COLLIDE_PARTICLES`,
`Model.AttributeFrequency/Assignment`, `ModelBuilder.CustomAttribute`, `core.reset.reset_world_selected` identical body,
`SolverBase`, `CouplingInterface` with all 11 hook signatures identical); the collider mesh helpers
(`Mesh.create_sphere/capsule/cylinder/cone/box/plane`) give bitwise-identical vertices and indices in 1.5.2 and 1.7 for
the arguments the MPM passes; the coupled-solver files (`interface.py`, `solver_coupled_proxy.py`, `proxy_utils.py`)
are identical across stock 1.5.2, the fork and 1.7. Remaining dependencies are on the shared Warp fork.

## 4. Recommendation and design (b)

1. **Solver**: the vendored fork copy (`scripts/vendor_mpm152.py`; `metalsim.physics.mpm.solver.solver_class()` picks
   Newton's own on the 1.5.2 line and the copy elsewhere). `newton_compat.install` patches both copies (CUDA stopping
   rule, float32 accumulation, chunk graph / GPU exit); in the main simulator it is called with
   `rigid_deterministic=None, device_args=None` so it changes nothing MuJoCo Warp compiles.
2. **Coupling = Isaac's `SolverCoupledProxy` semantics with `BatchSim` as the source entry**, built from the solver's
   own coupling hooks so the MPM side runs Isaac's code:
   * media model (Newton 1.7 `ModelBuilder`, N worlds, `separate_worlds=True`): particles from emitters, plus one
     proxy body (`BodyFlags.PROXY`) per selected MuJoCo body and its collision geoms as shapes (box, sphere, capsule,
     cylinder, plane, mesh; ellipsoid as a mesh) and the static geoms per world (`metalsim/physics/mpm/colliders.py`);
   * proxy mass: `effective` (default; Isaac's `body_invweight0` formula from MuJoCo Warp's model, x `mass_scale`,
     default 1), `body` (link mass, as `example_mpm_twoway_coupling`) or `kinematic` (0: one-way, as the ANYmal
     example); per body overridable;
   * per `BatchSim.step()` (one coupled step, dt = substeps x timestep): `xfrc_applied` = user wrench + last harvested
     wrench for the MuJoCo substeps (user part restored afterwards, so a task writing `xfrc_applied` keeps its meaning);
     then a kernel writes the proxies' `body_q` / `body_qd` from MuJoCo Warp's `xpos`, `xquat`, `xipos`, `cvel`,
     `subtree_com` (staggered: end of step; lagged: begin pose + rewind); `mpm_substeps` MPM steps; harvest with the
     solver's ordered `coupling_harvest_proxy_wrenches` into a per-(world, body) wrench buffer. All of it Warp launches,
     captured in BatchSim's step graph (new pre/post hook lists that are empty without MPM, so a scene without MPM
     launches exactly what it launched before).
   * Isaac's quirks are the default (`harvest="last_substep"`); `harvest="mean"` sums all substeps' impulses as a
     documented option.
3. **API** (`metalsim.physics.mpm.scene`): `MPMMaterial` (Isaac's `MPMParticleMaterialCfg` fields), emitters
   `GridEmitter` (Isaac's `_emit_grid`, float32 corners, `cell_center` / `boundary`, jitter) and `PointsEmitter`,
   `MPMConfig` (Isaac's `MPMSolverCfg` defaults), `MPMCoupling(sim, ...)`, `add_emitter`, `add_proxy_bodies`,
   `finalize()`; `particle_q` (zero-copy Warp array, `(N*P, 3)`, world-major) for `Tier0/Tier2Renderer.set_particles`;
   `reset(mask)` per world; `check_status()` (grid overflow).
4. **Tests**: vendored copy = pin; vendored solver bitwise = Newton's in `.venv-newton152` and in `.venv` (pure MPM
   column); box sinking into a bed vs the same scene through Newton's `SolverCoupledProxy` + `SolverMuJoCo` in
   `.venv-newton152` (within the two-way floors); momentum / energy bookkeeping; angle of repose of a pile; determinism
   (two runs bitwise at 1 / 64 / 1,024 worlds); rigid neutrality (a scene without MPM: same launches, same results).
