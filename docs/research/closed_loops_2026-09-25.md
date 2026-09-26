# Closed-loop mechanisms on MetalSim: Newton Kamino on Metal vs MuJoCo Warp soft loop closure

2026-09-25, low priority. Scope: kinematic loops (four-bar linkages, parallel-linkage legs like Digit's, coupled
hands) ahead of a user robot that may have them. **Measured** = run here (M4 Max); Metal and every Warp/Newton
run (including Warp's CPU device) went through `scripts/gpu_run.sh NAME low MIN` with `gpu_lock.py status` at the
top of each log; pure MuJoCo C / numpy runs ran directly. **Source** = read in the cited file or page.

Scripts: `scripts/diagnostics/closed_loops/` (`mechanisms.py` defines the test mechanisms once and emits MJCF, a
Newton builder and an exact planar reference; `mjw_loops.py` MuJoCo C / MuJoCo Warp; `kamino_probe.py`
SolverKamino; `ref_traj.py`; `analyze.py`; `c_sweep.sh`, `kamino_jobs.sh`, `mjw_jobs.sh`, `run_kamino_tests.py`).
Data and logs: `runs/closed_loops/`.

<!-- SUMMARY -->

## 1. Does the user robot have loops?

No, not in its simulation model. `metalsim/tron1/sim.py` builds LimX's `WF_TRON1A` MJCF
(`assets/tron1/WF_TRON1A/xml/robot.xml`): `neq = 0`, 9 joints (free + abad/hip/knee/wheel per side), 11 bodies in a
pure tree (measured: `build_model(SimParams(payload_mass=0))`). The URDF (`urdf/robot.urdf`) models each knee as a
single revolute joint with a `SimpleTransmission`, reduction 1. Whether the hardware drives the knee through a
rod from a hip-mounted motor could not be established from a primary LimX source (spec page lists three 48 V
actuators per leg; no mechanism description): if it does, LimX's own model already reduces it to a direct knee,
which is how the controller sees it (`knee_*_Joint` in the SDK's motor order).

## 2. Research

### 2.1 Newton's SolverKamino (Disney Research)

Sources: [K] `upstream/newton/newton/_src/solvers/kamino/` (Newton fork 1.7.0.dev, 90e23324) and
`upstream/newton-1.5.2/...` (tag v1.5.2); [TR] Tsounis, Grandia, Bächer, "On Solving the Dynamics of Constrained
Rigid Multi-Body Systems with Kinematic Loops", arXiv:2504.19771 (v1, 2025-04-28); [IL3] Isaac Lab `v3.0.0-EA`
sparse clone `upstream/IsaacLab3`.

* **Status.** "`SolverKamino` is currently in BETA (`BETA 1`) ... we discourage users of Newton from depending on
  it ... A more stable `BETA 2` version is planned for release during the summer of 2026" ([K] README.md).
  Developed by Disney Research with NVIDIA and Google DeepMind (README). Disney's project page
  (`disneyresearch.github.io/kamino/`) returned 404.
* **Formulation.** Maximal coordinates (per-body CoM pose and twist) [TR §IV-A]. Forward dynamics posed as a
  nonlinear complementarity problem over bilateral joint constraints, bounded-multiplier (effort) constraints and
  unilateral joint-limit and contact constraints (`solver_kamino.py` class docstring). In [TR]: Delassus matrix
  D = J M⁻¹ Jᵀ (Eq. 11), post-event velocity v⁺ = Dλ + v_f (Eq. 13), De Saxcé-augmented velocity (Eq. 14), NCP
  K* ∋ v̂(λ) ⊥ λ ∈ K (Eq. 15); contacts are Signorini + Coulomb cone with restitution (Eqs. 40b, 45b).
  Constraints are **hard**: loop-closing joints are ordinary bilateral joint rows ("closes the loop" joint outside
  the articulation, `newton/tests/utils/basics.py` `build_boxes_fourbar`), rank deficiency from redundant loop rows
  handled by the proximal regularisation.
* **Solver.** Default backend Proximal-ADMM ("realizes the Proximal-ADMM algorithm described in [TR] and is based on
  the work of J. Carpentier et al." (arXiv:2405.17020), diagonal preconditioning, Nesterov acceleration with the
  Goldstein et al. fast-ADMM restart, optional residual-balancing penalty updates; `_src/solvers/padmm/__init__.py`).
  Defaults: 200 iterations, primal/dual/complementarity tolerance 1e-6, warm start from containers
  (`config.py` `PADMMSolverConfig`). Opt-in `"dvi"` backend: projected iterations with a direct bilateral block
  solve (`solver_kamino.py` `Config.dynamics_solver`). The iteration loop is a `wp.capture_while` on
  `state.done` when `use_graph_conditionals` (default True), else an unrolled loop of max iterations
  (`padmm/solver.py:414`).
* **Stabilisation.** Baumgarte per constraint class: α = 0.01 bilateral joints, β = 0.01 joint limits, γ = 0.01
  contacts, contact margin δ = 1e-6 m (`config.py:225-258`); overridable per constraint. Isaac Lab 3.0 overrides α to
  **0.1** (`isaaclab_newton/physics/kamino_manager_cfg.py` `KaminoConstraintsCfg`).
* **Time integration.** `"euler"` (semi-implicit, default) or `"moreau"` (Moreau–Jean midpoint; needs Kamino's
  collision detector) (`solver_kamino.py` `Config.integrator`; `_src/integrators/`).
* **Drives.** Effort (explicit `joint_f`) and implicit PD (`POSITION_VELOCITY` target mode with `target_ke/kd`,
  armature, viscous damping as "dynamic" constraint rows; `_src/core/conversions.py`), joint Coulomb friction and
  effort limits (Newton v1.6.0 release notes).
* **Limits (beta 1).** Refuses particles, springs, triangles/edges/tets, muscles, `DISTANCE` and `ROD` joints, and
  bodies with singular inertia (`solver_kamino.py` `_validate_model_compatibility`); immovability decided once at
  construction; several joint-property changes require recreating the solver (`notify_model_changed`).
* **Isaac Lab 3.0.** Ships a `newton_kamino` physics preset (`KaminoPADMMSolverCfg`: 100 iterations, tolerances
  1e-4, ρ₀ 0.05, α 0.1, integrator `moreau`, `use_graph_conditionals=False`), a Kamino-only four-bar task
  `Isaac-Fourbar-Pole-Swingup` (dt 1/120 s, `integrator="euler"`, ρ₀ 0.1; "the closed four-bar kinematic loop is
  resolved as an equality constraint by kamino", `isaaclab_tasks/core/fourbar_pole/fourbar_pole_manager_env_cfg.py`)
  and the Disney DR Legs asset ("parallel-linkage bipedal lower body with 30 joints (12 actuated + 18 passive
  linkage DOFs) plus 6 loop-closing joints", `isaaclab_assets/robots/dr_legs.py`).

### 2.2 MuJoCo's loop closure

* **Mechanism.** Equality constraints over a spanning tree: `connect` (3 rows, ball joint at a point), `weld`
  (6 rows, `relpose`), `joint` (polynomial coupling of two scalar joints). For `connect`, "the position of body2's
  anchor is determined such that it coincides with body1's anchor in the initial configuration (qpos0)" (MuJoCo
  XML reference, equality/connect), so the model must be assembled closed at qpos0 (ours are, to 5e-16 m).
* **Softness.** Every constraint is soft. Reference acceleration a_ref = −b v − k r with b = 2/(d_max·timeconst),
  k = d(r)/(d_max² timeconst² dampratio²); regulariser R = (1−d)/d · Â (MuJoCo documentation, Computation >
  Parameters). Defaults solref "0.02 1", solimp "0.9 0.95 0.001 0.5 2"; with `refsafe`, timeconst is clamped to
  ≥ 2·timestep. So a loop closure is a critically damped spring-like constraint with 20 ms time constant and
  impedance 0.9–0.95 unless tuned.
* **MuJoCo's own guidance.** "equality constraints can be used to create 'loop joints' ... Gaming engines represent
  all joints this way. The same can be done in MuJoCo but is not recommended – because it leads to both slower and
  less accurate simulation" (Computation > Equality). That sentence is about replacing tree joints; loops that
  cannot be in the tree have no other route in MuJoCo. Menagerie's Agility Cassie closes its achilles-rod and
  plantar-rod loops with four `connect` equalities whose default class stiffens them to solref "0.005 1" (`upstream/mujoco_menagerie/agility_cassie/cassie.xml:10, 228-233`): the model author chose a 5 ms time constant (4x stiffer than the default) at a 0.5 ms timestep (line 4).
* **MuJoCo Warp.** `EqType` implements CONNECT, WELD, JOINT, TENDON, FLEX, FLEXSTRAIN; DISTANCE unsupported
  (`upstream/mujoco_warp/mujoco_warp/_src/types.py:729-747`); kernels `_equality_connect` / `_equality_weld` in
  `constraint.py`. No Metal-specific code path for equality rows in the fork; the only Metal caveat is general
  (graph conditionals off, `m.opt.graph_conditional = False`, so all solver iterations launch;
  `docs/research/mjwarp_throughput_2026-09-25.md`).

### 2.3 Isaac Lab 3.0 / PhysX and Digit

* Isaac Lab 3.0 EA release notes: "Closed-loop Digit articulations remain PhysX-only." Known-issues page:
  "Closed-loop articulations are not available on Newton (e.g. Agility Digit) ... **Affects:** `physics=newton_mjwarp`,
  `physics=newton_kamino`. Robots whose USD encodes a closed kinematic loop — such as the achilles rod and toe
  push-rods on the Agility Digit — are not currently validated on the Newton backends. The Digit-based contrib tasks
  are PhysX-only and do not expose a Newton preset at all" (`upstream/IsaacLab3/docs/source/refs/issues.rst:119-136`).
  The Python config (`isaaclab_assets/robots/agility.py`, `DIGIT_V4_CFG`) has no loop handling: the loop lives in
  the USD.
* PhysX: "Articulations must have a tree structure ... it is possible to create loops in the articulation by adding
  rigid-body Joints between articulation links" (PhysX 5.6 docs, Articulations > Closing loops). In USD the loop
  joint carries `physics:excludeFromArticulation` (UsdPhysics proposal, "a joint in the loop may use its
  excludeFromArticulation attribute flag ... and at this point the loop is then broken"). The loop joint is a
  maximal-coordinate constraint solved by the same PGS/TGS iterations as contacts, i.e. iterative and not exactly
  closed; the reduced-coordinate tree itself has "zero joint error by design".
* So on the Isaac side: Digit = PhysX tree + iterative loop joints; Newton/MuJoCo-Warp = unsupported for Digit;
  Newton/Kamino = supported for loops (DR Legs, four-bar task) but Digit not validated.

<!-- MEASUREMENTS -->
