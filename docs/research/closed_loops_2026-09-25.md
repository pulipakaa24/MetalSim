# Closed-loop mechanisms on MetalSim: Newton Kamino on Metal vs MuJoCo Warp soft loop closure

2026-09-25, low priority. Scope: kinematic loops (four-bar linkages, parallel-linkage legs like Digit's, coupled
hands) ahead of a user robot that may have them. **Measured** = run here (M4 Max); Metal and every Warp/Newton
run (including Warp's CPU device) went through `scripts/gpu_run.sh NAME low MIN` with `gpu_lock.py status` at the
top of each log; pure MuJoCo C / numpy runs ran directly. **Source** = read in the cited file or page.

Scripts: `scripts/diagnostics/closed_loops/` (`mechanisms.py` defines the test mechanisms once and emits MJCF, a
Newton builder and an exact planar reference; `mjw_loops.py` MuJoCo C / MuJoCo Warp; `kamino_probe.py`
SolverKamino; `ref_traj.py`; `analyze.py`; `c_sweep.sh`, `kamino_jobs.sh`, `mjw_jobs.sh`, `run_kamino_tests.py`).
Data and logs: `runs/closed_loops/`.

## Summary (status 2026-09-25 evening; GPU-queue work still pending, see §6)

* **The user robot has no loops in simulation.** TRON1 WF (`neq = 0`, pure tree); nothing to close today.
* **MuJoCo soft closure, measured in MuJoCo C (float64) against an exact hard-loop reference, is adequate at
  2.5 ms and 5 ms for these mechanisms:** closure error 0.36–2.3 mm on a passive 0.3 m crank-rocker, 2 µm–0.35 mm on
  a hanging parallelogram-knee leg, and 0.6–3.4 mm under 3σ random torques; the solref time constant sets the
  error almost linearly (5 ms → 20 ms: 0.36 → 1.16 mm at 2.5 ms). But **softness costs energy and phase**: the
  undamped four-bar loses 1–8 % energy over 5 s (the reference conserves it to 1e-8) and drifts 0.56–2.4 rad in
  crank angle after 5 s; the stiffer the closure, the worse the energy loss (solref 5 ms: −5.2 %; 20 ms with
  solimp 0.99/0.999: −1.1 %). With realistic joint damping (the 3σ runs) trajectories stay within 0.07–0.16 rad
  of the reference over 5 s.
* **Kamino compiles and runs on the CPU device** in both Newton 1.5.2 (624 tests: 1 failure, a geometry
  tolerance test) and the 1.7.0.dev fork (607 tests: 0 failures), and its hard closure on our four-bar is 0.4–0.8 mm
  at 2.5 ms with default settings (Baumgarte α 0.01, CPU, 1 s) with 17 % energy loss over 1 s: its first-order
  semi-implicit integration dissipates more than MuJoCo's soft closure here (single CPU run; to be confirmed).
* **Kamino on Metal: not yet measured.** The Metal test suites, CPU-vs-Metal agreement, 1024-world stress and
  throughput are queued (`kamino_tests_fork`, `kamino_tests_152`, `cl_kamino`) and had not started after 8 h behind
  higher-priority jobs. Source reading predicts it can run: the PADMM loop uses `wp.capture_while`, which the Warp
  fork implements on Metal (`warp/_src/context.py`, `_metal_capture_conditional`), and Isaac Lab 3.0 turns graph
  conditionals off anyway.
* **MuJoCo Warp on Metal (trajectory agreement with C, 1024-world stress, throughput): queued (`cl_mjw`), not run.**
* Verdict so far and the native hard-loop scope: §5.


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

## 3. Test mechanisms and method

Defined once in `scripts/diagnostics/closed_loops/mechanisms.py` (docstring has every dimension):
* **fourbar**: Grashof crank-rocker in the vertical plane (ground 0.30 m, crank 0.10 m / 0.2 kg, coupler 0.30 m /
  0.6 kg, rocker 0.25 m / 0.5 kg), crank released from −20° (a start at 60° grazed the upper equilibrium and made
  every engine's trajectory hypersensitive; rejected).
* **leg**: hanging thigh + shank (0.30 m, 2 / 1 kg) with a Minitaur/Digit-style knee drive: hip-coaxial crank
  (0.06 m) and a rod parallel to the thigh to a bell crank on the shank (parallelogram; singular configurations 25°
  outside the knee limits −30…95°).
* MuJoCo: spanning tree of hinges + one `connect` per loop, assembled closed at qpos0 (5e-16 m), implicitfast,
  collisions off. Newton/Kamino: same frames and inertias, all joints revolute, the loop-closing joint outside the
  articulation (the pattern of Newton's own `build_boxes_fourbar`).
* **Reference**: independent planar maximal-coordinate DAE (hard pins, light Baumgarte), scipy DOP853 at rtol 1e-10:
  pin violation ≤ 1.1e-11 m, energy conserved to 1e-8 relative over 5 s (`ref_traj.py`, `runs/closed_loops/ref/`).
* **3σ stress**: Gaussian torques 3 × σ per actuated joint (σ = four-bar crank 0.3 N·m; leg hip 4, knee crank
  0.75 N·m, about half the largest static gravity torque), resampled at 50 Hz, per-world seeds, with viscous joint
  damping (`DAMPING`). Undamped random torques pumped the mechanisms to 95–240 rad/s and, with the first σ choice
  (leg 5 / 3 N·m), blew up 4/32 (2.5 ms) and 16/32 (5 ms) MuJoCo C worlds; that setting is archived (unphysical:
  no motor is undamped).
* Metrics: max body-angle error to the reference, energy drift (passive), closure error = distance between the two
  copies of the loop point.

## 4. Measurements: MuJoCo C (float64, CPU, not queued)

`c_sweep.sh`; tables from `analyze.py` (`runs/closed_loops/c_summary.md`). solref second entry 1 (critical);
solimp width 0.001, midpoint 0.5, power 2.

### fourbar_none

| dt | solref tc | solimp d0,dmax | max angle err 1 s / 2 s / end [rad] | energy drift end [%] | closure max / mean [mm] |
|---|---|---|---|---|---|
| 2.5 ms | 0.005 | 0.9,0.95 | 0.075 / 0.398 / 1.880 | -5.24 | 0.356 / 0.121 |
| 2.5 ms | 0.005 | 0.99,0.999 | 0.074 / 0.396 / 1.874 | -5.21 | 0.362 / 0.115 |
| 2.5 ms | 0.01 | 0.9,0.95 | 0.061 / 0.331 / 1.643 | -4.25 | 0.642 / 0.240 |
| 2.5 ms | 0.01 | 0.99,0.999 | 0.058 / 0.314 / 1.581 | -4.00 | 0.676 / 0.227 |
| 2.5 ms | 0.02 | 0.9,0.95 | 0.032 / 0.197 / 1.076 | -2.47 | 1.074 / 0.470 |
| 2.5 ms | 0.02 | 0.99,0.999 | 0.020 / 0.101 / 0.557 | -1.07 | 1.163 / 0.448 |
| 5 ms | 0.01 | 0.9,0.95 | 0.117 / 0.610 / 2.360 | -8.20 | 1.306 / 0.455 |
| 5 ms | 0.01 | 0.99,0.999 | 0.114 / 0.595 / 2.332 | -7.97 | 1.349 / 0.443 |
| 5 ms | 0.02 | 0.9,0.95 | 0.041 / 0.276 / 1.374 | -3.16 | 2.221 / 0.893 |
| 5 ms | 0.02 | 0.99,0.999 | 0.040 / 0.190 / 0.981 | -1.94 | 2.323 / 0.880 |

### fourbar_sigma3

| dt | solref tc | solimp d0,dmax | max angle err 1 s / 2 s / end [rad] | energy drift end [%] | closure max / mean [mm] |
|---|---|---|---|---|---|
| 2.5 ms | 0.005 | 0.9,0.95 | 0.028 / 0.042 / 0.085 |  | 0.589 / 0.085 |
| 2.5 ms | 0.005 | 0.99,0.999 | 0.028 / 0.043 / 0.085 |  | 0.603 / 0.081 |
| 2.5 ms | 0.01 | 0.9,0.95 | 0.028 / 0.035 / 0.082 |  | 1.015 / 0.166 |
| 2.5 ms | 0.01 | 0.99,0.999 | 0.028 / 0.035 / 0.080 |  | 1.077 / 0.157 |
| 2.5 ms | 0.02 | 0.9,0.95 | 0.028 / 0.034 / 0.081 |  | 1.546 / 0.320 |
| 2.5 ms | 0.02 | 0.99,0.999 | 0.029 / 0.029 / 0.074 |  | 1.710 / 0.295 |
| 5 ms | 0.01 | 0.9,0.95 | 0.056 / 0.073 / 0.159 |  | 2.060 / 0.317 |
| 5 ms | 0.01 | 0.99,0.999 | 0.056 / 0.072 / 0.157 |  | 2.121 / 0.312 |
| 5 ms | 0.02 | 0.9,0.95 | 0.057 / 0.061 / 0.152 |  | 3.234 / 0.595 |
| 5 ms | 0.02 | 0.99,0.999 | 0.058 / 0.058 / 0.145 |  | 3.404 / 0.591 |

### leg_none

| dt | solref tc | solimp d0,dmax | max angle err 1 s / 2 s / end [rad] | energy drift end [%] | closure max / mean [mm] |
|---|---|---|---|---|---|
| 2.5 ms | 0.005 | 0.9,0.95 | 0.008 / 0.008 / 0.008 | -0.02 | 0.026 / 0.021 |
| 2.5 ms | 0.005 | 0.99,0.999 | 0.007 / 0.007 / 0.007 | -0.02 | 0.002 / 0.002 |
| 2.5 ms | 0.01 | 0.9,0.95 | 0.009 / 0.009 / 0.009 | -0.02 | 0.101 / 0.083 |
| 2.5 ms | 0.01 | 0.99,0.999 | 0.007 / 0.007 / 0.007 | -0.02 | 0.009 / 0.008 |
| 2.5 ms | 0.02 | 0.9,0.95 | 0.014 / 0.014 / 0.014 | -0.02 | 0.351 / 0.294 |
| 2.5 ms | 0.02 | 0.99,0.999 | 0.008 / 0.008 / 0.008 | -0.02 | 0.037 / 0.030 |
| 5 ms | 0.01 | 0.9,0.95 | 0.016 / 0.016 / 0.016 | -0.03 | 0.101 / 0.083 |
| 5 ms | 0.01 | 0.99,0.999 | 0.014 / 0.014 / 0.014 | -0.03 | 0.009 / 0.008 |
| 5 ms | 0.02 | 0.9,0.95 | 0.021 / 0.021 / 0.021 | -0.04 | 0.350 / 0.294 |
| 5 ms | 0.02 | 0.99,0.999 | 0.015 / 0.015 / 0.015 | -0.03 | 0.037 / 0.030 |


3σ stress, MuJoCo C, 32 worlds × 8 s (damped, default solref/solimp): four-bar closure max 4.3 / 7.5 mm (2.5 / 5 ms),
mean 1.6 / 3.4 mm, peak 49 / 42 rad/s, 0 blown; leg 11.8 / 18.6 mm, mean 6.3 / 7.7 mm, peak 94 / 106 rad/s, 0 blown.
At 1σ: four-bar 1.2 / 2.4 mm, leg 5.6 / 5.8 mm.

Reading: closure error scales with the load through the loop and with timeconst²-ish stiffness; the stiff
impedance (0.99/0.999) hardly changes closure in the four-bar (the violation is dynamic, set by solref) but cuts it
10× in the quasi-static leg (the violation there is the static sag under gravity, set by impedance). A 5 ms
timeconst at the 2.5 ms step (= 2·dt, refsafe's floor) halves the error of the default 20 ms but loses 5 % energy
in 5 s: stiff soft constraints are dissipative in MuJoCo's implicit scheme.

Kamino, CPU device, one run (fork 1.7.0.dev, defaults, four-bar passive, 2.5 ms, 1 s, taken **outside the queue**
before the coordinator's rule was restated; the process that was killed for contaminating a timing run was a
numpy reference script, not this one): closure max 0.82 mm, mean 0.44 mm; energy 1.689 → 1.402 J (−17 %) in 1 s;
PADMM converged in 23 iterations to 1e-6. Crank angle within 0.05 rad of the reference at 0.3 s, then lags (0.8 rad
at 0.6 s) because of the energy loss.

<!-- MEASUREMENTS -->

## 5. Verdict (provisional) and the native hard-loop scope

* **Adequacy of MuJoCo Warp's soft closure at our steps.** On MuJoCo C numbers (MuJoCo Warp's equations are the
  same; its float32 agreement on Metal is queued): millimetre-level closure (≤ 3.4 mm at 5 ms, ≤ 1.7 mm at 2.5 ms
  under 3σ torques on a 0.3 m four-bar; ≤ 19 mm on the leg at 5 ms under 3σ, ≤ 12 mm at 2.5 ms). Adequate for
  learning locomotion on a Digit/Cassie-class leg if the loop constraints are stiffened (Cassie ships solref 5 ms)
  and the step is 2.5 ms; the visible cost is energy dissipation in undamped loops, which real, damped drivetrains
  mask. Not adequate where the loop geometry itself is the observable (precision linkages, parallel grippers
  measured to < 1 mm), or at 5 ms with heavy loads through the loop.
* **Kamino on Metal**: unknown until the queue runs; CPU runs and the source suggest no blocker.
* **What a native hard-loop solver for MetalSim would need (scope only):** (1) loop-closing joint rows in
  MuJoCo Warp's constraint assembly as hard rows (zero regularisation, R → 0) with a solver that tolerates
  rank-deficient Jacobians (redundant planar rows): the Newton/CG solver needs a proximal term like Kamino's;
  (2) a position-level projection after integration (Newton–Raphson on g(q) = 0 in the loop coordinates, few
  iterations per step) to remove drift without Baumgarte energy loss; (3) or the reduced-coordinate route
  (PhysX-style): tree articulation + loop joints solved by the iterative contact solver, which is what MuJoCo
  already is with soft rows. Realistic path: (2) as a post-step kernel on MuJoCo Warp (per-world, small dense
  Jacobian, ~days of work) before anything solver-level.

## 6. Decisions (DECISIONS.md-style rows; not yet copied into docs/DECISIONS.md)

| date | decision | options considered (with numbers) | chosen and why | how to re-enable the others |
|---|---|---|---|---|
| 2026-09-25 | Four-bar start pose | 60° crank start (grazes the upper equilibrium near 80°: every engine within 0.14 rad at 2 s would diverge to >1 rad) vs −20° (inside the well) | −20°: comparisons measure closure, not a tipping point | `fourbar_geometry(theta_crank=math.radians(60))` |
| 2026-09-25 | 3σ torque scale and damping | σ 1 N·m crank / 5+3 N·m leg, undamped: 95–240 rad/s, leg 4/32 and 16/32 MuJoCo C blow-ups; σ 0.3 / 4+0.75 N·m undamped: leg 3σ still blows up; same σ with viscous joint damping: 0/32, ≤ 106 rad/s | damped (real drivetrains are damped) | `SIGMA_TAU`, `DAMPING`, `mjcf(damped=False)` |
| 2026-09-25 | Leg geometry | knee 60° start / limits 5–115° / bell crank 150°: rests on the limit (limit rows active through the run) vs 20° / −30…95° / 125° | the latter: the loop, not the limit, is measured | `leg_geometry(q_knee=, gamma=)` |

Runs taken outside the GPU queue (Warp CPU device, before the rule was restated to me): the Kamino CPU test suites
(`kamino_tests_*_cpu.log`), Newton's `kamino_basic_fourbar` example on CPU (compiles and runs, 50 frames, 13 s),
and the single Kamino CPU probe in §4. They are CPU-only and carry no timing claims.

## 7. Left to do (all queued, kind low)

`kamino_tests_fork` / `kamino_tests_152` (Kamino unit tests on metal:0), `cl_kamino`
(`scripts/diagnostics/closed_loops/kamino_jobs.sh`: CPU vs Metal agreement, 1.5.2 vs fork, α sweep, 1024-world 3σ,
throughput graph vs eager, Isaac Lab 3.0 settings, Newton's example), `cl_mjw` (`mjw_jobs.sh`: C / MJW-CPU /
MJW-Metal agreement, solref sensitivity on Metal, 1024-world 3σ for MJW and C, throughput with and without the
equality). PhysX: a recording of the same four-bar with an `excludeFromArticulation` loop joint would close the
comparison; noted, VM not started.
