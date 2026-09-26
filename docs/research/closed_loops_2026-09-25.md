# Closed-loop mechanisms on MetalSim: Newton Kamino on Metal vs MuJoCo Warp soft loop closure

2026-09-25, low priority. Scope: kinematic loops (four-bar linkages, parallel-linkage legs like Digit's, coupled
hands) ahead of a user robot that may have them. **Measured** = run here (M4 Max); Metal and every Warp/Newton
run (including Warp's CPU device) went through `scripts/gpu_run.sh NAME low MIN` with `gpu_lock.py status` at the
top of each log; pure MuJoCo C / numpy runs ran directly. **Source** = read in the cited file or page.

Scripts: `scripts/diagnostics/closed_loops/` (`mechanisms.py` defines the test mechanisms once and emits MJCF, a
Newton builder and an exact planar reference; `mjw_loops.py` MuJoCo C / MuJoCo Warp; `kamino_probe.py`
SolverKamino; `ref_traj.py`; `analyze.py`; `c_sweep.sh`, `kamino_jobs.sh`, `mjw_jobs.sh`, `run_kamino_tests.py`).
Data and logs: `runs/closed_loops/`.

## Summary (2026-09-26)

* **The user robot has no loops in simulation.** TRON1 WF (`neq = 0`, pure tree); nothing to close today.
* **Kamino runs on Metal.** SolverKamino (PADMM backend, the default) in Newton 1.5.2 and the 1.7.0.dev fork runs
  on the Metal Warp fork with no code change, and matches the CPU device to **1.0e-3 rad after 5 s** (four-bar,
  2.5 ms; 1e-5 rad at 0.5 s) and 9e-5 rad on the leg (measured, queued). Two Metal defects found, neither in our
  mechanisms' path: (1) the blocked-Cholesky (LLTB) **backward solve is wrong on Metal for matrices larger than one
  16-row block** (x error up to 2.3 in Newton's own tests) because a Newton native snippet stages tiles in
  `__shared__` arrays only under `__CUDA_ARCH__` and in per-thread arrays otherwise; Newton's switch
  `NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE=1` selects the generic path (verification queued); (2) the DVI
  backend's factorisation kernel asks for 82,432 bytes of threadgroup memory, the M4 Max allows 32,768: DVI does not
  run on Metal (70 of 607 fork tests error on it). Throughput: single-world steps take ~100 ms (200 host-synchronised
  PADMM iterations); 1024-world numbers queued.
* **Hard vs soft closure, both measured against an exact reference** (planar DAE, 1e-11 m): at 2.5 ms Kamino's
  closure error is 0.58 mm at its default Baumgarte α 0.01 and **0.044 mm at α 0.5**; MuJoCo's soft `connect` is
  0.36–1.16 mm (solref 5–20 ms). But Kamino at its default α loses **32 % of the energy in 5 s** on the undamped
  four-bar (α 0.1, Isaac Lab 3.0's value: −14 %; α 0.5: −1.0 %), against −1.1 to −5.2 % for MuJoCo; trajectory
  error after 5 s: Kamino α 0.5 0.38 rad, MuJoCo best 0.56 rad, Kamino default 2.6 rad. With joint damping and 3σ
  torques all are within 0.08–0.5 rad over 5 s.
* **MuJoCo Warp soft closure is adequate at 2.5 ms and 5 ms** for these mechanisms: MuJoCo Warp (float32, CPU
  device) reproduces MuJoCo C's closure numbers (four-bar 1.19 vs 1.07 mm, leg 0.35 vs 0.35 mm at 2.5 ms) and its
  trajectories (leg 1e-4 rad over 2 s; undamped four-bar 0.015 rad at 1 s, 0.32 rad at 5 s); under 3σ
  torques, 1024 worlds × 8 s in MuJoCo C: 0 blown, closure max 5.6 / 11.7 mm (four-bar, 2.5 / 5 ms) and 14.7 /
  25.6 mm (leg). Metal-device MuJoCo Warp runs (agreement, 1024-world stress, throughput) are queued (`cl_mjw`).
* **Verdict** (§5): MuJoCo Warp's soft closure is adequate for learning on loop-legged robots at 2.5 ms with the
  loop equalities stiffened (solref ≈ 5–10 ms, as Menagerie's Cassie does), mm-level closure; Kamino is the more
  accurate closure on Metal (hard, 0.04 mm at α 0.5) but is 1–2 orders slower per step here and beta.

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

### 4.1 MuJoCo C, 1024 worlds × 8 s under 3σ torques (400 control steps, damped; CPU, not queued)

| mechanism | dt | blown | closure max over worlds and time | per-step p99, max over time | peak joint speed |
|---|---|---|---|---|---|
| four-bar | 2.5 ms | 0 / 1024 | 5.6 mm | 2.6 mm | 68 rad/s |
| four-bar | 5 ms | 0 / 1024 | 11.7 mm | 5.3 mm | 69 rad/s |
| leg | 2.5 ms | 0 / 1024 | 14.7 mm | 8.1 mm | 121 rad/s |
| leg | 5 ms | 0 / 1024 | 25.6 mm | 10.7 mm | 136 rad/s |

(`runs/closed_loops/mjw/C_w1024.log`.) Default solref/solimp; the 5 ms-timeconst variant on Metal is queued.

## 4b. Kamino and MuJoCo Warp, CPU device and Metal

Same metrics as §4 (`analyze.py`), single world. CPU-device runs 2026-09-26 while the queue was paused
(`runs/closed_loops/cpu/`); Metal runs through the queue (`runs/closed_loops/kamino/`, log
`runs/closed_loops/kamino_jobs.log`). Kamino defaults unless stated: PADMM, 200 iterations, tolerance 1e-6, α 0.01,
semi-implicit Euler. PADMM hit the 200-iteration cap on every step checked (`status`: iterations 200).

| engine | case | dt | angle err 1 s / 2 s / 5 s [rad] | energy drift [%] | closure max / mean [mm] |
|---|---|---|---|---|---|
| Kamino fork, Metal | four-bar passive | 2.5 ms | 0.80 / 2.16 / 2.58 | −32.5 | 0.58 / 0.31 |
| Kamino 1.5.2, Metal | four-bar passive | 2.5 ms | 0.80 / 2.16 / 2.58 | −32.5 | 0.58 / 0.31 |
| Kamino fork, Metal, α 0 | four-bar passive | 2.5 ms | 0.90 / 2.20 / 2.54 | −46.3 | 7.2 / 3.8 |
| Kamino fork, Metal, α 0.1 | four-bar passive | 2.5 ms | 0.22 / 0.97 / 2.75 | −14.2 | 0.17 / 0.045 |
| Kamino fork, Metal, α 0.5 | four-bar passive | 2.5 ms | 0.024 / 0.060 / 0.38 | −1.0 | 0.044 / 0.010 |
| Kamino fork, CPU, IL3 settings (α 0.1, tol 1e-4, 100 it.) | four-bar passive | 2.5 ms | 0.22 / 0.97 / 2.75 | −14.2 | 0.17 / 0.045 |
| Kamino fork, Metal | four-bar passive | 5 ms | 1.24 / 2.44 / 2.44 | −40.8 | 1.63 / 1.01 |
| MuJoCo Warp, CPU | four-bar passive | 2.5 ms | 0.021 / 0.14 / 0.80 | −1.8 | 1.19 / 0.49 |
| MuJoCo Warp, CPU | four-bar passive | 5 ms | 0.041 / 0.17 / 0.92 | −1.9 | 2.44 / 0.92 |
| Kamino fork, Metal | leg passive (2 s) | 2.5 ms | 0.007 / 0.014 / – | −0.10 | 0.85 / 0.48 |
| Kamino fork, Metal | leg passive (2 s) | 5 ms | 0.016 / 0.031 / – | −0.23 | 2.66 / 1.64 |
| MuJoCo Warp, CPU | leg passive (2 s) | 2.5 ms | 0.014 / 0.014 / – | −0.02 | 0.35 / 0.29 |
| MuJoCo Warp, CPU | leg passive (2 s) | 5 ms | 0.021 / 0.021 / – | −0.04 | 0.35 / 0.29 |
| Kamino fork, Metal | four-bar 3σ, damped | 2.5 ms | 0.13 / 0.22 / 0.52 | | 0.82 / 0.31 |
| Kamino fork, CPU | four-bar 3σ, damped | 5 ms | 0.27 / 0.27 / 0.86 | | 2.61 / 1.17 |
| MuJoCo Warp, CPU | four-bar 3σ, damped | 2.5 ms | 0.028 / 0.038 / 0.076 | | 1.66 / 0.33 |
| MuJoCo Warp, CPU | four-bar 3σ, damped | 5 ms | 0.057 / 0.069 / 0.14 | | 3.43 / 0.61 |

* **CPU vs Metal (Kamino)**: max body-angle difference 1e-5 rad at 0.5 s, 1.0e-3 rad at 5 s (four-bar, fork; 1.4e-3
  for 1.5.2), 9e-5 rad at 2 s (leg), 2e-4 rad at 5 s (four-bar, 5 ms). Newton 1.5.2 and the fork agree to 2.4e-5 rad
  over 5 s on the CPU.
* Kamino's energy loss falls monotonically with the Baumgarte gain (α 0 → 0.5: −46 → −1 %). Likely mechanism (not
  verified in code): with weak stabilisation the position drift grows and each step's velocity-level constraint
  solve, applied at a drifted configuration, removes energy. At α 0.5 Kamino is the most
  accurate engine here. The loss depends on the mechanism: the quasi-static leg loses 0.1 %.
* MuJoCo Warp float32 on the CPU device vs MuJoCo C float64 (same MJCF, default solref): leg 1.1e-4 rad over 2 s;
  damped 3σ four-bar 1.1e-3 rad at 1 s, 7.9e-3 rad at 5 s; undamped passive four-bar 0.015 rad at 1 s, 0.06 at 2 s,
  0.32 rad at 5 s (a conservative 1-DoF system accumulates phase differences). Closure agrees to 10 %.
* Wall time per 5 s single-world four-bar run: Kamino ~200 s on Metal, 63 s on the CPU device (100 / 32 ms per
  step: 200 PADMM iterations whose loop condition is read on the host through `capture_while` outside a capture);
  MuJoCo Warp 3 s on the CPU device.

### 4c. Kamino test suites on Metal (queued runs, 2026-09-25/26)

| suite | CPU device | Metal |
|---|---|---|
| Newton 1.5.2 (`newton._src.solvers.kamino.tests`, 52 modules) | 624 run, 1 fail (box-on-box tolerance), 0 err | 624 run, 12 fail, 22 err |
| fork 1.7.0.dev (`newton.tests.kamino`, 45 modules) | 607 run, 0 fail, 0 err | 607 run, 10 fail, 70 err |

Every Metal-only failure falls in two groups: (a) `Kernel requests 82432 bytes of threadgroup memory but device
"Apple M4 Max" supports at most 32768 bytes` in `llt_blocked_factorize_kernel` / `llt_blocked_rcm_factorize_kernel`
(all DVI-backend tests, the DVI configurations of the joint-effort-limit and joint-friction tests, body flags); (b)
LLTB / LLTB-RCM solve tests with multi-block matrices: factorisation L correct, backward solve x wrong (error up to
2.3). Cause of (b), from source: `_tile_builtins.make_tile_matmul_left_transpose_update_func` declares its staging
arrays `__shared__` under `#if defined(__CUDA_ARCH__)` and as plain per-thread arrays otherwise; each thread fills
only its own tile registers, so on Metal the reduction reads uninitialised entries. It triggers only when the
system spans more than one tile block (the tests use 16-row blocks) (not our four-bar or leg: CPU and Metal agree), i.e. it will hit
real robots. Remedy without code change: `NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE=1` (Newton's own switch; the
fallback uses `tile_transpose` + `tile_matmul`), verification queued (`cl_kamino2`). Upstream fix: stage through
`WP_TILE_ALLOC` / threadgroup memory on non-CUDA GPUs. (a) needs a smaller block size for Metal's 32 KB limit.
Newton's `kamino_basic_fourbar` example ran on the CPU device (50 frames); its Metal run is queued.

A 1024-world × 8 s Kamino stress run held the GPU for > 6 min without output (alive at 17 % CPU, waiting on
command buffers: ~100 ms per step × 3,200 steps) and was killed at the coordinator's request 2026-09-26 02:06; the
remaining steps are re-queued as 2 s runs, one ticket each, with a 600 s per-step cap (`kamino_jobs3.sh`).

## 5. Verdict and the native hard-loop scope

* **Is MuJoCo Warp's soft closure adequate at our step sizes?** Yes for learning on loop-legged robots, with
  numbers: closure 0.4–1.2 mm (2.5 ms) and 1.3–2.4 mm (5 ms) on a passive 0.3 m four-bar, 0.002–0.35 mm on the leg;
  under 3σ torques over 1024 worlds × 8 s, 0 blow-ups, worst closure 5.6 / 14.7 mm at 2.5 ms (four-bar / leg) and
  11.7 / 25.6 mm at 5 ms. Stiffen the loop equalities (solref 5–10 ms; the error scales roughly with the time
  constant) and prefer 2.5 ms. Costs: soft stiff constraints dissipate (−1 to −5 % energy in 5 s undamped), and the
  closure error grows with load; not adequate where sub-millimetre loop geometry is the observable.
* **Does Kamino run on Metal?** Yes (PADMM, both Newton versions, CPU-matching to 1e-3 rad over 5 s), with two
  Metal defects in its linear algebra that our small mechanisms do not reach: the multi-block LLTB solve (wrong
  results; Newton switch available) and the DVI factorisation's 82 KB threadgroup request (DVI unusable). Use α ≈ 0.5
  for loops: at the default 0.01 it loses 32 % of the energy in 5 s. Per-step cost is 1–2 orders above MuJoCo Warp
  here (throughput queued).
* **Isaac / PhysX reference.** Isaac Lab 3.0 keeps closed-loop Digit PhysX-only; PhysX closes loops with iterative
  maximal-coordinate joints next to the reduced-coordinate tree, i.e. also not exactly closed. A PhysX recording of
  this four-bar would place PhysX on the same axis; noted, VM not started.
* **What a native hard-loop solver would need (scope only):** (1) hard loop rows in MuJoCo Warp's constraint solve
  (zero regularisation) plus a proximal/regularised treatment of the rank-deficient loop Jacobian (Kamino's P-ADMM
  does this); (2) or, cheaper, a position-level projection kernel after each MuJoCo Warp step (Gauss–Newton on the
  loop residual in joint space, per world, small dense Jacobian) and the matching velocity projection, which removes
  drift without Baumgarte energy loss; (3) or adopt Kamino for loop robots once its Metal linear-algebra defects are
  fixed upstream and its per-step cost is acceptable. (2) is the realistic first step.

## 6. Decisions

The three set-up rows (four-bar start pose, 3σ scale and damping, leg geometry) are in `docs/DECISIONS.md`
(2026-09-26).

Runs taken outside the GPU queue (Warp CPU device, before the rule was restated to me): the Kamino CPU test suites
(`kamino_tests_*_cpu.log`), Newton's `kamino_basic_fourbar` example on CPU (compiles and runs, 50 frames, 13 s),
and the single Kamino CPU probe in §4. They are CPU-only and carry no timing claims.

## 7. Left to do (queued, kind low; results land in `runs/closed_loops/`)

* `cl_mjw` (`mjw_jobs.sh` → `runs/closed_loops/mjw/`, log `mjw_jobs.log`): MuJoCo Warp on Metal vs MuJoCo C and the
  reference (four-bar, leg, 3σ; 2.5 and 5 ms), solref sensitivity on Metal, 1024-world 3σ on Metal, throughput
  with and without the loop equality at 1024 / 4096 worlds.
* `cl_kamino2` (`kamino_jobs2.sh`, log `kamino_jobs2.log`): LLTB tests and Kamino runs on Metal with
  `NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE=1`, leg 1024-world 3σ at α 0.5, leg throughput.
* `cl_k3_*` (`kamino_jobs3.sh STEP`, logs `kamino_jobs3_STEP.log`): 1024-world 3σ (four-bar, leg, leg at 5 ms),
  throughput graph vs eager, Isaac Lab 3.0 settings on Metal, Newton's example on Metal.
* PhysX recording of the four-bar (not started).
