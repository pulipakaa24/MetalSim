# Isaac Lab 3.0-EA like-for-like: the 3.0 G1 task, Isaac's own MuJoCo-Warp settings, training (2026-09-25)

Question: what does it take to compare MetalSim's G1 training against Isaac Lab 3.0.0-EA's own runs
(`runs/parity3/isaac/`: flat Newton/MuJoCo-Warp +27.4 and PhysX +28.7 at iteration 1499, rough Newton +14.5 /
level 5.8) like for like, given that 3.0's task is harder than the 2.3.2 one MetalSim's port matches, and that
Isaac's default backend is itself MuJoCo Warp with settings of its own?

Sources: Isaac Lab tag `v3.0.0-EA` (commit ae37b028, 2026-09-16) cloned to `upstream/IsaacLab3`; Isaac Lab tag
`v2.3.2` (37ddf626) sparse-cloned to `upstream/IsaacLab232`; Newton 1.5.2 (cf5378db, Isaac Lab 3.0's pin) in
`upstream/newton-1.5.2`; the live settings recorded on the L4 (`runs/parity3/isaac/newton_mjwarp_settings.json`,
`runs/parity3/isaac/fidelity/newton_mjwarp/{meta.json,newton_generated.xml}`). Paths below are relative to each
repository's `source/` unless stated. Every number is measured unless marked otherwise.

## 1. The 3.0 G1 task differences, ported (`reward_cfg="flat_il3"` / `"rough_il3"`)

Code: `metalsim/learn/g1_il3.py` (kernels, constants, citations), switches in `G1VelocityTask`
(`reward_cfg`, `il3_events`, `il3_backend`, `solver_cfg`), tests in `tests/test_g1_task_terms.py` (`test_il3_*`) and
`tests/test_solver_presets.py`. The 2.3.2 configs (`flat`, `rough`, `rough_isaac`) are unchanged.

### 1.1 Events (diff of 3.0 `isaaclab_tasks/core/velocity/{velocity_env_cfg.py, config/g1/rough_env_cfg.py}` against 2.3.2)

| event | 2.3.2 G1 | 3.0 G1 (source) | how it is applied in Isaac | here |
|---|---|---|---|---|
| `push_robot` | `None` | `push_by_setting_velocity`, interval, `interval_range_s=(10, 15)`, `velocity_range` x, y ±0.5 m/s (`velocity_env_cfg.py` `EventsCfg`; G1 no longer removes it) | `isaaclab/envs/mdp/events.py:2099`: root **COM** velocity (`root_vel_w` = `root_com_vel_w`) `+= U(range)` per axis, unlisted axes 0; `write_root_velocity_to_sim_index`. `isaaclab/managers/event_manager.py`: per-env timer `U(10, 15)` at start and on each reset of the env, `-= step_dt` every env step, fire and redraw below 1e-6. `manager_based_rl_env.py` step order: physics, terminations, rewards, resets, command update, **interval events**, observations | `g1_il3.g1_push_il3` between the command update and the observation; timer drawn in the reset kernel; the frame origin's velocity changes by the same vector as the COM's (ω unchanged) |
| `add_base_mass` | `None` | `randomize_rigid_body_mass` on `torso_link` (G1 override of "base"), startup, `mass_distribution_params=(1/1.25, 1.25)`, `operation="scale"`, `distribution="log_uniform"` | `events.py:665`: mass = default × exp(U(log 0.8, log 1.25)) per env, `recompute_inertia=True` (default): inertia tensor × the same ratio; COM unchanged. Newton backend: `SolverMuJoCo.notify_model_changed(BODY_INERTIAL_PROPERTIES)` (Newton `solver_mujoco.py` ~4440) runs `set_const_fixed` + `set_const_0` and re-scales the force-space joint-limit solref from the new `dof_invweight0` | per-world `body_mass`, `body_inertia` (+ `body_subtreemass`, `body_invweight0`, `dof_invweight0`, and `jnt_solref` on the isaaclab3 preset) on the torso; MuJoCo Warp `set_const`; `stat.meaninertia` restored to nominal (a shared field) |
| `reset_base` | pose xy ±0.5, yaw ±3.14, **velocity 0** | the base config's: pose xy ±0.5, yaw ±3.14, **velocity ±0.5 on x, y, z, roll, pitch, yaw** | `events.py:2127`: velocity = default (0) + U(range), world frame, written as the root COM velocity ("the root's center of mass rather than the root's frame"); orientation = default × quat_from_euler_xyz(0, 0, yaw) | `g1_il3.g1_reset_il3`: MuJoCo's free joint takes v_origin = v_com − ω × (R r_com), ω_body = Rᵀ ω; yaw ±3.14 (the 2.3.2 port keeps its ±π) |
| unchanged | `physics_material` 0.8 / 0.6 degenerate, `reset_robot_joints` (1, 1), zero `base_external_force_torque`, `base_com` None | same | – | – |

### 1.2 Other differences that change the MDP

* **Contact sensor on Newton** (`isaaclab_newton/sensors/contact_sensor/contact_sensor.py`): `force_threshold` None →
  **0 N** on Newton (1 N on PhysX, `isaaclab_physx/.../contact_sensor.py:131`, and in 2.3.2), so a foot is "in contact"
  for air/contact time as soon as its normal force is non-zero. The sensor is updated once per 5 ms physics tick with the
  forces of the tick's last 2.5 ms substep (`NewtonManager._update_sensors` → `solver.update_contacts`); the G1's
  implicit actuators are not all CUDA-graph-safe, so `handles_decimation()` is False and the env loops the four ticks,
  each followed by `scene.update(dt=physics_dt)`. History 3 ticks (15 ms, three samples). Reproduced: `ContactSensor`
  launched on every 2nd substep, `dt` 5 ms, history 3, threshold 0 N (`il3_backend="physx"`: 1 N).
* **`feet_slide` / `illegal_contact`** read `net_normal_forces_w_history` in 3.0 (2.3.2: `net_forces_w_history`, which
  2.3.2 documents as the normal force). MetalSim's sensor reports the normal force in both.
* **Root linear velocity = the pelvis COM's.** `base_lin_vel`, `track_lin_vel_xy_yaw_frame_exp` and `lin_vel_z_l2`
  read `root_lin_vel_*`, which is `root_com_lin_vel_*` in both versions (3.0
  `isaaclab/assets/articulation/base_articulation_data.py:1384-1398`; 2.3.2 `articulation_data.py:1041`). The 2.3.2
  ports (and their tests) used MuJoCo's free-joint velocity, i.e. the pelvis frame origin, 7.6 cm above the pelvis COM
  (difference ω × r, up to ~4 cm/s at 0.5 rad/s). **This is a defect of the 2.3.2 ports**, found here; the il3
  configs use the COM velocity (checked against MuJoCo C's `mj_objectVelocity`); the 2.3.2 configs are left as they
  were (rule: keep them untouched; ledger row below).
* **Rough terrain on Newton is a heightfield.** 3.0's `ROUGH_TERRAINS_CFG` sets `convert_to_heightfield=True` on every
  sub-terrain (`isaaclab/terrains/config/rough.py`; the hf terrains default to True), and the terrain importer tags the
  mesh for conversion only if all do (`terrain_importer.py` `_tag_heightfield_collider`); Newton then rasterizes the
  mesh to a heightfield at the 0.1 m horizontal scale (`isaaclab_newton/physics/newton_manager.py`
  `_inject_terrain_heightfields` → `newton.Heightfield.create_from_mesh`, "rays are cast straight down ... on a regular
  grid"). Stair risers and box walls become 0.1 m ramps on Isaac's Newton backend: MetalSim's `terrain_collision="hfield"`,
  which is `rough_il3`'s default (the 2.3.2 rough port keeps its exact boxes, which match PhysX's mesh). The height
  scanner still ray-casts the mesh (exact scan, unchanged).
* **Friction.** Newton combines two shapes' friction by the maximum when their priorities are equal (Newton
  `kernels.py` `contact_params`; Isaac Lab 3.0 does not pass PhysX's `friction_combine_mode="multiply"`), so the G1's
  contacts use max(robot 0.8, ground 1.0) = **1.0** on Isaac's Newton backend and 0.8 × 1.0 = 0.8 on PhysX. MetalSim's
  model has robot geoms at 1.0 and the ground at 0.8, and MuJoCo also takes the maximum: 1.0, i.e. Newton's value
  (the PARITY §1.2 row "friction 0.8/0.6 fixed" describes the config, not what MuJoCo applies).

### 1.3 Unchanged between the tags (diff)

Reward terms, weights and formulas (`isaaclab_tasks/core/velocity/mdp/rewards.py`: only `.torch` accessors and the
normal-force field above; `isaaclab/envs/mdp/rewards.py`: `joint_torques_l2` reads `actuators.applied_effort`, the
same implicit-PD estimate as 2.3.2's `applied_torque`); observation terms and noise (`UniformNoiseCfg` is additive like
2.3.2's `AdditiveUniformNoiseCfg`; projected gravity now normalizes the gravity vector, a no-op); command ranges and
heading control (`velocity_command.py`: only new success metrics); terminations (field above); curriculum
`terrain_levels_vel`; terrain generator and `ROUGH_TERRAINS_CFG` values (only `convert_to_heightfield`); actuator gains,
armature and effort limits (`G1_CFG`: `effort_limit_sim` → `joint_effort_limit`); decimation 4, `sim.dt` 0.005, 20 s
episodes; PPO hyper-parameters (flat 1500 iterations; rough `max_iterations` 5000 on Newton, 3000 otherwise; the
reference runs used 1500). Quaternions are XYZW in 3.0; no observation contains one. The asset (Isaac Sim 6.1 USD, as
Newton built it, `newton_generated.xml`) matches MetalSim's 5.1 copy: 44 bodies, masses within 4e-6 relative, COMs
within 4e-7 m, inertias within 5e-6 relative, all 37 joint ranges and armatures, and all 37 kp / kd equal (measured
from the recording).

### 1.4 Tests (`tests/test_g1_task_terms.py`, `tests/test_solver_presets.py`)

26 passed on Metal (`runs/il3/tests_2.log`, 2026-09-25 15:37; the first pass, `tests_1.log`, failed 4 on test mistakes:
a float32 timer edge, the default 5 ms physics step in two tests, and a witness-matching assumption; fixed in the tests):

| test | asserts |
|---|---|
| `test_il3_reward_terms_match_isaac_formulas[flat_il3, rough_il3]` | all 13 terms equal Isaac's formulas × weights on the live state, linear-velocity terms from MuJoCo C's `mj_objectVelocity` of the pelvis COM, contact terms from the 3.0 sensor semantics |
| `test_il3_base_lin_vel_is_the_root_com_velocity` | observation 0:3 = pelvis-COM velocity in the pelvis frame (±0.1 noise), and differs from the frame-origin velocity on moving robots |
| `test_il3_reset_root_state_distribution_and_application` | 4096 resets: x, y offsets U(±0.5) and yaw U(±3.14) (KS p > 1e-3, support), pure-yaw orientation, default joints, zero joint velocity; the COM linear and angular velocities read back by MuJoCo C are U(±0.5) on each of six axes and uncorrelated; push timers U(10, 15); `il3_events=False` zeroes the velocities |
| `test_il3_push_robot_interval_and_velocity` | EventManager's rule: fire iff `time_left − 0.02 < 1e-6` (a timer at 0.020002 s counts down), fired envs' xy velocity += U(±0.5) (KS), z and angular unchanged, MuJoCo C's COM velocity changes by exactly that vector, new timers U(10, 15), non-fired timers −0.02 |
| `test_il3_add_base_mass_distribution_and_constants[default, isaaclab3]` | torso mass ratio log-uniform on [log 0.8, log 1.25] (KS), inertia × the same ratio, other bodies unchanged, subtree mass, and `dof_invweight0` / `body_invweight0` per world equal to MuJoCo C's `mj_setConst` on the same masses (rtol 2e-3); with Isaac's preset the joint-limit solref re-scaled as Newton does (rtol 3e-3); nominal `meaninertia` kept |
| `test_il3_contact_sensor_ticks` | sensor updated every 2nd 2.5 ms substep, times in whole 5 ms ticks, 3-tick history, in contact iff force > 0 N (Newton), 1 N for `il3_backend="physx"` |
| `test_il3_rough_defaults` | rough_il3 = rough_isaac rewards / commands, heightfield collision, exact scan |
| `tests/test_solver_presets.py` (4) | model fields equal the recording (options, geom solref / solimp / gap, all 37 joint-limit solrefs, ranges, armature); refreshed contact distance = fresh collision within 2e-6 m after a move; once-per-tick vs every-substep stand stable and within 5 mm; caps 100 and 20 agree to float noise |

The 2.3.2 tests in the same files (reward sets, observation layout, contact history, feet slide, terrain) still pass:
the kernels' new `root_com` argument is zero for them.

## 2. Isaac's own MuJoCo-Warp settings as a preset (`solver_cfg="isaaclab3"`)

Code: `metalsim/physics/solver_presets.py` (a new module; `contact_tuning.py` belongs to the contact-fidelity agent
and is only read). `G1VelocityTask(solver_cfg=...)`, `metalsim.parity.record_g1 --solver_cfg ...` and
`scripts/diagnostics/il3_transfer.py` take the preset. Tests: `tests/test_solver_presets.py`.

### 2.1 What is honoured, and what is not

| Isaac Lab 3.0 / Newton 1.5.2 setting (source) | honoured? | how / why not |
|---|---|---|
| 5 ms tick × 2 substeps = 2.5 ms solver step (`NewtonCfg.num_substeps=2`, `sim.dt` 0.005) | yes | the task's 2.5 ms substeps, 8 per control step |
| Newton solver, caps 100 / 50 (`MJWarpSolverCfg`) | yes (`isaaclab3`); cap 20 in the training preset | caps copied; see the cost row |
| early exit at tolerance 1e-6, ls_tolerance 0.01 | **numerically yes, cost no** | MuJoCo Warp marks every world done at the tolerance and skips it in later iterations whether or not a CUDA conditional graph node stops the loop (`mujoco_warp/_src/solver.py` `_solve_done`, `ctx.done`); Metal has no conditional node, so all 100 iterations still launch: 27.5 K vs 53.7 K env-steps/s (§2.3). A cap of 20 gives the same states to float noise (1.2e-7 after 0.2 s, `test_iteration_cap_only_changes_worlds_that_reach_it`) whenever no world needs more than 20 iterations (probe below) |
| implicitfast, pyramidal cone, impratio 1 | yes | already the task's |
| contact solref (1.82 ms, 1.375) from ke 1.6e5 / kd 1100 via `convert_solref(ke, kd, 1, 1)` (shape `solref_mode` default MJCF_DEFAULT, so no force-space override), solimp (0.9, 0.95, 0.001, 0.5, 2) | yes | geom solref / solimp; MuJoCo's refsafe raises 1.82 ms to 2 dt = 5 ms in both engines |
| friction: max(robot 0.8, ground 1.0) = 1.0 | yes (already) | MuJoCo's max rule with robot geoms at 1.0 |
| margin 0, gap 0.01 per shape (pair 0.02, `gap_sum`) | yes | geom gap 0.01; inactive contacts detected out to 2 cm |
| per-joint limit solref, 4 ms – 0.46 s, damping ratio 0.03 – 0.35 (`update_jnt_solref_from_invweight0`, force-space `joint_limit_ke/kd`), solimp default | yes | the recorded live values per joint; refsafe floors the finger joints' 4 ms to 5 ms in both; after `add_base_mass` re-scaled per world as Newton does |
| collision once per 5 ms tick, contacts reused by both substeps with dist/pos refreshed from body poses (`NewtonManager._simulate_*`, `convert_newton_contacts_to_mjwarp_kernel` fast path) | **yes, as an option** | `collision_every=2`: MuJoCo Warp's collision on the first substep; on the second the same contacts, normal and frame, dist/pos from body-local witness points (refreshed distance = a fresh collision pass to 2e-6 m, `test_once_per_tick_contacts_follow_the_bodies`). It matched the recording **worse** than colliding every substep (§2.2), so training uses every-substep collision; the emulation is kept (`isaaclab3`) |
| Newton's own `CollisionPipeline` (`use_mujoco_contacts=False`: explicit broad phase, contact reduction, Newton's box/mesh-vs-plane narrow phase) | **no** | a different collision library; MetalSim runs MuJoCo Warp's (the C-exact plane-convex set). The contact *set* on the feet can differ (count, points), which is the likely reason the once-per-tick emulation does not help |
| `njmax` 95 / `nconmax` 10 (flat), 300 / 300 (rough) | not copied | capacities, not physics: an overflow drops constraints nondeterministically. MetalSim keeps 256 / 32 (flat) |
| contact sensor per tick, 0 N threshold | yes (task side, §1.2) | – |

### 2.2 Fidelity protocol against Isaac's Newton/MuJoCo-Warp recording (measured 2026-09-25)

`python -m metalsim.parity.record_g1 --isaac runs/parity3/isaac/fidelity/newton_mjwarp --out runs/il3/fidelity/<preset>
--no_render [--contact_tuning P | --solver_cfg P]`, then `metalsim.parity.compare --no_render` (momentum-impulse
columns from `metalsim.parity.momentum`, the contact agent's method); table by `runs/il3/fidelity_table.py`, logs
`runs/il3/fidelity.log`, reports `runs/il3/fidelity/report_*/report.json`. "default" = the task's current MuJoCo
defaults (10 / 20 iterations, contact and limit solref 0.02 / 1); "hardlimits" and "tau10_impact_hardlimits" = the
contact agent's hard-limit presets (the latter its provisional pick).



Solver iterations per substep (last substep of each control step, mean / max): Isaac hold 1.46 / 4, random 1.52 / 5,
drop 1.47 / 6; `isaaclab3` 1.33 / 5, 1.60 / 5, 1.32 / 4; every-substep variant 1.47 / 4, 1.55 / 5, 1.34 / 4; default
(tolerance 1e-8, cap 10) 1.24 / 5, 2.08 / 6, 1.23 / 5.

Reading:
* **Soft vs hard joint limits (the key row).** Isaac's own MuJoCo Warp lets joints pass their limits: on the random
  protocol its largest excursion is 0.227 rad with 112 of 250 control steps over 0.01 rad. The isaaclab3 limits
  reproduce that (0.216 rad, 118 steps; every-substep variant 0.219, 120); the hard-limit presets hold the limits
  (0.068 – 0.075 rad) and MuJoCo's default is in between on excursion size but over the limit more often (0.184, 187
  steps). On hold and drop Isaac stays within 0.01 rad, as do all presets except the default (0.030 rad, 8 steps).
  So against Isaac Lab 3.0's Newton reference the hard-limit preset is the *wrong* direction; against PhysX (hard
  limits, PARITY §1.7) it was right. The two Isaac backends disagree here, and MetalSim now has a preset for each.
* End-state joint error: the isaaclab3 family is 10–60× closer on the hold (0.0001 – 0.0003 rad vs 0.017 – 0.018 for
  default / hard limits) and 7–9× on the drop (0.0025 – 0.0033 vs 0.021). The contact and limit settings together
  decide the settled pose; tau10_impact_hardlimits is in between (0.0095 / 0.0125).
* Impulses per event match Isaac within 1 % on every preset (landing 193 vs 195 N s, torso 238 – 241 vs 241 N s):
  momentum is not what separates them. The largest 20 ms mean landing force separates them: Isaac 2086 N; every-substep
  2187 N (+5 %), default / hard limits 2430 N (+16 %), once-per-tick 2979 N (+43 %).
* Once per tick vs every substep: every-substep is closer on 7 of 11 rows (random divergence 0.26 vs 0.12 s, drop root
  height RMSE 2.2 vs 5.9 mm, landing 20 ms force, peak joint speed 15.8 vs 7.1 rad/s against Isaac's 14.9, hold end
  error, penetration similar), once-per-tick on the drop end error (0.0025 vs 0.0032) and hold root height. Honouring
  the reuse without Newton's own contact set does not buy fidelity, so it stays an option.

**Transfer of Isaac Lab 3.0's own checkpoints into MetalSim** (`scripts/diagnostics/il3_transfer.py`, `runs/il3/step2b.log`,
`runs/il3/transfer_*.json`; flat_il3 task with the events off, command (0.5, 0, 0) for 8 s = 4.0 m commanded, 4 envs, mean
action; cells: x travelled mean (range) / final pelvis z / torso contacts / largest joint-limit excursion and its joint).
No Isaac-side play of the 3.0 checkpoints was recorded, so the columns compare presets with each other and with the
command, not with Isaac's own distance (the 2.3.2 PhysX checkpoints walked 3.0–3.2 m in Isaac, PARITY §1.5).


Isaac Lab 3.0 newton_mjwarp checkpoints in MetalSim, command (0.5, 0, 0) for 8 s (commanded 4.0 m), 4 envs, mean action; x travelled [m] mean (min-max) / final pelvis z [m] / torso contacts / max limit excursion [rad]
| checkpoint | default | hardlimits | isaaclab3 | isaaclab3_collide_every_substep | isaaclab3_hardlimits |
|---|---|---|---|---|---|
| it 500 | 3.38 (3.33-3.44) / 0.693 / 0 / 0.084 (left_six_joint) | 3.39 (3.33-3.44) / 0.695 / 0 / 0.084 (left_six_joint) | 3.44 (3.39-3.49) / 0.695 / 0 / 0.032 (left_six_joint) | 3.45 (3.42-3.50) / 0.695 / 0 / 0.032 (left_six_joint) | 3.39 (3.36-3.42) / 0.695 / 0 / 0.081 (left_six_joint) |
| it 1000 | 3.64 (3.62-3.66) / 0.672 / 0 / 0.138 (left_six_joint) | 3.64 (3.63-3.66) / 0.672 / 0 / 0.138 (left_six_joint) | 3.72 (3.69-3.75) / 0.674 / 0 / 0.049 (left_six_joint) | 3.72 (3.69-3.74) / 0.673 / 0 / 0.047 (left_six_joint) | 3.69 (3.68-3.71) / 0.675 / 0 / 0.133 (left_six_joint) |
| it 1499 | 3.66 (3.64-3.70) / 0.663 / 0 / 0.137 (left_six_joint) | 3.66 (3.64-3.70) / 0.663 / 0 / 0.137 (left_six_joint) | 3.78 (3.76-3.79) / 0.662 / 0 / 0.051 (left_four_joint) | 3.78 (3.75-3.80) / 0.664 / 0 / 0.051 (left_four_joint) | 3.69 (3.66-3.73) / 0.666 / 0 / 0.140 (left_six_joint) |

Isaac Lab 3.0 isaacsim_physx checkpoints in MetalSim, command (0.5, 0, 0) for 8 s (commanded 4.0 m), 4 envs, mean action; x travelled [m] mean (min-max) / final pelvis z [m] / torso contacts / max limit excursion [rad]
| checkpoint | default | hardlimits | isaaclab3 | isaaclab3_collide_every_substep | isaaclab3_hardlimits |
|---|---|---|---|---|---|
| it 500 | 3.19 (3.13-3.26) / 0.653 / 0 / 0.097 (left_six_joint) | 3.19 (3.14-3.26) / 0.654 / 0 / 0.097 (left_six_joint) | 3.27 (3.21-3.33) / 0.656 / 0 / 0.035 (left_six_joint) | 3.30 (3.21-3.35) / 0.657 / 0 / 0.035 (left_six_joint) | 3.23 (3.17-3.28) / 0.654 / 0 / 0.096 (left_six_joint) |
| it 1000 | 3.35 (3.31-3.38) / 0.653 / 0 / 0.111 (right_six_joint) | 3.35 (3.31-3.38) / 0.653 / 0 / 0.112 (right_six_joint) | 3.42 (3.39-3.49) / 0.653 / 0 / 0.041 (right_six_joint) | 3.44 (3.40-3.51) / 0.654 / 0 / 0.040 (right_six_joint) | 3.39 (3.34-3.43) / 0.652 / 0 / 0.110 (right_six_joint) |
| it 1499 | 3.52 (3.49-3.57) / 0.644 / 0 / 0.121 (left_six_joint) | 3.52 (3.49-3.57) / 0.644 / 0 / 0.121 (left_six_joint) | 3.51 (3.47-3.54) / 0.646 / 0 / 0.045 (left_six_joint) | 3.53 (3.51-3.59) / 0.646 / 0 / 0.046 (left_six_joint) | 3.56 (3.52-3.60) / 0.646 / 0 / 0.121 (left_six_joint) |

Every checkpoint walks in every preset (no falls). Isaac's numerics add 2–3 % distance (3.72–3.78 vs 3.64–3.66 m for
the Newton checkpoints at 1000 / 1499). The largest excursion is always a finger joint (`*_six_joint`, `left_four_joint`):
the policy's finger targets lie past the limits; Isaac's finger-limit solref (5 ms after refsafe, ζ 0.33–0.35) is
*stiffer* than the hard-limit preset (5 ms, ζ 1; stiffness ∝ 1/ζ²), so Isaac's settings hold the fingers 2.7× closer
(0.047–0.051 vs 0.133–0.140 rad) while leaving the large leg joints soft (time constants 0.16–0.46 s).

### 2.3 Cost (4096 envs, flat, full env step with rewards / resets / observations, synchronized, random actions)

Re-taken 2026-09-25 16:07 through the idle-checked queue (device utilisation 0 % at start, `runs/il3/bench_retake2.log`,
`bench_retake2.queue.out`); the first measurement (`bench.log`, 14:55) fell in a window where a macOS system service
held ~99 % of the GPU (coordinator's contamination notice) and is superseded, although it agreed within 1 %.

| variant | env-steps/s |
|---|---|
| 2.3.2 flat task, MuJoCo defaults (10 / 20 iterations) | 54,236 |
| flat_il3, MuJoCo defaults | 53,880 |
| flat_il3 + hardlimits | 54,019 |
| flat_il3 + isaaclab3 (cap 100, once per tick) | 27,540 |
| flat_il3 + isaaclab3, collision every substep (cap 100) | 27,261 |
| flat_il3 + isaaclab3, cap 20 (once per tick) | 50,357 |
| flat_il3 + isaaclab3, collision every substep, cap 20 (the training preset) | 49,455 |

The 3.0 task terms cost nothing measurable (−0.7 %, noise); the 100-iteration cap halves the rate on Metal (all iterations
launch; per-world exit makes most of them no-ops); a cap of 20 recovers 91–93 %.

**Does the cap of 20 bind?** `runs/il3/niter_probe.py` (flat_il3, 4096 envs, uniform random actions in [−1, 1], 300 control steps, robots falling and lying, every substep read): mean 2.14 iterations per substep, 99.99th percentile 8, maximum 12, and **0 of 9,830,400** substep-worlds above 20, for both collision variants. On this task the cap of 20 is therefore Isaac's cap of 100 to float noise.


## 3. Training like for like

### 3.1 Flat (measured 2026-09-25, 17:16–18:15)

Run: `runs/il3/run_train.sh flat isaaclab3_every_substep_cap20 1500 0` = `python -m metalsim.learn.g1_velocity 4096 flat train
1500 ... 0.0025 --reward_cfg flat_il3 --solver_cfg isaaclab3_every_substep_cap20 --seed 0` (PPOWarp, Isaac's G1 flat PPO
configuration, 4096 envs, 2.5 ms physics, seed 0). Log `runs/il3/train_flat_il3_isaaclab3_every_substep_cap20_s0.log`
(with a `terms it N {...}` line per iteration), checkpoints every 100 iterations in `runs/il3/ckpt/`. Comparison:
`python scripts/diagnostics/g1_learning_curves.py il3 flat <log>` (`runs/il3/curves_flat.md`); Isaac's side is
`runs/parity3/isaac/train/train_flat_{newton_mjwarp,isaacsim_physx}_terms.txt`. add_base_mass drew torso scales 0.800–1.250
(dof_invweight0 ratios 0.95–1.05, joint-limit time-constant ratios 0.98–1.02); 139,712 pushes over the run (≈0.8 per
finished full episode, as a 10–15 s timer on 20 s episodes gives).

Isaac Lab 3.0 (newton_mjwarp, isaacsim_physx; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim runs/il3/train_flat_il3_isaaclab3_every_substep_cap20_s0.log (PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)
| iteration | Isaac newton_mjwarp: length / return / lin track / yaw track | Isaac isaacsim_physx: length / return / lin track / yaw track | MetalSim: length / return / lin track / yaw track |
|---|---|---|---|
| 50 | 53 / -5.0 / 0.012 / 0.004 | 51 / -5.0 / 0.013 / 0.004 | 50 (50) / -5.3 (-5.3) / 0.010 / 0.004 |
| 100 | 73 / -5.0 / 0.022 / 0.007 | 177 / -6.5 / 0.044 / 0.015 | 64 (66) / -5.0 (-5.0) / 0.019 / 0.007 |
| 150 | 536 / -9.4 / 0.218 / 0.061 | 978 / -6.6 / 0.486 / 0.105 | 319 (333) / -8.4 (-8.5) / 0.099 / 0.036 |
| 200 | 988 / -4.6 / 0.526 / 0.128 | 977 / +4.3 / 0.739 / 0.179 | 875 (883) / -8.1 (-8.0) / 0.380 / 0.117 |
| 250 | 982 / +3.4 / 0.729 / 0.175 | 972 / +9.4 / 0.809 / 0.245 | 951 (958) / -1.5 (-1.7) / 0.593 / 0.158 |
| 300 | 998 / +8.4 / 0.830 / 0.234 | 1000 / +14.4 / 0.872 / 0.374 | 996 (980) / +3.5 (+3.4) / 0.746 / 0.189 |
| 400 | 968 / +14.0 / 0.841 / 0.372 | 982 / +20.4 / 0.904 / 0.569 | 991 (980) / +9.1 (+8.7) / 0.823 / 0.261 |
| 500 | 1000 / +19.6 / 0.917 / 0.530 | 1000 / +23.8 / 0.922 / 0.672 | 988 (992) / +13.6 (+13.7) / 0.876 / 0.364 |
| 750 | 1000 / +24.6 / 0.935 / 0.679 | 982 / +26.2 / 0.929 / 0.745 | 977 (985) / +19.0 (+19.2) / 0.904 / 0.516 |
| 1000 | 989 / +25.9 / 0.932 / 0.723 | 998 / +28.0 / 0.937 / 0.778 | 990 (992) / +22.2 (+22.2) / 0.922 / 0.610 |
| 1250 | 997 / +27.1 / 0.940 / 0.748 | 996 / +28.3 / 0.943 / 0.789 | 996 (987) / +23.5 (+23.3) / 0.920 / 0.651 |
| 1499 | 1000 / +27.5 / 0.940 / 0.753 | 1000 / +28.9 / 0.944 / 0.795 | 995 (990) / +24.9 (+24.7) / 0.927 / 0.689 |

Per-term at iteration 1000 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9323 | +0.9367 | +0.9221 |
| track_ang_vel_z_exp | +0.7231 | +0.7784 | +0.6103 |
| feet_air_time | +0.0417 | +0.0462 | +0.0542 |
| feet_slide | -0.0121 | -0.0115 | -0.0146 |
| joint_deviation (hip+arms+fingers+torso) | -0.1446 | -0.1394 | -0.1690 |
| flat_orientation_l2 | -0.0076 | -0.0064 | -0.0049 |
| action_rate_l2 | -0.1872 | -0.1664 | -0.2319 |
| termination_penalty | -0.0022 | +0.0000 | -0.0038 |
| lin_vel_z_l2 | -0.0044 | -0.0044 | -0.0060 |
| ang_vel_xy_l2 | -0.0123 | -0.0109 | -0.0138 |
| dof_torques_l2 | -0.0080 | -0.0095 | -0.0081 |
| dof_acc_l2 | -0.0108 | -0.0103 | -0.0193 |
| dof_pos_limits | -0.0021 | -0.0029 | -0.0031 |
| falls (base_contact fraction of episode ends) | 0.0064 | 0.0012 | 0.0189 |

Per-term at iteration 1499 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9399 | +0.9436 | +0.9270 |
| track_ang_vel_z_exp | +0.7526 | +0.7946 | +0.6894 |
| feet_air_time | +0.0488 | +0.0491 | +0.0670 |
| feet_slide | -0.0107 | -0.0105 | -0.0120 |
| joint_deviation (hip+arms+fingers+torso) | -0.1399 | -0.1321 | -0.1643 |
| flat_orientation_l2 | -0.0060 | -0.0049 | -0.0048 |
| action_rate_l2 | -0.1802 | -0.1565 | -0.2166 |
| termination_penalty | -0.0017 | +0.0000 | -0.0040 |
| lin_vel_z_l2 | -0.0041 | -0.0052 | -0.0058 |
| ang_vel_xy_l2 | -0.0122 | -0.0111 | -0.0136 |
| dof_torques_l2 | -0.0079 | -0.0091 | -0.0083 |
| dof_acc_l2 | -0.0108 | -0.0101 | -0.0171 |
| dof_pos_limits | -0.0018 | -0.0022 | -0.0030 |
| falls (base_contact fraction of episode ends) | 0.0060 | 0.0027 | 0.0198 |

Throughput (training loop, env-steps/s):
  Isaac newton_mjwarp: median 59,405 (L4), total iteration time 41.3 min over 1500 iterations
  Isaac isaacsim_physx: median 47,257 (L4), total iteration time 52.2 min over 1500 iterations
  MetalSim: median 42,589 (M4 Max, incl. the monitor), 1500 iterations


Reading (one seed on each side; MetalSim's three 2.3.2-task seeds spread ±1.3 at iteration 1000, PARITY §1.5):
* **Return at 1499: +24.9 (±5 mean +24.7) vs Isaac Newton +27.5 and PhysX +28.9**: 2.6 below Isaac's own MuJoCo-Warp
  run (≈2 seed standard deviations), with the same full episodes (995 vs 1000) and the same linear tracking (0.927 vs
  0.940). The rise is slower: +3.5 vs +8.4 at iteration 300, +13.6 vs +19.6 at 500, +22.2 vs +25.9 at 1000; Isaac's
  Newton run is itself 1–6 behind its PhysX run until iteration 1000.
* **Per term at 1499** the gap is: yaw tracking −0.063 (0.689 vs 0.753) the largest, then action rate −0.036, joint
  deviation −0.024, dof_acc −0.006, falls (termination −0.002; 2.0 % vs 0.6 % of episode ends), partly offset by feet air
  time +0.018 (0.067 vs 0.049). Summed over the terms that is −0.13 per second, i.e. −2.6 over the 20 s episode, the whole
  return gap. Feet slide (−0.012 vs −0.011) and torques (−0.0083 vs −0.0079) match.
* Against MetalSim's own 2.3.2-task result (+28.4 / 26.3 / 26.0 at 1000 over three seeds, no events): the 3.0 task costs
  MetalSim about 5 return at iteration 1000 (+22.2) where it cost Isaac ≈1.4 (2.3.2 PhysX +27.3 → 3.0 Newton +25.9) –
  MetalSim's policy is more affected by the pushes / heavier torsos / reset velocities, visible in the yaw tracking and the
  fall fraction.
* **Blow-ups**: 44 non-finite episodes in 11 iterations (13 of them in iterations 11–13 while the policy flails; 31 after
  iteration 1000), vs none on the 2.3.2 task with MuJoCo defaults. Isaac's soft leg limits (time constants up to 0.46 s)
  plus random targets let joints travel far past their limits; the monitor logged "joint limit violated by 0.1 rad"
  warnings through the run. Not seen on Isaac's side (its logs have no such counter).
* **Throughput**: MetalSim training loop median 42,589 env-steps/s on the M4 Max (incl. the anomaly monitor and the per-term
  logging sync) vs Isaac 59,405 (Newton) and 47,257 (PhysX) on the L4; 59 vs 41.3 / 52.2 min. Caveat: from ~17:30 a queue
  bug (below) let other agents' jobs start without the lock, so part of this run may have shared the GPU; the rate is
  labelled accordingly (it matches the step benchmark of 49.5 K minus the PPO update, so contamination was small).

**Queue bug found on the way (fixed, 18:17)**: `scripts/gpu_lock.py` (e10a80c) declared a positional `cmd` and an option
`--cmd`; the option overwrote the sub-command, so `acquire` printed the status and returned at once and every
`gpu_run.sh` job started without the lock. Fixed with `dest="job_cmd"`; reported to the coordinator.

ROUGH_PENDING

## 4. Remaining setup differences

Between MetalSim's `flat_il3` / `rough_il3` run and Isaac Lab 3.0's `Isaac-Velocity-{Flat,Rough}-G1` on
`newton_mjwarp` after this port:

1. **Collision library**: Newton's `CollisionPipeline` (explicit broad phase, contact reduction, its own narrow phase)
   vs MuJoCo Warp's collision (C-exact plane-convex set; per-triangle heightfield patch on rough). Not portable without
   porting Newton's pipeline.
2. **Contact reuse within the 5 ms tick**: emulated (`collision_every=2`) but not used for training, because with MuJoCo
   Warp's contact set it matched the recording worse than colliding every substep (§2.2).
3. **Iteration cap 20 instead of 100** in the training preset (same tolerance and per-world exit; binds only when a world
   needs more than 20 iterations: probe in §2.3). On Metal, the conditional graph node that makes the cap of 100 free on
   CUDA does not exist.
4. **Learner**: MetalSim's PPOWarp (checked against rsl_rl 3.1.2, PARITY §1.5) vs rsl_rl 5.4.1 with the same
   hyper-parameters; different random generators and noise streams.
5. **Randomness**: same distributions, different generators (torch CUDA Philox vs Warp's `rand_init`/numpy); the mass
   draw per env, push timers and reset draws are therefore not the same values.
6. **Logging statistic**: Isaac's "Mean reward" / "Mean episode length" = mean of the last 100 finished episodes;
   Episode_Reward terms = per-step means over reset envs, averaged over the iteration's steps with resets. MetalSim: means
   over the episodes finished in that iteration (≈100 per iteration once episodes are full), per-term episode sums /
   20 s over those episodes. Isaac's iterations are 0-based, PPOWarp's 1-based.
7. **Hardware and rate**: L4 (CUDA) vs M4 Max (Metal); throughput is reported, not matched.
8. **Asset build**: Isaac Sim 6.1 USD through Newton's importer vs the 5.1 USD through `metalsim.scene.usd_to_mjcf`:
   masses, inertias, COMs, ranges, armature and gains equal to 5e-6 relative (§1.3); the collision shapes (4 geoms: two
   foot boxes, torso box, plane) the same.
9. **`njmax` / `nconmax`**: Isaac's 95 / 10 (flat) could drop contacts on overflow; MetalSim's buffers do not overflow.
10. **Joint-limit solref after mass randomization**: re-scaled per world with MetalSim's own `dof_invweight0` ratio;
    Isaac's absolute `dof_invweight0` is not recorded (the nominal per-joint values are).
11. **2.3.2 ports**: still read the frame-origin root velocity (§1.2); left as they were, flagged in the ledger.


## 5. Decisions (DECISIONS.md-style rows; options archived behind flags)

| date | decision | options considered (with numbers) | chosen and why | how to re-enable the others |
|---|---|---|---|---|
| 2026-09-25 | Isaac Lab 3.0 task port | (a) disable 3.0's events on the Isaac side; (b) port push_robot, add_base_mass, ±0.5 reset velocities, the Newton contact-sensor semantics (5 ms ticks, 3-tick history, 0 N air-time threshold), the COM root velocity and the rasterized rough heightfield | (b): the reference runs already exist with the events on | `reward_cfg="flat"/"rough_isaac"` (2.3.2), `il3_events=False`, `il3_backend="physx"` (1 N threshold) |
| 2026-09-25 | Root linear velocity in the 2.3.2 ports | frame-origin velocity (as ported) vs the pelvis COM velocity (Isaac's `root_lin_vel_*` in 2.3.2 and 3.0; ω × 7.6 cm, up to ~4 cm/s) | COM for the il3 configs; the 2.3.2 configs left untouched (user rule), recorded here as a defect | kernel argument `root_com` (zero = origin) |
| 2026-09-25 | Joint limits against Isaac Lab 3.0's Newton reference | hard limits (5 ms, 0.99–0.999): random-protocol max excursion 0.068 rad / 134 steps > 0.01; Isaac's live soft limits (4 ms–0.46 s, ζ 0.03–0.35): 0.216 / 118; Isaac Newton itself 0.227 / 112 | soft (Isaac's recorded values) when the reference is Newton; hard stays the PhysX-reference choice | `solver_cfg="isaaclab3_hardlimits"`, `contact_tuning` presets |
| 2026-09-25 | Collision once per 5 ms tick | once per tick with witness-point refresh (Newton's fast path) vs every substep: every substep closer on 7 of 11 protocol rows (landing 20 ms force 2187 vs 2979 N, Isaac 2086; drop root-height RMSE 2.2 vs 5.9 mm; random divergence 0.26 vs 0.12 s), cost equal (27.2 K vs 27.5 K) | every substep (Newton's own contact set is not reproduced, so the reuse alone does not help) | `solver_cfg="isaaclab3"` (`collision_every=2`) |
| 2026-09-25 | Iteration cap for Isaac's numerics on Metal | cap 100 (Isaac's): 27.5 K env-steps/s; cap 20: 50.4 K, states equal to 1.2e-7 when no world exceeds 20 | cap 20 for training, with the probe count of worlds above 20 reported | `solver_cfg="isaaclab3_collide_every_substep"` (cap 100) |
| 2026-09-25 | Rough collision surface for the 3.0 Newton reference | exact boxes (2.3.2 rough port, matches PhysX's mesh) vs the 0.1 m heightfield (what Newton rasterizes in 3.0) | heightfield for `rough_il3` | `terrain_collision="boxes_local"` |

