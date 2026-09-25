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

RESULTS_STEP1

## 2. Isaac's own MuJoCo-Warp settings as a preset (`solver_cfg="isaaclab3"`)

PENDING_STEP2

## 3. Training like for like

PENDING_STEP3

## 4. Remaining setup differences

PENDING_DIFFS

## 5. Decisions (DECISIONS.md-style rows; options archived behind flags)

PENDING_DECISIONS
