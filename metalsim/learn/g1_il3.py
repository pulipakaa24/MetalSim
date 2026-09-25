"""Isaac Lab 3.0.0-EA's G1 velocity-task differences from 2.3.2, for ``G1VelocityTask(reward_cfg="flat_il3" |
"rough_il3")``.

Source: Isaac Lab tag v3.0.0-EA (commit ae37b028, cloned to ``upstream/IsaacLab3``); paths below are relative to
``upstream/IsaacLab3/source``. The 2.3.2 side is ``upstream/IsaacLab232`` (tag v2.3.2, 37ddf626) and
``assets/isaac/``.

Events (``isaaclab_tasks/core/velocity/velocity_env_cfg.py`` ``EventsCfg``; ``config/g1/rough_env_cfg.py``
``G1RoughEnvCfg.__post_init__``, inherited unchanged by ``flat_env_cfg.py``). 2.3.2's G1 config set
``push_robot = None``, ``add_base_mass = None`` and zero reset velocities; 3.0's no longer does:

* ``push_robot``: ``mdp.push_by_setting_velocity``, mode "interval", ``interval_range_s=(10.0, 15.0)``,
  ``velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}``. ``isaaclab/envs/mdp/events.py:2099``: the root
  COM velocity (``root_vel_w`` = ``root_com_vel_w``) gets ``+= U(range)`` per axis (unlisted axes (0, 0)),
  written back with ``write_root_velocity_to_sim_index``. Timer (``isaaclab/managers/event_manager.py``): per env
  (``is_global_time`` False), ``time_left ~ U(10, 15)`` at start and on every reset of that env
  (``resample_interval_on_reset`` True); every env step ``time_left -= step_dt`` and envs with
  ``time_left < 1e-6`` are pushed and redrawn. Order in ``ManagerBasedRLEnv.step``
  (``isaaclab/envs/manager_based_rl_env.py``): physics, terminations, rewards, resets, command update, interval
  events, observations. With 20 s episodes an episode sees at most one push.
* ``add_base_mass``: ``mdp.randomize_rigid_body_mass`` on ``torso_link`` (G1 override of the base's "base"),
  mode "startup" (once, all envs), ``mass_distribution_params=(1/1.25, 1.25)``, ``operation="scale"``,
  ``distribution="log_uniform"`` (``exp(U(log 0.8, log 1.25))``, ``isaaclab/utils/math.py`` ``sample_log_uniform``),
  ``recompute_inertia`` True by default: the inertia tensor is scaled by the same ratio (``events.py:665``);
  the COM is unchanged. On the Newton backend the mass change triggers ``SolverMuJoCo.notify_model_changed``
  (Newton 1.5.2 ``solver_mujoco.py`` ~4440): ``set_const_fixed`` + ``set_const_0`` (subtree masses,
  ``body_invweight0``, ``dof_invweight0``) and the force-space joint-limit solref re-scaled from the new
  ``dof_invweight0``. Reproduced here per world (``apply_add_base_mass``).
* ``reset_base``: ``mdp.reset_root_state_uniform`` with the base config's ranges, which 3.0's G1 no longer
  overrides: ``pose_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}``, ``velocity_range``
  ±0.5 on x, y, z, roll, pitch, yaw. ``events.py:2127``: velocities = default (0) + U(range), world frame, set as
  the root COM velocity (``write_root_velocity_to_sim_index`` docstring: "the root's center of mass rather than
  the root's frame"); orientation = default * quat_from_euler_xyz(0, 0, yaw).
* unchanged: ``physics_material`` static 0.8 / dynamic 0.6 (degenerate ranges), ``reset_robot_joints`` scale
  (1, 1) with zero velocity, ``base_external_force_torque`` zero (returns early), ``base_com`` None.

Other 3.0 differences that change the MDP, all reproduced for the il3 configs:

* contact sensor (``isaaclab_newton/sensors/contact_sensor/contact_sensor.py``): on the Newton backend
  ``force_threshold`` None becomes **0.0** (air/contact time: in contact iff |normal force| > 0), vs 1.0 on PhysX
  (``isaaclab_physx/.../contact_sensor.py:131``) and in 2.3.2; the sensor is updated once per 5 ms physics tick
  (``scene.update(dt=physics_dt)`` after each ``sim.step``; the G1's implicit actuators are not
  "all-graphable", so ``handles_decimation()`` is False and the env loops the 4 ticks) with the forces of
  the tick's last substep (``NewtonManager._update_sensors`` -> ``solver.update_contacts``), history 3 ticks.
  ``feet_slide`` and ``illegal_contact`` now read ``net_normal_forces_w_history`` (2.3.2: ``net_forces_w_history``,
  which in 2.3.2 was documented as the normal force; here the sensor's normal force in both).
* root velocity semantics (unchanged between versions, but not reproduced by the 2.3.2 ports): ``base_lin_vel``,
  ``track_lin_vel_xy_yaw_frame_exp`` and ``lin_vel_z_l2`` read ``root_lin_vel_*`` = the root body's **COM**
  velocity (3.0 ``isaaclab/assets/articulation/base_articulation_data.py:1384``; 2.3.2
  ``articulation_data.py:1041``); the 2.3.2 ports read MuJoCo's free-joint velocity (the pelvis frame origin, 7.6 cm
  above the pelvis COM). The il3 configs use the COM velocity; the 2.3.2 configs are left as they were.
* rough terrain on Newton (``isaaclab/terrains/config/rough.py``): every sub-terrain sets
  ``convert_to_heightfield=True``, so Newton rasterizes the terrain mesh to a heightfield at the 0.1 m horizontal
  scale (``isaaclab/terrains/terrain_importer.py`` ``_tag_heightfield_collider``; ``isaaclab_newton/physics/
  newton_manager.py`` ``_inject_terrain_heightfields`` -> ``newton.Heightfield.create_from_mesh``, rays cast down on
  a regular grid): stair risers and box walls become 0.1 m ramps, i.e. MetalSim's ``terrain_collision="hfield"``.
  The height scanner still casts against the mesh (exact scan). ``rough_il3`` therefore defaults to "hfield".

Unchanged (checked by diff of the two tags): reward terms and weights, observation terms and noise (``UniformNoiseCfg``
is additive, as 2.3.2's ``AdditiveUniformNoiseCfg``), command ranges and heading control, terminations (except the
force field above), curriculum ``terrain_levels_vel``, terrain generator and ``ROUGH_TERRAINS_CFG`` values, actuator
gains / armature / effort limits (``G1_CFG``: ``effort_limit_sim`` renamed ``joint_effort_limit``), decimation 4,
``sim.dt`` 0.005, episode 20 s, PPO hyper-parameters (flat 1500 iterations; rough ``max_iterations`` 5000 on
Newton, 3000 otherwise; the reference runs used 1500). Quaternions are XYZW in 3.0 but no observation contains one
(projected gravity only).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import warp as wp

PUSH_INTERVAL_S = (10.0, 15.0)
PUSH_VEL = 0.5                   # push_robot velocity_range x, y: (-0.5, 0.5)
RESET_VEL = 0.5                  # reset_base velocity_range, all six axes
RESET_YAW = 3.14                 # reset_base pose_range yaw (-3.14, 3.14)
RESET_XY = 0.5
MASS_SCALE = (1.0 / 1.25, 1.25)  # add_base_mass, log-uniform, operation "scale"
MASS_BODY = "torso_link"
SENSOR_TICK_S = 0.005            # contact sensor update period (sim.dt) on both 3.0 backends
SENSOR_HISTORY = 3               # ContactSensorCfg history_length
AIR_TIME_THRESHOLD = {"newton_mjwarp": 0.0, "physx": 1.0}
PUSH_EPS = 1e-6                  # EventManager: time_left < 1e-6


@wp.func
def _quat_mat(q: wp.vec4) -> wp.mat33:
    w = q[0]; x = q[1]; y = q[2]; z = q[3]
    return wp.mat33(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))


@wp.kernel
def g1_reset_il3(reset_mask: wp.array(dtype=wp.bool), default_q: wp.array(dtype=float), origins: wp.array2d(dtype=float),
                 seed: int, step_idx: wp.array(dtype=int), root_com: wp.vec3, vel_range: float, yaw_range: float, xy_range: float,
                 push_lo: float, push_hi: float, push_left: wp.array(dtype=float),
                 qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                 last_action: wp.array2d(dtype=float), prev_action: wp.array2d(dtype=float)):
    """Isaac Lab 3.0 resets of the G1 velocity task: reset_root_state_uniform (xy +-0.5 m, yaw +-3.14, root COM velocity
    U(+-0.5) on the six world axes), reset_joints_by_scale (1, 1) with zero joint velocity, and the push_robot timer
    redrawn in U(10, 15) s (EventManager.reset). MuJoCo's free joint takes the frame origin's world linear velocity
    and the body-frame angular velocity: v_origin = v_com - w x (R r_com), w_body = R^T w."""
    e = wp.tid()
    if not reset_mask[e]:
        return
    rng = wp.rand_init(seed + 11, step_idx[0] * 6151 + e)
    for i in range(qpos.shape[1]):
        qpos[e, i] = default_q[i]
    for i in range(qvel.shape[1]):
        qvel[e, i] = 0.0
    qpos[e, 0] = origins[e, 0] + wp.randf(rng, -xy_range, xy_range)
    qpos[e, 1] = origins[e, 1] + wp.randf(rng, -xy_range, xy_range)
    qpos[e, 2] = origins[e, 2] + default_q[2]
    yaw = wp.randf(rng, -yaw_range, yaw_range)
    q = wp.vec4(wp.cos(0.5 * yaw), 0.0, 0.0, wp.sin(0.5 * yaw))
    qpos[e, 3] = q[0]; qpos[e, 4] = q[1]; qpos[e, 5] = q[2]; qpos[e, 6] = q[3]
    v_com = wp.vec3(wp.randf(rng, -vel_range, vel_range), wp.randf(rng, -vel_range, vel_range), wp.randf(rng, -vel_range, vel_range))
    w_w = wp.vec3(wp.randf(rng, -vel_range, vel_range), wp.randf(rng, -vel_range, vel_range), wp.randf(rng, -vel_range, vel_range))
    R = _quat_mat(q)
    v_o = v_com - wp.cross(w_w, R * root_com)
    w_b = wp.transpose(R) * w_w
    qvel[e, 0] = v_o[0]; qvel[e, 1] = v_o[1]; qvel[e, 2] = v_o[2]
    qvel[e, 3] = w_b[0]; qvel[e, 4] = w_b[1]; qvel[e, 5] = w_b[2]
    for i in range(last_action.shape[1]):
        last_action[e, i] = 0.0; prev_action[e, i] = 0.0
    push_left[e] = wp.randf(rng, push_lo, push_hi)


@wp.kernel
def g1_push_il3(seed: int, step_idx: wp.array(dtype=int), step_dt: float, push_lo: float, push_hi: float, vel_range: float,
                push_left: wp.array(dtype=float), qvel: wp.array2d(dtype=float), pushes: wp.array(dtype=int)):
    """push_robot (interval 10-15 s, per env): time_left -= step_dt; below 1e-6 the root COM velocity gets
    += U(+-0.5) in world x and y and the timer is redrawn. The angular velocity is unchanged, so the frame origin's
    velocity changes by the same vector."""
    e = wp.tid()
    t = push_left[e] - step_dt
    if t < 1.0e-6:
        rng = wp.rand_init(seed + 41, step_idx[0] * 7907 + e)
        t = wp.randf(rng, push_lo, push_hi)
        qvel[e, 0] = qvel[e, 0] + wp.randf(rng, -vel_range, vel_range)
        qvel[e, 1] = qvel[e, 1] + wp.randf(rng, -vel_range, vel_range)
        wp.atomic_add(pushes, 0, 1)
    push_left[e] = t


@wp.kernel
def g1_term_episode_sums(step_idx: wp.array(dtype=int), buf_done: wp.array2d(dtype=float), terms: wp.array2d(dtype=float),
                         dt: float, episode_s: float, ep_terms: wp.array2d(dtype=float), term_stats: wp.array(dtype=float),
                         term_count: wp.array(dtype=int)):
    """Isaac's RewardManager episode sums per term (value x dt), reported at the episode's end divided by the
    maximum episode length in seconds (Episode_Reward/<term>); summed here over the episodes ending this iteration."""
    e = wp.tid()
    s = (step_idx[0] - 1) % buf_done.shape[0]
    nk = terms.shape[1]
    for k in range(nk):
        ep_terms[e, k] = ep_terms[e, k] + terms[e, k] * dt
    if buf_done[s, e] > 0.5:
        for k in range(nk):
            wp.atomic_add(term_stats, k, ep_terms[e, k] / episode_s)
            ep_terms[e, k] = 0.0
        wp.atomic_add(term_count, 0, 1)
        if terms[e, 7] < 0.0:
            wp.atomic_add(term_count, 1, 1)


TERM_NAMES = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "feet_slide", "joint_deviation",
              "flat_orientation_l2", "action_rate_l2", "termination_penalty", "lin_vel_z_l2", "ang_vel_xy_l2",
              "dof_torques_l2", "dof_acc_l2", "dof_pos_limits"]


def sample_mass_scale(n: int, seed: int) -> np.ndarray:
    """log-uniform in [1/1.25, 1.25] (exp of U(log lo, log hi)), float32 like torch."""
    rng = np.random.default_rng(int(seed) + 1_000_003)
    lo, hi = MASS_SCALE
    return np.exp(rng.uniform(math.log(lo), math.log(hi), n)).astype(np.float32)


def apply_add_base_mass(sim, model: mujoco.MjModel, scale: np.ndarray, force_space_limits: bool) -> dict:
    """add_base_mass on a BatchSim whose per_world_fields include body_mass, body_inertia, body_subtreemass,
    body_invweight0, dof_invweight0 (+ jnt_solref when ``force_space_limits``): torso mass and inertia x scale per
    world, MuJoCo Warp's set_const (subtree masses, invweights), stat.meaninertia restored to the nominal value
    (a shared field every world would otherwise overwrite), and, for Newton's force-space joint limits, the limit
    solref re-scaled as Newton's update_jnt_solref_from_invweight0 does: timeconst ∝ 1 / dof_invweight0,
    dampratio ∝ sqrt(dof_invweight0)."""
    import mujoco_warp as mjw
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MASS_BODY)
    m = sim.m
    n = sim.n
    mass = m.body_mass.numpy(); inertia = m.body_inertia.numpy()
    assert mass.shape[0] == n and inertia.shape[0] == n, "body_mass / body_inertia must be per-world fields"
    mass[:, b] = model.body_mass[b] * scale
    inertia[:, b] = model.body_inertia[b][None, :] * scale[:, None]
    m.body_mass.assign(mass); m.body_inertia.assign(inertia)
    invw0 = m.dof_invweight0.numpy().copy()
    meaninertia = m.stat.meaninertia.numpy().copy()
    with wp.ScopedDevice(sim.device):
        mjw.set_const(m, sim.d)
    wp.synchronize_device(sim.device)
    m.stat.meaninertia.assign(meaninertia)
    invw = m.dof_invweight0.numpy()
    out = {"body": MASS_BODY, "scale_min": float(scale.min()), "scale_max": float(scale.max()),
           "dof_invweight0_ratio_range": [float((invw / np.maximum(invw0, 1e-12)).min()), float((invw / np.maximum(invw0, 1e-12)).max())]}
    if force_space_limits:
        sr = m.jnt_solref.numpy()
        assert sr.shape[0] == n, "jnt_solref must be a per-world field for force-space limits"
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof = model.jnt_dofadr[j]
            r = invw[:, dof] / invw0[:, dof]
            sr[:, j, 0] = model.jnt_solref[j, 0] / r
            sr[:, j, 1] = model.jnt_solref[j, 1] * np.sqrt(r)
        m.jnt_solref.assign(sr)
        out["limit_timeconst_ratio_range"] = [float((sr[:, 1:, 0] / model.jnt_solref[None, 1:, 0]).min()),
                                              float((sr[:, 1:, 0] / model.jnt_solref[None, 1:, 0]).max())]
    wp.synchronize_device(sim.device)
    return out
