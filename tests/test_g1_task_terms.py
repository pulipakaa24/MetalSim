"""G1 task terms recomputed from Isaac Lab's formulas (isaaclab mdp rewards / observations, as
transcribed in assets/isaac/g1_rewards.py and g1_velocity_env_cfg.py) against the Warp kernels'
per-term outputs on the live simulation state."""
import mujoco
import numpy as np
import pytest
import importlib.util

import torch
import warp as wp

from metalsim.learn.g1_velocity import G1VelocityTask, ACTION_SCALE, CONTROL_DT, benchmark_step, g1_foot_vel_mjwarp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

# every invariant below must hold on both physics engines (same kernels, same MuJoCo-layout state)
ENGINES = [pytest.param({}, id="mjwarp"),
           pytest.param({"engine": "newton", "newton_iterations": 4, "newton_dt": 0.00125}, id="newton",
                        marks=pytest.mark.skipif(importlib.util.find_spec("newton") is None,
                                                 reason="Newton not installed (optional: pip install -e '.[newton]')"))]


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _one_step(task, n, seed_actions=3):
    """Random actions for a few steps, then one control step by hand, stopping before the command update so
    cmd is what the reward saw. Returns the done flags of that step."""
    benchmark_step(task, num_frames=6, warmup=0)     # random actions
    from metalsim.learn.warp_policy import RolloutBuffers, bump
    class _Pol:
        step_idx = wp.zeros(1, dtype=int, device="metal:0")
    pol = _Pol(); wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0")
    bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, n), dtype=float, device="metal:0")
    a = wp.array(np.random.default_rng(seed_actions).uniform(-1, 1, (n, task.act_dim)).astype(np.float32), dtype=float, device="metal:0")
    task.launch_apply_action(a); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs)
    task.sim.synchronize()
    return bufs.done.numpy()[0] > 0.5


def _isaac_terms(task, e, flat, rough_isaac=False):
    """Isaac Lab's reward terms (weights applied, before x dt) recomputed in numpy from the live state:
    G1FlatEnvCfg (flat), G1RoughEnvCfg exactly (rough_isaac) or G1RoughEnvCfg as first ported (neither)."""
    exact = flat or rough_isaac
    d = task.sim.d
    qpos = d.qpos.numpy()[e]; qvel = d.qvel.numpy()[e]; cmd = task.cmd.numpy()[e]
    last = task.last_action.numpy()[e]; prev = task.prev_action.numpy()[e]
    default = task.default_q.numpy(); group = task.group.numpy()
    sd = d.sensordata.numpy()[e]; qacc = d.qacc.numpy()[e]; tau = d.qfrc_actuator.numpy()[e]
    fv = (d.foot_vel if task.engine == "newton" else task.foot_vel).numpy()[e]      # what the reward kernel read
    R = _rot(qpos[3:7])
    v_w = qvel[0:3]; w_b = qvel[3:6]; w_w = R @ w_b; v_b = R.T @ v_w
    yaw = np.arctan2(R[1, 0], R[0, 0])
    v_yaw = np.array([np.cos(yaw) * v_w[0] + np.sin(yaw) * v_w[1], -np.sin(yaw) * v_w[0] + np.cos(yaw) * v_w[1]])
    t = {}
    # track_lin_vel_xy_yaw_frame_exp (std 0.5) weight 1.0; track_ang_vel_z_world_exp (std 0.5) weight 1.0 flat / 2.0 rough
    t[0] = 1.0 * np.exp(-np.sum((cmd[:2] - v_yaw) ** 2) / 0.25)
    t[1] = (1.0 if flat else 2.0) * np.exp(-((cmd[2] - w_w[2]) ** 2) / 0.25)
    # feet_air_time_positive_biped (threshold 0.4): flat 0.75 x min over feet of the in-mode time where exactly
    # one foot is in contact, zero when |command[:, :2]| <= 0.1
    in_c = np.array([sd[task.touch_adr[f]] > 1.0 for f in range(2)])
    air = task.air_time.numpy()[e]; con = task.contact_time.numpy()[e]
    slide_c = in_c
    if getattr(task, "contact", None) is not None:
        # Isaac's ContactSensor data: in_contact = current_contact_time > 0; feet_slide contact = max over the
        # net_forces_w_history norm > 1 N
        cs = task.contact.numpy()
        air = cs["current_air_time"][e, :2]; con = cs["current_contact_time"][e, :2]
        in_c = con > 0.0
        slide_c = np.linalg.norm(cs["net_forces_w_history"][e, :, :2], axis=-1).max(0) > 1.0
    mode = np.where(in_c, con, air)
    if exact:
        single = in_c.sum() == 1
        t[2] = (0.75 if flat else 0.25) * min(np.min(np.where(single, mode, 0.0)), 0.4) * float(np.linalg.norm(cmd[:2]) > 0.1)
    else:
        t[2] = 0.25 * np.sum(np.minimum(mode, 0.4)) * float(in_c.sum() == 1 and np.linalg.norm(cmd) > 0.1)
    # feet_slide -0.1: |body_lin_vel_w[foot].xy| summed over feet in contact
    t[3] = sum(-0.1 * np.linalg.norm(fv[f, :2]) for f in range(2) if slide_c[f])
    # joint_deviation_l1: hips yaw/roll -0.1, torso -0.1, arms -0.1, fingers -0.05
    dq = qpos[7:] - default[7:]
    t[4] = -0.1 * np.sum(np.abs(dq[np.isin(group, [0, 3, 4])])) - 0.05 * np.sum(np.abs(dq[group == 5]))
    # flat_orientation_l2 -1.0 (projected gravity xy); action_rate_l2 -0.005
    g_b = R.T @ np.array([0, 0, -1.0])
    t[5] = -1.0 * np.sum(g_b[:2] ** 2)
    t[6] = -0.005 * np.sum((last - prev) ** 2)
    hk = group <= 1                                  # ".*_hip_.*", ".*_knee_joint"
    if flat:
        t[8] = -0.2 * v_b[2] ** 2                    # lin_vel_z_l2 on root_lin_vel_b
        t[9] = -0.05 * np.sum(w_b[:2] ** 2)          # ang_vel_xy_l2 on root_ang_vel_b
        t[10] = -2.0e-6 * np.sum(tau[6:][hk] ** 2)   # dof_torques_l2 (hips, knees)
        t[11] = -1.0e-7 * np.sum(qacc[6:][hk] ** 2)  # dof_acc_l2 (hips, knees)
    elif rough_isaac:                                # G1RoughEnvCfg: lin_vel_z 0, ang_vel_xy body frame, rough weights
        t[8] = 0.0
        t[9] = -0.05 * np.sum(w_b[:2] ** 2)
        t[10] = -1.5e-7 * np.sum(tau[6:][group <= 2] ** 2)   # hips, knees, ankles
        t[11] = -1.25e-7 * np.sum(qacc[6:][hk] ** 2)
    else:
        t[8] = 0.0
        t[9] = -0.05 * np.sum(w_w[:2] ** 2)
        t[10] = -1.5e-7 * np.sum(tau[6:][group <= 2] ** 2)
        t[11] = -1.25e-7 * np.sum(qacc[6:][hk] ** 2)
    # joint_pos_limits -1.0 on the ankles: soft limits (G1_CFG soft_joint_pos_limit_factor 0.9) for the flat set
    m = task.model
    jr = np.array([m.jnt_range[m.actuator_trnid[i][0]] for i in range(m.nu)])
    if exact:
        mid = jr.mean(1); half = 0.5 * (jr[:, 1] - jr[:, 0]); jr = np.stack([mid - 0.9 * half, mid + 0.9 * half], 1)
    q = qpos[7:]; an = group == 2
    t[12] = -1.0 * np.sum(np.maximum(jr[an, 0] - q[an], 0.0) + np.maximum(q[an] - jr[an, 1], 0.0))
    return t


@pytest.mark.parametrize("engine", ENGINES)
def test_reward_terms_match_isaac_formulas(engine):
    """Flat terrain uses Isaac's G1FlatEnvCfg: every one of the 13 terms the kernel exposes equals Isaac's formula x weight."""
    n = 8
    task = G1VelocityTask(n, terrain="flat", seed=1, **engine)
    assert task.reward_cfg == "flat"
    done = _one_step(task, n)
    terms = task.terms.numpy()
    tol = {10: (1e-3, 1e-6), 11: (1e-3, 1e-6)}           # squared torques / accelerations: float32 sums
    checked_air = 0
    for e in range(n):
        ref = _isaac_terms(task, e, flat=True)
        for k, v in ref.items():
            if k in (2, 3) and done[e]:
                continue                                  # finished envs: timers / contact sensor already reset
            rtol, atol = tol.get(k, (1e-4, 1e-5))
            np.testing.assert_allclose(terms[e, k], v, rtol=rtol, atol=atol, err_msg=f"env {e} term {k}")
            checked_air += int(k == 2)
        assert terms[e, 7] in (0.0, -200.0)                 # is_terminated, weight -200 (x dt in the sum like every term)
    assert checked_air > 0
    assert np.all(np.abs(terms[:, 0]) <= 1.0) and np.all(np.abs(terms[:, 1]) <= 1.0)


def test_rough_reward_set_unchanged():
    """The rough set as first ported (also what flat-terrain runs used before the G1FlatEnvCfg port)."""
    n = 8
    task = G1VelocityTask(n, terrain="flat", seed=1, reward_cfg="rough")
    done = _one_step(task, n)
    terms = task.terms.numpy()
    for e in range(n):
        ref = _isaac_terms(task, e, flat=False)
        for k, v in ref.items():
            if k == 2 and done[e]:
                continue
            rtol, atol = (1e-3, 1e-6) if k in (10, 11) else (1e-4, 1e-5)
            np.testing.assert_allclose(terms[e, k], v, rtol=rtol, atol=atol, err_msg=f"env {e} term {k}")


def test_rough_isaac_reward_set_matches_isaac_formulas():
    """reward_cfg="rough_isaac" = Isaac's G1RoughEnvCfg exactly: all 13 terms against Isaac's formulas with the rough
    weights, commands with lin_vel_y = (0, 0), ContactSensor-based contact terms (MuJoCo Warp)."""
    n = 8
    task = G1VelocityTask(n, terrain="flat", seed=1, reward_cfg="rough_isaac")
    assert task.isaac_flat == 2 and task.lin_vel_y == 0.0 and task.contact is not None
    done = _one_step(task, n)
    terms = task.terms.numpy()
    checked_air = 0
    for e in range(n):
        ref = _isaac_terms(task, e, flat=False, rough_isaac=True)
        for k, v in ref.items():
            if k in (2, 3) and done[e]:
                continue
            rtol, atol = (1e-3, 1e-6) if k in (10, 11) else (1e-4, 1e-5)
            np.testing.assert_allclose(terms[e, k], v, rtol=rtol, atol=atol, err_msg=f"env {e} term {k}")
            checked_air += int(k == 2)
    assert checked_air > 0
    task.reset_all(); idx = wp.zeros(1, dtype=int, device="metal:0")
    task.resample.assign(np.ones(n, bool)); task.launch_obs(idx); task.sim.synchronize()
    assert np.all(task.cmd.numpy()[:, 1] == 0.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_flat_commands_and_contact_history_termination(engine):
    """G1FlatEnvCfg command ranges (lin_vel_x in [0, 1], lin_vel_y in [-0.5, 0.5]) and the torso termination on the
    max force over the contact-history window (Isaac: 3 physics steps of 5 ms; MuJoCo Warp here: the last
    round(15 ms / dt) substeps; Newton: the final substep)."""
    n = 256
    task = G1VelocityTask(n, terrain="flat", seed=4, **engine)
    task.reset_all()
    idx = wp.zeros(1, dtype=int, device="metal:0")
    task.launch_obs(idx); task.sim.synchronize()
    cmd = task.cmd.numpy(); st = task.standing.numpy()
    assert np.all(np.abs(cmd[~st, 1]) <= 0.5) and np.abs(cmd[~st, 1]).max() > 0.4
    assert np.all((cmd[~st, 0] >= 0.0) & (cmd[~st, 0] <= 1.0))
    if task.engine != "mjwarp":
        return                                            # Newton: only the final substep's contact force is available
    assert task.hist_substeps == int(round(0.015 / task.physics_dt))
    # random actions until robots fall: the termination fires exactly where the window max exceeds 1 N (the window
    # includes the final substep; finished envs are reset afterwards, which clears sensordata but not the history)
    from metalsim.learn.warp_policy import RolloutBuffers, bump
    class _Pol:
        step_idx = wp.zeros(1, dtype=int, device="metal:0")
    pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, n), dtype=float, device="metal:0")
    rng = np.random.default_rng(10); fired = 0
    for k in range(80):                                   # random actions: most robots fall within ~40 steps
        a = wp.array(rng.standard_normal((n, task.act_dim)).astype(np.float32), dtype=float, device="metal:0")
        wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0")
        task.launch_apply_action(a); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
        task.sim.synchronize()
        h = task.torso_hist.numpy(); term = task.terms.numpy()[:, 7]
        np.testing.assert_array_equal(term < 0, h > 1.0)
        fired += int((h > 1.0).sum())
    assert fired > 0


@pytest.mark.parametrize("engine", ENGINES)
def test_observation_layout_and_action_mapping(engine):
    n = 4
    task = G1VelocityTask(n, terrain="flat", seed=2, **engine)
    task.reset_all()
    idx = wp.zeros(1, dtype=int, device="metal:0")
    task.launch_obs(idx); task.sim.synchronize()
    obs = task.obs.numpy(); qpos = task.sim.d.qpos.numpy(); qvel = task.sim.d.qvel.numpy(); default = task.default_q.numpy()
    nj = task.nj
    assert obs.shape == (n, 12 + 3 * nj)
    # joint_pos_rel (noise +-0.01), joint_vel_rel (noise +-1.5), last action (no noise)
    assert np.all(np.abs(obs[:, 12:12 + nj] - (qpos[:, 7:] - default[7:])) <= 0.01 + 1e-6)
    assert np.all(np.abs(obs[:, 12 + nj:12 + 2 * nj] - qvel[:, 6:]) <= 1.5 + 1e-6)
    assert np.all(obs[:, 12 + 2 * nj:] == task.last_action.numpy())
    # projected gravity of an upright base is (0, 0, -1) up to +-0.05 noise
    assert np.all(np.abs(obs[:, 6:8]) <= 0.05 + 1e-6) and np.all(np.abs(obs[:, 8] + 1.0) <= 0.05 + 1e-6)
    # JointPositionAction: target = default + 0.5 * action, written to ctrl (position actuators)
    a = wp.array(np.random.default_rng(0).uniform(-1, 1, (n, nj)).astype(np.float32), dtype=float, device="metal:0")
    task.launch_apply_action(a); task.sim.synchronize()
    np.testing.assert_allclose(task.sim.d.ctrl.numpy(), default[7:] + ACTION_SCALE * a.numpy(), atol=1e-6)
    np.testing.assert_allclose(task.last_action.numpy(), a.numpy(), atol=1e-6)
    assert abs(task.decimation * task.physics_dt - CONTROL_DT) < 1e-9     # 50 Hz control on either engine
    assert task.max_t == 1000                                 # 20 s episodes at 50 Hz


def _foot_vel_now(task):
    """The foot velocities the reward kernel receives, for the task's current state."""
    if task.engine == "newton":
        # XPBD's maximal-coordinate state is not exactly a reduced-coordinate state (unconverged joint
        # constraints: feet up to ~1.6 cm / ~0.4 m/s off the joint-space reconstruction under random actions),
        # so make the bodies consistent with the reported qpos/qvel first (the task's own reset path: FK),
        # then read foot_vel through the normal output conversion
        sim = task.sim
        sim._reset_mask.fill_(True); sim.launch_reset()
        with wp.ScopedDevice("metal:0"):
            sim._sync_out()
        sim.synchronize()
        return sim.d.foot_vel.numpy()
    d = task.sim.d
    task.sim.forward(); task.sim.synchronize()             # MuJoCo Warp: cvel/xpos/subtree_com of the current qpos/qvel
    wp.launch(g1_foot_vel_mjwarp, dim=task.n, inputs=[d.cvel, d.xipos if task.feet_slide_velocity == "com" else d.xpos,
                                                      d.subtree_com, task.foot_body, task.foot_root],
              outputs=[task.foot_vel], device="metal:0")
    task.sim.synchronize()
    return task.foot_vel.numpy()


@pytest.mark.parametrize("engine", ENGINES)
def test_feet_slide_velocity_is_the_foot_body_world_velocity(engine):
    """Isaac's feet_slide reads body_lin_vel_w of the feet, which in Isaac Lab 2.3.2 is the body's centre-of-mass
    linear velocity in the world frame (body_com_lin_vel_w). On a moving G1 (random actions), the velocity handed
    to the reward kernel equals MuJoCo C's mj_objectVelocity(mjOBJ_BODY, flg_local=0) (the COM's) for the same
    qpos/qvel; with feet_slide_velocity="origin" (MuJoCo Warp only) it is the frame origin's (mjOBJ_XBODY).
    The archived Newton path reports the frame origin."""
    n = 4
    variants = [("com", mujoco.mjtObj.mjOBJ_BODY), ("origin", mujoco.mjtObj.mjOBJ_XBODY)] if not engine else [(None, mujoco.mjtObj.mjOBJ_XBODY)]
    for conv, obj in variants:
        kw = dict(engine) if engine else {"feet_slide_velocity": conv}
        task = G1VelocityTask(n, terrain="flat", seed=6, **kw)
        benchmark_step(task, num_frames=8, warmup=0)            # random actions in [-1, 1]: robots moving
        qpos = task.sim.d.qpos.numpy().astype(np.float64); qvel = task.sim.d.qvel.numpy().astype(np.float64)
        fv = _foot_vel_now(task)
        m = task.model; d = mujoco.MjData(m)
        assert np.abs(fv).max() > 0.05                          # feet actually moving
        for e in range(n):
            d.qpos[:] = qpos[e]; d.qvel[:] = qvel[e]; mujoco.mj_forward(m, d)
            for f in range(2):
                v6 = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, obj, int(task.foot_body[f]), v6, 0)
                np.testing.assert_allclose(fv[e, f], v6[3:], atol=2e-3, err_msg=f"{conv}")


def test_contact_sensor_flags_match_touch_sites():
    """Flat set on MuJoCo Warp: the feet contact flags come from the ContactSensor (Isaac's net normal force,
    contact iff |F| > 1 N). On a G1 dropped from the initial pose and standing (zero actions), they equal
    the touch-site flags (touch > 1 N) the task used before, every substep-final step, every env."""
    n = 64
    task = G1VelocityTask(n, terrain="flat", seed=3)
    assert task.contact is not None and task.contact.T == task.hist_substeps
    task.reset_all()
    a = wp.zeros((n, task.act_dim), dtype=float, device="metal:0")
    seen_air = seen_contact = 0
    for k in range(100):
        task.launch_apply_action(a); task.sim.launch_step(); task.sim.synchronize()
        cs = task.contact.numpy(); sd = task.sim.d.sensordata.numpy()
        f_sensor = np.linalg.norm(cs["net_forces_w"][:, :2], axis=-1) > 1.0
        f_touch = np.stack([sd[:, task.touch_adr[f]] for f in range(2)], 1) > 1.0
        np.testing.assert_array_equal(f_sensor, f_touch, err_msg=f"step {k}")
        # Isaac's air/contact timers agree with the flags
        np.testing.assert_array_equal(cs["current_contact_time"][:, :2] > 0.0, f_sensor)
        seen_air += int((~f_sensor).sum()); seen_contact += int(f_sensor.sum())
    assert seen_contact > 0.9 * 100 * 2 * n * 0.5 and seen_air >= 0


def test_rough_terrain_seed_and_isaac_env_assignment():
    """Rough terrain: default seed 42 (Isaac's rsl_rl default, from which Isaac generates the terrain), terrain
    generated from the task seed, env i on column floor(i / (n / num_cols)) (Isaac's TerrainImporter),
    initial levels in [0, max_init_terrain_level = 5]."""
    from metalsim.learn.terrain import isaac_rough_terrain
    n = 64
    task = G1VelocityTask(n, terrain="rough")
    assert task.seed == 42
    ref = isaac_rough_terrain(seed=42)
    np.testing.assert_array_equal(task.hfield["H"], ref["H"])
    nc = task.hfield["num_cols"]
    # Isaac TerrainImporter._compute_env_origins_curriculum, verbatim: torch's floor division (fmod-based, float32)
    # puts e.g. env 16 of 64 over 20 columns on column 4, where exact arithmetic says 5
    ref_cols = torch.div(torch.arange(n), (n / nc), rounding_mode="floor").to(torch.long).numpy()
    assert ref_cols[16] == 4 and ref_cols[15] == 4 and ref_cols[17] == 5
    np.testing.assert_array_equal(task.col.numpy(), ref_cols)
    lv = task.level.numpy()
    assert lv.min() >= 0 and lv.max() <= 5 and len(np.unique(lv)) > 1
    tab = task.hfield["origin_table"]()
    np.testing.assert_allclose(task.origins.numpy(), tab[lv, task.col.numpy()], atol=1e-6)
    assert G1VelocityTask(4, terrain="flat").seed == 0
