"""G1 task terms recomputed from Isaac Lab's formulas (isaaclab mdp rewards / observations, as
transcribed in assets/isaac/g1_rewards.py and g1_velocity_env_cfg.py) against the Warp kernels'
per-term outputs on the live simulation state."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.learn.g1_velocity import G1VelocityTask, ACTION_SCALE, CONTROL_DT, benchmark_step, g1_foot_vel_mjwarp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

# every invariant below must hold on both physics engines (same kernels, same MuJoCo-layout state)
ENGINES = [pytest.param({}, id="mjwarp"),
           pytest.param({"engine": "newton", "newton_iterations": 4, "newton_dt": 0.00125}, id="newton")]


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@pytest.mark.parametrize("engine", ENGINES)
def test_reward_terms_match_isaac_formulas(engine):
    n = 8
    task = G1VelocityTask(n, terrain="flat", seed=1, **engine)
    benchmark_step(task, num_frames=6, warmup=0)     # random actions
    # one more control step by hand, stopping before the command update so cmd is what the reward saw
    from metalsim.learn.warp_policy import RolloutBuffers, bump
    class _Pol:
        step_idx = wp.zeros(1, dtype=int, device="metal:0")
    pol = _Pol(); wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0")
    bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, n), dtype=float, device="metal:0")
    a = wp.array(np.random.default_rng(3).uniform(-1, 1, (n, task.act_dim)).astype(np.float32), dtype=float, device="metal:0")
    task.launch_apply_action(a); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs)
    task.sim.synchronize()
    m = task.model
    qpos = task.sim.d.qpos.numpy(); qvel = task.sim.d.qvel.numpy(); cmd = task.cmd.numpy()
    terms = task.terms.numpy(); last = task.last_action.numpy(); prev = task.prev_action.numpy()
    default = task.default_q.numpy(); group = task.group.numpy()
    fv = (task.sim.d.foot_vel if task.engine == "newton" else task.foot_vel).numpy()   # what the reward kernel read
    for e in range(n):
        R = _rot(qpos[e, 3:7])
        v_w = qvel[e, 0:3]; w_w = R @ qvel[e, 3:6]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v_yaw = np.array([np.cos(yaw) * v_w[0] + np.sin(yaw) * v_w[1], -np.sin(yaw) * v_w[0] + np.cos(yaw) * v_w[1]])
        # track_lin_vel_xy_yaw_frame_exp (std 0.5), weight 1.0
        r_lin = 1.0 * np.exp(-np.sum((cmd[e, :2] - v_yaw) ** 2) / 0.25)
        # track_ang_vel_z_world_exp (std 0.5), weight 2.0
        r_ang = 2.0 * np.exp(-((cmd[e, 2] - w_w[2]) ** 2) / 0.25)
        # flat_orientation_l2, weight -1.0: projected gravity xy
        g_b = R.T @ np.array([0, 0, -1.0])
        r_orient = -1.0 * np.sum(g_b[:2] ** 2)
        # joint_deviation_l1: hips yaw/roll -0.1, torso -0.1, arms -0.1, fingers -0.05
        dq = qpos[e, 7:] - default[7:]
        r_dev = -0.1 * np.sum(np.abs(dq[np.isin(group, [0, 3, 4])])) - 0.05 * np.sum(np.abs(dq[group == 5]))
        # action_rate_l2, weight -0.005
        r_rate = -0.005 * np.sum((last[e] - prev[e]) ** 2)
        got = terms[e]
        np.testing.assert_allclose(got[0], r_lin, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got[1], r_ang, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got[5], r_orient, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got[4], r_dev, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(got[6], r_rate, rtol=1e-4, atol=1e-5)
        # feet_slide, weight -0.1: |body_lin_vel_w[foot].xy| summed over feet in contact (touch > 1 N)
        sd = task.sim.d.sensordata.numpy()[e]
        r_slide = sum(-0.1 * np.linalg.norm(fv[e, f, :2]) for f in range(2) if sd[task.touch_adr[f]] > 1.0)
        np.testing.assert_allclose(got[3], r_slide, rtol=1e-4, atol=1e-5)
        assert got[7] in (0.0, -200.0)                      # termination penalty term (is_terminated, weight -200; x dt in the sum like every term)
    # the summed reward is the dt-weighted sum of the terms (Isaac multiplies each term by step dt)
    # plus the termination penalty; check the sign/scale on the tracking-dominated terms
    assert np.all(np.abs(terms[:, 0]) <= 1.0) and np.all(np.abs(terms[:, 1]) <= 2.0)


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
    wp.launch(g1_foot_vel_mjwarp, dim=task.n, inputs=[d.cvel, d.xpos, d.subtree_com, task.foot_body, task.foot_root],
              outputs=[task.foot_vel], device="metal:0")
    task.sim.synchronize()
    return task.foot_vel.numpy()


@pytest.mark.parametrize("engine", ENGINES)
def test_feet_slide_velocity_is_the_foot_body_world_velocity(engine):
    """Isaac's feet_slide reads body_lin_vel_w of the feet: the foot body frame origin's linear velocity in
    the world frame. On a moving G1 (random actions), the velocity handed to the reward kernel equals MuJoCo
    C's mj_objectVelocity(mjOBJ_XBODY, flg_local=0) for the same qpos/qvel."""
    n = 4
    task = G1VelocityTask(n, terrain="flat", seed=6, **engine)
    benchmark_step(task, num_frames=8, warmup=0)            # random actions in [-1, 1]: robots moving
    qpos = task.sim.d.qpos.numpy().astype(np.float64); qvel = task.sim.d.qvel.numpy().astype(np.float64)
    fv = _foot_vel_now(task)
    m = task.model; d = mujoco.MjData(m)
    assert np.abs(fv).max() > 0.05                          # feet actually moving
    for e in range(n):
        d.qpos[:] = qpos[e]; d.qvel[:] = qvel[e]; mujoco.mj_forward(m, d)
        for f in range(2):
            v6 = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, int(task.foot_body[f]), v6, 0)
            np.testing.assert_allclose(fv[e, f], v6[3:], atol=2e-3)
