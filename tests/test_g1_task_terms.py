"""G1 task terms recomputed from Isaac Lab's formulas (isaaclab mdp rewards / observations, as
transcribed in assets/isaac/g1_rewards.py and g1_velocity_env_cfg.py) against the Warp kernels'
per-term outputs on the live simulation state."""
import numpy as np
import pytest
import torch
import warp as wp

from orchard.learn.g1_velocity import G1VelocityTask, ACTION_SCALE, CONTROL_DT, benchmark_step

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def test_reward_terms_match_isaac_formulas():
    n = 8
    task = G1VelocityTask(n, terrain="flat", seed=1)
    benchmark_step(task, num_frames=6, warmup=0)     # random actions
    # one more control step by hand, stopping before the command update so cmd is what the reward saw
    from orchard.learn.warp_policy import RolloutBuffers, bump
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
        assert got[7] in (0.0, -200.0)                      # termination penalty (is_terminated, -200)
    # the summed reward is the dt-weighted sum of the terms (Isaac multiplies each term by step dt)
    # plus the termination penalty; check the sign/scale on the tracking-dominated terms
    assert np.all(np.abs(terms[:, 0]) <= 1.0) and np.all(np.abs(terms[:, 1]) <= 2.0)


def test_observation_layout_and_action_mapping():
    n = 4
    task = G1VelocityTask(n, terrain="flat", seed=2)
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
    assert task.decimation == 4 and abs(task.decimation * task.model.opt.timestep - CONTROL_DT) < 1e-9
    assert task.max_t == 1000                                 # 20 s episodes at 50 Hz
