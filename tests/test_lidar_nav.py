"""Lidar-based RL task: the planar lidar observation equals MuJoCo's mj_ray on the same state, and the
env steps, rewards and resets are finite (a MetalSim task; Isaac Lab publishes no lidar RL benchmark)."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.learn.lidar_nav import LidarNavEnv, LidarNavConfig

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


def test_scan_matches_mj_ray_and_env_runs():
    env = LidarNavEnv(LidarNavConfig(num_envs=32))
    obs = env.reset(); env.synchronize()
    assert obs.shape == (32, 68) and torch.isfinite(obs).all()
    r = env.lidar.out["range"].cpu().numpy()
    m = env.model
    robot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot"); site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    worst = 0.0
    for e in range(4):
        d = mujoco.MjData(m); d.qpos[:] = env.sim.d.qpos.numpy()[e]; mujoco.mj_forward(m, d)
        for k, (az, _) in enumerate(env.lidar.beams):
            gid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(m, d, d.site_xpos[site], np.array([np.cos(az), np.sin(az), 0.0]), np.array([1, 1, 1, 1, 0, 0], np.uint8), 1, robot, gid)
            ref = dist if 0 < dist < 8.0 else 0.0
            worst = max(worst, abs(ref - r[e, k]))
    assert worst < 1e-3
    a = torch.zeros(32, 2, device="mps")
    for _ in range(320):        # past the 300-step horizon: resets happen
        obs, rew, done, info = env.step(torch.rand_like(a) * 2 - 1)
    env.synchronize()
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all() and torch.isfinite(info["episode_reward"]).all()
    assert bool(done.any()) or int(env.t.max()) < 300
