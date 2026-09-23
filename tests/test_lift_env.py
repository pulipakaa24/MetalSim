"""Phase 2 environment: the GPU SO-101 lift task runs end to end with no host synchronization."""
import time

import numpy as np
import pytest
import torch
import warp as wp

from orchard.interop import warp_metal as wm
from orchard.learn.so101_lift import LiftConfig, SO101LiftEnv

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


def test_lift_env_runs_and_resets():
    env = SO101LiftEnv(LiftConfig(num_envs=32, decimate_faces=2000, max_episode_steps=40))
    obs = env.reset()
    env.synchronize()
    assert obs["image"].shape == (32, 128, 128, 3) and obs["image"].device.type == "mps"
    assert obs["qpos"].shape == (32, 6)
    img0 = obs["image"].clone()
    # box positions are randomized inside the reset annulus
    box = env.sim.t.xpos[:, env.box_body].cpu().numpy()
    r = np.linalg.norm(box[:, :2], axis=1)
    assert np.all(r > 0.14) and np.all(r < 0.27) and np.all(np.abs(box[:, 2] - 0.03) < 0.01)
    assert (img0.float().std(dim=(1, 2, 3)) > 0).all()   # every tile has content
    c0 = wm.counters()
    rewards = []
    dones = []
    for k in range(45):
        a = torch.rand(32, env.act_dim, device="mps") * 2 - 1
        obs, rew, done, info = env.step(a)
        rewards.append(rew)
        dones.append(done)
    d = wm.counters() - c0
    env.synchronize()
    assert d.syncs == 0, "the step loop waited for the GPU"
    flags = env.sim.overflow_flags()
    assert not any(k in flags for k in ("NEFC", "NARROWPHASE", "CCD", "BROADPHASE")), f"buffer overflow: {flags}"
    R = torch.stack(rewards).cpu().numpy()
    D = torch.stack(dones).cpu().numpy()
    assert np.isfinite(R).all() and R.min() >= 0
    assert D[39].all() and not D[38].any()          # time limit at 40 steps resets every env
    assert (env.t.cpu().numpy() == 5).all()          # counters reset and advanced 5 more steps
    assert not torch.equal(obs["image"], img0)


def test_lift_env_throughput():
    n = 64
    env = SO101LiftEnv(LiftConfig(num_envs=n, decimate_faces=2000))
    env.reset()
    a = torch.zeros(n, env.act_dim, device="mps")
    for _ in range(5):
        env.step(a)
    env.synchronize()
    steps = 100
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(a)
    env.synchronize()
    dt = time.perf_counter() - t0
    print(f"SO-101 lift env (physics 4 substeps + render 128px + reward/reset on MPS), N={n}: "
          f"{n * steps / dt:,.0f} env-steps/s")
