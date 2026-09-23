"""Isaac Lab's Cartpole-RGB-Camera task on this stack, for the published-benchmark comparison.

Isaac-Cartpole-RGB-Camera-Direct-v0: 100x100 RGB tiled camera per env, 1024 envs (published:
50K / 45K / 32K steps/s for step / +inference / +train on an RTX 4090 with Isaac Sim 6.1), pole
angle/velocity reward, reset when the cart leaves ±3 m or the pole exceeds ±90 deg, episode 5 s,
decimation 2 at 120 Hz physics (control at 60 Hz). The scene here is an equivalent MJCF cartpole
(same dimensions class as Isaac's cartpole asset: 1 m pole, sliding cart on a rail), a camera
looking at the cart from the side as in Isaac's task, and the same reward terms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from orchard.interop import torch_bridge as tb
from orchard.interop import warp_metal as wm
from orchard.physics.batch import BatchSim, BatchSimOptions
from orchard.render.tier0 import Tier0Renderer

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.008333" integrator="implicitfast"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.5 0.7" rgb2="0.1 0.1 0.15" width="32" height="192"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.25 0.3" rgb2="0.35 0.35 0.4" width="64" height="64"/>
    <material name="grid" texture="grid" texrepeat="4 4" texuniform="true"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"/>
    <camera name="cam" pos="0 -4.5 1.6" xyaxes="1 0 0 0 0.2 0.98" fovy="45"/>
    <geom name="rail" type="capsule" fromto="-3.5 0 1 3.5 0 1" size="0.02" rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" range="-3 3" damping="0.02"/>
      <geom name="cart" type="box" size="0.2 0.1 0.1" rgba="0.2 0.4 0.9 1" mass="1.0" contype="0" conaffinity="0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.005"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 1.0" size="0.03" rgba="0.9 0.4 0.2 1" mass="0.1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="slide" joint="slider" gear="100" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


@dataclass
class CartpoleRGBConfig:
    num_envs: int = 1024
    width: int = 100
    height: int = 100
    decimation: int = 2
    episode_seconds: float = 5.0
    seed: int = 0
    render: bool = True


class CartpoleRGBEnv:
    """Isaac Lab's Cartpole-RGB task shape: obs {"image": (N,H,W,3) uint8, "state": (N,4)}, action (N,1)."""

    def __init__(self, cfg: CartpoleRGBConfig | None = None):
        self.cfg = cfg or CartpoleRGBConfig()
        self.n = self.cfg.num_envs
        self.model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        self.sim = BatchSim(self.model, self.n, options=BatchSimOptions(substeps=self.cfg.decimation, njmax=32))
        self.rend = Tier0Renderer(self.model, self.n, width=self.cfg.width, height=self.cfg.height, camera="cam",
                                  outputs=("rgb",)) if self.cfg.render else None
        self.dev = torch.device("mps")
        self.gen = torch.Generator(device="mps").manual_seed(self.cfg.seed)
        self.learner_event = wm.SharedEvent("metal:0", "orchard.cartpole")
        self.max_steps = int(self.cfg.episode_seconds / (self.model.opt.timestep * self.cfg.decimation))
        self.t = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
        self.act_dim = 1
        self.sim.synchronize()

    def _learner_done(self):
        v = self.learner_event.next_value()
        tb.signal_event(self.learner_event, v)
        self.sim.wait(self.learner_event, v)

    def _randomize(self, mask):
        q = self.sim.t.qpos; v = self.sim.t.qvel
        mf = mask.unsqueeze(1).float()
        u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.dev)
        q[:, 0:1] = mf * u(-1.0, 1.0, self.n, 1) + (1 - mf) * q[:, 0:1]
        q[:, 1:2] = mf * u(-0.25, 0.25, self.n, 1) + (1 - mf) * q[:, 1:2]
        v[:, 0:1] = mf * u(-0.5, 0.5, self.n, 1) + (1 - mf) * v[:, 0:1]
        v[:, 1:2] = mf * u(-0.5, 0.5, self.n, 1) + (1 - mf) * v[:, 1:2]
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)

    def _render(self, v):
        if self.rend is None:
            return
        vr = self.rend.render(self.sim, v)
        self.rend.wait_sim_after_render(self.sim, vr)
        self.rend.after(vr)

    def _obs(self):
        return {"image": self.rend.out.rgb if self.rend is not None else None,
                "state": torch.cat([self.sim.t.qpos[:, :2], self.sim.t.qvel[:, :2]], 1)}

    def reset(self):
        mask = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        v = self.sim.reset(mask); self.sim.after(v)
        self._randomize(mask); self._learner_done()
        v = self.sim.forward(); self._render(v); self.sim.after(v)
        return self._obs()

    def step(self, actions):
        self.sim.t.ctrl[:] = torch.clamp(actions, -1, 1)
        self._learner_done()
        vs = self.sim.step(); self.sim.after(vs)
        q = self.sim.t.qpos; qd = self.sim.t.qvel
        x, th, xd, thd = q[:, 0], q[:, 1], qd[:, 0], qd[:, 1]
        # Isaac Lab cartpole reward: alive 1.0, terminated -2.0, pole pos -1.0*th^2, cart vel -0.01|xd|, pole vel -0.005|thd|
        oob = (x.abs() > 3.0) | (th.abs() > np.pi / 2)
        self.t += 1
        trunc = self.t >= self.max_steps
        done = oob | trunc
        reward = 1.0 - 2.0 * oob.float() - 1.0 * th * th - 0.01 * xd.abs() - 0.005 * thd.abs()
        vres = self.sim.reset(done); self.sim.after(vres)
        self._randomize(done); self._learner_done()
        vf = self.sim.forward(); self._render(vf); self.sim.after(vf)
        return self._obs(), reward, done, {"truncated": trunc & ~oob}

    def synchronize(self):
        self.sim.synchronize()
        if self.rend is not None:
            self.rend.ctx.synchronize()
        torch.mps.synchronize()


def benchmark(n=1024, steps=200, render=True):
    """Isaac Lab's benchmark_non_rl methodology: env-steps/s of step (physics + render + reward/reset)."""
    import time
    env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=n, render=render))
    env.reset()
    a = torch.zeros(n, 1, device="mps")
    for _ in range(5):
        env.step(a)
    env.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(a)
    env.synchronize()
    return n * steps / (time.perf_counter() - t0)


if __name__ == "__main__":
    import sys, warp as wp
    wp.config.quiet = True
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    print(f"Cartpole (state only), N={n}: {benchmark(n, render=False):,.0f} env-steps/s")
    print(f"Cartpole-RGB 100x100, N={n}: {benchmark(n, render=True):,.0f} env-steps/s  (Isaac Lab, RTX 4090, published: 50,000 step-only)")
