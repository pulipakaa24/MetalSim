"""Lidar-based RL: goal navigation among random obstacles from a planar lidar scan.

Isaac Lab publishes no lidar RL benchmark, so this is a MetalSim task, not a parity row: a wheeled
base (velocity-controlled planar slides) must reach a random goal 4-7 m away among 12 randomly placed
boxes, observing only a 64-beam planar lidar (Metal ray queries against the batched acceleration
structure that the physics state refits every step), the goal vector and its own velocity.

Observation (68): 64 ranges / max_range, goal delta (2) / 8, velocity (2) / 2.
Action (2): commanded planar velocity in [-1, 1] x 1.5 m/s.
Reward: 10 x progress towards the goal per step, +10 at the goal (< 0.4 m), -1 per step in contact
with an obstacle, -0.01 |a|^2. Episode 15 s at 20 Hz control (5 physics substeps at 100 Hz).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.sensors.raytrace import RayTracer

N_OBST = 12
MAX_RANGE = 8.0
ARENA = 6.0          # half-size of the square arena the obstacles and goals are drawn from


def build_xml(n_obst=N_OBST):
    obst = "\n".join(
        f'    <body name="obst{k}" pos="{20 + k} 0 0.3"><freejoint/>'
        f'<geom name="obst{k}" type="box" size="0.25 0.25 0.3" mass="200" rgba="0.75 0.35 0.2 1"/></body>' for k in range(n_obst))
    return f"""
<mujoco model="lidar_nav">
  <option timestep="0.01" integrator="implicitfast"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.25 0.3" rgb2="0.35 0.35 0.4" width="64" height="64"/>
    <material name="grid" texture="grid" texrepeat="12 12" texuniform="true"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 6" dir="0.2 0.3 -0.9" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="{ARENA + 2} {ARENA + 2} 0.1" material="grid" group="4"/>
    <camera name="cam" pos="0 -11 7" xyaxes="1 0 0 0 0.55 0.83" fovy="50"/>
    <body name="robot" pos="0 0 0.16">
      <joint name="x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="y" type="slide" axis="0 1 0" damping="2"/>
      <geom name="robot" type="cylinder" size="0.25 0.15" mass="5" rgba="0.2 0.5 0.9 1" group="4"/>
      <site name="lidar" pos="0 0 0.1" size="0.02"/>
      <site name="touch" type="cylinder" size="0.27 0.16" rgba="0 0 0 0"/>
    </body>
{obst}
  </worldbody>
  <actuator>
    <velocity name="vx" joint="x" kv="30" ctrlrange="-1.5 1.5"/>
    <velocity name="vy" joint="y" kv="30" ctrlrange="-1.5 1.5"/>
  </actuator>
  <sensor><touch name="contact" site="touch"/></sensor>
</mujoco>
"""


@dataclass
class LidarNavConfig:
    num_envs: int = 1024
    n_beams: int = 64
    control_hz: float = 20.0
    episode_seconds: float = 15.0
    seed: int = 0


class LidarNavEnv:
    """Torch-driven env (physics on the Warp queue, lidar on the Metal RT queue, event-ordered)."""

    def __init__(self, cfg: LidarNavConfig | None = None):
        self.cfg = cfg or LidarNavConfig()
        self.n = self.cfg.num_envs
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        m = self.model
        self.decimation = int(round(1.0 / (self.cfg.control_hz * m.opt.timestep)))
        self.sim = BatchSim(m, self.n, options=BatchSimOptions(substeps=self.decimation, njmax=512, nconmax=96,
                                                                 solver_iterations=10, ls_iterations=10))
        # lidar: 64 beams in the horizontal plane of the robot; the robot's own geom and the floor are
        # in group 4 so the scan sees only obstacles (max_group=3)
        self.rt = RayTracer(m, self.n, max_range=MAX_RANGE, max_group=3, include_planes=False)
        az = np.linspace(-np.pi, np.pi, self.cfg.n_beams, endpoint=False)
        self.lidar = self.rt.make_lidar("lidar", az, np.zeros_like(az))
        self.dev = torch.device("mps")
        self.gen = torch.Generator(device="mps").manual_seed(self.cfg.seed)
        self.learner_event = wm.SharedEvent("metal:0", "metalsim.lidar_nav")
        self.max_steps = int(self.cfg.episode_seconds * self.cfg.control_hz)
        self.t = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
        self.goal = torch.zeros(self.n, 2, device=self.dev)
        self.prev_dist = torch.zeros(self.n, device=self.dev)
        self.ep_ret = torch.zeros(self.n, device=self.dev)
        self.act_dim = 2
        self.obs_dim = self.cfg.n_beams + 4
        self.touch_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "contact")])
        self.sim.synchronize()

    # -- ordering helpers ----------------------------------------------------------------------------
    def _learner_done(self):
        v = self.learner_event.next_value(); tb.signal_event(self.learner_event, v); self.sim.wait(self.learner_event, v)

    def _scan(self, after):
        v = self.rt.trace(self.sim, [self.lidar], after_value=after)
        self.sim.wait(self.rt.event, v)          # the next physics step must not move geoms under the rays
        tb.wait_event(self.rt.event, v)          # torch reads the ranges after the trace

    # -- episodes ------------------------------------------------------------------------------------
    def _randomize(self, mask):
        """Robot at a random point, goal 4-7 m away, obstacles at random positions clear of both."""
        n = self.n; u = lambda *s: torch.rand(*s, generator=self.gen, device=self.dev)
        q = self.sim.t.qpos; qd = self.sim.t.qvel
        mf = mask.float().unsqueeze(1)
        start = (u(n, 2) * 2 - 1) * (ARENA - 1.0)
        ang = u(n) * 2 * math.pi; r = 4.0 + 3.0 * u(n)
        goal = start + torch.stack([r * torch.cos(ang), r * torch.sin(ang)], 1)
        goal = goal.clamp(-ARENA + 0.5, ARENA - 0.5)
        q[:, 0:2] = mf * start + (1 - mf) * q[:, 0:2]
        qd[:, 0:2] = (1 - mf) * qd[:, 0:2]
        self.goal = torch.where(mask.unsqueeze(1), goal, self.goal)
        for k in range(N_OBST):
            base = 2 + 7 * k
            p = (u(n, 2) * 2 - 1) * ARENA
            # keep 1 m clear of the start and the goal
            for c in (start, goal):
                d = p - c; dist = d.norm(dim=1, keepdim=True).clamp_min(1e-3)
                p = torch.where(dist < 1.2, c + d / dist * 1.2, p)
            q[:, base:base + 2] = mf * p + (1 - mf) * q[:, base:base + 2]
            q[:, base + 2] = mf[:, 0] * 0.3 + (1 - mf[:, 0]) * q[:, base + 2]
            q[:, base + 3:base + 7] = mf * torch.tensor([1.0, 0, 0, 0], device=self.dev) + (1 - mf) * q[:, base + 3:base + 7]
            qd[:, 2 + 6 * k:8 + 6 * k] = (1 - mf) * qd[:, 2 + 6 * k:8 + 6 * k]
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.ep_ret = torch.where(mask, torch.zeros_like(self.ep_ret), self.ep_ret)
        self.prev_dist = torch.where(mask, (self.goal - q[:, 0:2]).norm(dim=1), self.prev_dist)

    def _obs(self):
        rng = self.lidar.out["range"]
        ranges = torch.where(rng > 0, rng, torch.full_like(rng, MAX_RANGE)) / MAX_RANGE
        pos = self.sim.t.qpos[:, 0:2]; vel = self.sim.t.qvel[:, 0:2]
        return torch.cat([ranges, (self.goal - pos) / 8.0, vel / 2.0], 1)

    def reset(self):
        mask = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        v = self.sim.reset(mask); self.sim.after(v)
        self._randomize(mask); self._learner_done()
        v = self.sim.forward(); self._scan(v); self.sim.after(v)
        return self._obs()

    def step(self, actions):
        self.sim.t.ctrl[:] = actions.clamp(-1, 1) * 1.5
        self._learner_done()
        vs = self.sim.step(); self._scan(vs); self.sim.after(vs)
        pos = self.sim.t.qpos[:, 0:2]
        dist = (self.goal - pos).norm(dim=1)
        contact = self.sim.t.sensordata[:, self.touch_adr] > 1.0
        reached = dist < 0.4
        self.t += 1
        trunc = self.t >= self.max_steps
        done = reached | trunc
        reward = 10.0 * (self.prev_dist - dist) + 10.0 * reached.float() - 1.0 * contact.float() - 0.01 * (actions ** 2).sum(1)
        self.prev_dist = dist
        self.ep_ret += reward
        info = {"episode_reward": self.ep_ret.clone(), "episode_length": self.t.float(), "is_success": reached, "contact": contact}
        vres = self.sim.reset(done); self.sim.after(vres)
        self._randomize(done); self._learner_done()
        vf = self.sim.forward(); self._scan(vf); self.sim.after(vf)
        return self._obs(), reward, done, info

    def synchronize(self):
        self.sim.synchronize(); self.rt.ctx.synchronize(); torch.mps.synchronize()


# --------------------------------------------------------------------------------------------------
# PPO on vector observations (rsl_rl configuration), torch on MPS

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 128, 128)):
        super().__init__()
        def mlp(out):
            layers, d = [], obs_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ELU()]; d = h
            layers.append(nn.Linear(d, out)); return nn.Sequential(*layers)
        self.actor, self.critic = mlp(act_dim), mlp(1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))


@dataclass
class VecPPOConfig:
    iterations: int = 300
    rollout: int = 24
    epochs: int = 5
    minibatches: int = 4
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 10


def train(env: LidarNavEnv, cfg: VecPPOConfig | None = None, log=print, checkpoint=None):
    cfg = cfg or VecPPOConfig(); dev = env.dev; n, T = env.n, cfg.rollout
    net = MLPActorCritic(env.obs_dim, env.act_dim).to(dev); opt = torch.optim.Adam(net.parameters(), lr=cfg.lr); lr = cfg.lr
    obs_b = torch.zeros(T, n, env.obs_dim, device=dev); act_b = torch.zeros(T, n, env.act_dim, device=dev)
    logp_b = torch.zeros(T, n, device=dev); val_b = torch.zeros(T, n, device=dev); rew_b = torch.zeros(T, n, device=dev); done_b = torch.zeros(T, n, device=dev)
    obs = env.reset(); t0 = time.perf_counter(); steps = 0; hist = []
    for it in range(1, cfg.iterations + 1):
        ep_ret, ep_len, ep_succ, ep_contact = [], [], [], []
        for t in range(T):
            with torch.no_grad():
                mean = net.actor(obs); std = net.log_std.exp(); d = torch.distributions.Normal(mean, std)
                a = d.sample(); obs_b[t] = obs; act_b[t] = a; logp_b[t] = d.log_prob(a).sum(-1); val_b[t] = net.critic(obs).squeeze(-1)
            obs, r, done, info = env.step(a)
            rew_b[t] = r; done_b[t] = done.float()
            ep_ret.append(torch.where(done, info["episode_reward"], torch.zeros_like(r))); ep_len.append(torch.where(done, info["episode_length"], torch.zeros_like(r)))
            ep_succ.append(done & info["is_success"]); ep_contact.append(info["contact"])
        steps += T * n
        with torch.no_grad():
            last_v = net.critic(obs).squeeze(-1)
        adv = torch.zeros_like(rew_b); gae = torch.zeros(n, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]; nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + cfg.gamma * nv * nonterm - val_b[t]; gae = delta + cfg.gamma * cfg.lam * nonterm * gae; adv[t] = gae
        ret = adv + val_b; N = T * n
        o = obs_b.reshape(N, -1); a_ = act_b.reshape(N, -1); lp0 = logp_b.reshape(N); advn = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(N); retf = ret.reshape(N); v_old = val_b.reshape(N)
        stats = {"pg": 0.0, "vf": 0.0, "kl": 0.0}; mb = N // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                mean = net.actor(o[idx]); std = net.log_std.exp(); d = torch.distributions.Normal(mean, std)
                lp = d.log_prob(a_[idx]).sum(-1); ratio = (lp - lp0[idx]).exp()
                pg = -torch.min(ratio * advn[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advn[idx]).mean()
                v = net.critic(o[idx]).squeeze(-1); v_c = v_old[idx] + (v - v_old[idx]).clamp(-cfg.clip, cfg.clip)
                vf = torch.max((v - retf[idx]) ** 2, (v_c - retf[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean(); loss = pg + vf - cfg.ent_coef * ent
                with torch.no_grad():
                    kl = (lp0[idx] - lp).mean().item()
                if kl > 2 * cfg.desired_kl: lr = max(1e-5, lr / 1.5)
                elif 0 < kl < 0.5 * cfg.desired_kl: lr = min(1e-2, lr * 1.5)
                for g in opt.param_groups: g["lr"] = lr
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm); opt.step()
                stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["kl"] += kl
        k = cfg.epochs * cfg.minibatches
        if it % cfg.log_every == 0 or it == cfg.iterations:
            env.synchronize()
            rets = torch.stack(ep_ret); lens = torch.stack(ep_len); succ = torch.stack(ep_succ); dones = done_b > 0
            cnt = int(dones.sum().item())
            mret = float(rets[dones].mean().item()) if cnt else 0.0; mlen = float(lens[dones].mean().item()) if cnt else 0.0
            msucc = float(succ[dones].float().mean().item()) if cnt else 0.0; mcontact = float(torch.stack(ep_contact).float().mean().item())
            sps = steps / (time.perf_counter() - t0); hist.append((it, steps, mret, mlen, msucc))
            log(f"it {it:4d} steps {steps:9d} sps {sps:8,.0f} | ep_ret {mret:7.2f} ep_len {mlen:6.1f} success {msucc:5.2f} (n={cnt}) contact/step {mcontact:.3f} | "
                f"pg {stats['pg'] / k:.3f} vf {stats['vf'] / k:.3f} kl {stats['kl'] / k:.4f} lr {lr:.1e}")
    if checkpoint:
        torch.save({"net": net.state_dict(), "obs_dim": env.obs_dim, "act_dim": env.act_dim}, checkpoint)
    return net, hist


if __name__ == "__main__":
    import sys, warp as wp
    wp.config.quiet = True
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    log_path = sys.argv[3] if len(sys.argv) > 3 else None
    f = open(log_path, "a") if log_path else None
    def log(m):
        print(m, flush=True)
        if f: f.write(m + "\n"); f.flush()
    env = LidarNavEnv(LidarNavConfig(num_envs=n))
    log(f"LidarNav: {n} envs, {env.cfg.n_beams}-beam lidar, obs {env.obs_dim}, control {env.cfg.control_hz} Hz, decimation {env.decimation}")
    train(env, VecPPOConfig(iterations=iters), log=log, checkpoint=sys.argv[4] if len(sys.argv) > 4 else None)
