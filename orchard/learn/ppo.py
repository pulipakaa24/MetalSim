"""PPO for pixel + proprioception observations on PyTorch MPS, tuned for launch count.

The learner is the wall-clock bottleneck on this platform (the plan's Section 3: ~90% of the
SB3 baseline). Design choices that follow from "launch count is the lever" on MPS (no graph
capture): observations stay on the GPU, rollout storage is preallocated, the update uses few
large minibatches, and per-step host work is a handful of small kernels. The rollout policy in
Warp (WS7) is a later step; this is the torch path with an instrumented time split.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    rollout: int = 64            # steps per env per update
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_every: int = 1
    seed: int = 0


class NatureCNN(nn.Module):
    def __init__(self, in_ch=3, feat=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, 128, 128)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, feat), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))


class ActorCritic(nn.Module):
    def __init__(self, act_dim, qpos_dim=6, feat=256):
        super().__init__()
        self.cnn = NatureCNN(3, feat)
        self.qpos = nn.Sequential(nn.Linear(qpos_dim, 64), nn.ReLU())
        self.pi = nn.Sequential(nn.Linear(feat + 64, 256), nn.ReLU(), nn.Linear(256, act_dim))
        self.v = nn.Sequential(nn.Linear(feat + 64, 256), nn.ReLU(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def features(self, image_u8, qpos):
        x = image_u8.permute(0, 3, 1, 2).float() * (1.0 / 255.0)
        return torch.cat([self.cnn(x), self.qpos(qpos)], dim=1)

    def forward(self, image_u8, qpos):
        h = self.features(image_u8, qpos)
        return self.pi(h), self.v(h).squeeze(-1)

    def dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp())


@dataclass
class Timers:
    env: float = 0.0
    policy: float = 0.0
    update: float = 0.0
    n_updates: int = 0

    def report(self):
        tot = self.env + self.policy + self.update + 1e-9
        return (f"env {self.env / tot:5.1%}  policy {self.policy / tot:5.1%}  update {self.update / tot:5.1%}")


class PPO:
    def __init__(self, env, cfg: PPOConfig | None = None, device="mps"):
        self.env, self.cfg, self.dev = env, cfg or PPOConfig(), torch.device(device)
        torch.manual_seed(self.cfg.seed)
        self.net = ActorCritic(env.act_dim).to(self.dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        n, T = env.n, self.cfg.rollout
        H, W, C = env.obs_space["image"]
        self.buf_img = torch.zeros((T, n, H, W, C), dtype=torch.uint8, device=self.dev)
        self.buf_q = torch.zeros((T, n, 6), device=self.dev)
        self.buf_a = torch.zeros((T, n, env.act_dim), device=self.dev)
        self.buf_logp = torch.zeros((T, n), device=self.dev)
        self.buf_v = torch.zeros((T, n), device=self.dev)
        self.buf_r = torch.zeros((T, n), device=self.dev)
        self.buf_done = torch.zeros((T, n), device=self.dev)
        self.timers = Timers()
        self.global_step = 0
        self.episodes = []   # (return, length, success) finished during training

    def collect(self, obs):
        cfg, n = self.cfg, self.env.n
        ep_ret, ep_len, ep_succ = [], [], []
        for t in range(cfg.rollout):
            t0 = time.perf_counter()
            with torch.no_grad():
                img, q = obs["image"], obs["qpos"]
                self.buf_img[t].copy_(img)
                self.buf_q[t].copy_(q)
                mean, v = self.net(img, q)
                d = self.net.dist(mean)
                a = d.sample()
                self.buf_a[t] = a
                self.buf_logp[t] = d.log_prob(a).sum(-1)
                self.buf_v[t] = v
            t1 = time.perf_counter()
            obs, r, done, info = self.env.step(a)
            self.buf_r[t] = r
            self.buf_done[t] = done.float()
            # episode statistics stay on the GPU; gathered once per rollout
            ep_ret.append(torch.where(done, info["episode_reward"], torch.zeros_like(r)))
            ep_len.append(torch.where(done, info["episode_length"], torch.zeros_like(info["episode_length"])))
            ep_succ.append(done & info["is_success"])
            self.timers.policy += t1 - t0
            self.timers.env += time.perf_counter() - t1
            self.global_step += n
        with torch.no_grad():
            _, last_v = self.net(obs["image"], obs["qpos"])
        self.env.synchronize()
        done_mask = torch.stack([torch.as_tensor(d > 0) for d in self.buf_done]).cpu().numpy()
        rets = torch.stack(ep_ret).cpu().numpy(); lens = torch.stack(ep_len).cpu().numpy()
        succ = torch.stack(ep_succ).cpu().numpy()
        for (t, e) in zip(*np.nonzero(done_mask)):
            self.episodes.append((float(rets[t, e]), int(lens[t, e]), bool(succ[t, e])))
        return obs, last_v

    def gae(self, last_v):
        cfg = self.cfg
        T = cfg.rollout
        adv = torch.zeros_like(self.buf_r)
        gae = torch.zeros(self.env.n, device=self.dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else self.buf_v[t + 1]
            nonterm = 1.0 - self.buf_done[t]
            delta = self.buf_r[t] + cfg.gamma * nv * nonterm - self.buf_v[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        return adv, adv + self.buf_v

    def update(self, adv, ret):
        cfg = self.cfg
        T, n = cfg.rollout, self.env.n
        N = T * n
        img = self.buf_img.reshape(N, *self.buf_img.shape[2:])
        q = self.buf_q.reshape(N, -1); a = self.buf_a.reshape(N, -1)
        logp_old = self.buf_logp.reshape(N); adv = adv.reshape(N); ret = ret.reshape(N)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = N // cfg.minibatches
        stats = {"pg": 0.0, "vf": 0.0, "kl": 0.0}
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=self.dev)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                mean, v = self.net(img[idx], q[idx])
                d = self.net.dist(mean)
                logp = d.log_prob(a[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
                vf = F.mse_loss(v, ret[idx])
                ent = d.entropy().sum(-1).mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats["pg"] += pg.detach(); stats["vf"] += vf.detach()
                stats["kl"] += (logp_old[idx] - logp).mean().detach()
        k = cfg.epochs * cfg.minibatches
        return {key: float(val) / k for key, val in stats.items()}

    def train(self, log=print):
        cfg = self.cfg
        obs = self.env.reset()
        t_start = time.perf_counter()
        it = 0
        while self.global_step < cfg.total_steps:
            obs, last_v = self.collect(obs)
            t0 = time.perf_counter()
            adv, ret = self.gae(last_v)
            stats = self.update(adv, ret)
            torch.mps.synchronize()
            self.timers.update += time.perf_counter() - t0
            self.timers.n_updates += 1
            it += 1
            if it % cfg.log_every == 0:
                recent = self.episodes[-100:]
                sps = self.global_step / (time.perf_counter() - t_start)
                msg = (f"it {it:4d} steps {self.global_step:9d} sps {sps:7,.0f} | "
                       f"ret {np.mean([e[0] for e in recent]) if recent else 0:7.2f} "
                       f"len {np.mean([e[1] for e in recent]) if recent else 0:5.1f} "
                       f"succ {np.mean([e[2] for e in recent]) if recent else 0:5.2f} (n={len(recent)}) | "
                       f"pg {stats['pg']:.3f} vf {stats['vf']:.3f} kl {stats['kl']:.4f} | {self.timers.report()}")
                log(msg)
        return self
