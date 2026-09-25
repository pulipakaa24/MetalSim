"""PPO for pixel + proprioception observations on PyTorch MPS, tuned for launch count.

The learner is the wall-clock bottleneck on this platform (the plan's Section 3: ~90% of the
SB3 baseline). Design choices that follow from "launch count is the lever" on MPS (no graph
capture): observations stay on the GPU, rollout storage is preallocated, the update uses few
large minibatches, and per-step host work is a handful of small kernels. The rollout policy in
Warp (WS7) is a later step; this is the torch path with an instrumented time split.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _clip_grad_norm(grads, max_norm: float):
    total = torch.sqrt(sum((g * g).sum() for g in grads))
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(coef)
    return total


_clip_compiled = None


def clip_grad_norm_fast(params, max_norm):
    """nn.utils.clip_grad_norm_ (same formula) as one compiled graph: on MPS the stock version costs ~4 ms per call
    for this 2.7M-parameter net (a per-tensor norm, stack, and per-tensor scale; foreach is unsupported on MPS)."""
    global _clip_compiled
    grads = [p.grad for p in params if p.grad is not None]
    if grads[0].device.type != "mps":
        return nn.utils.clip_grad_norm_(params, max_norm)
    if _clip_compiled is None:
        _clip_compiled = torch.compile(_clip_grad_norm, dynamic=False)
    return _clip_compiled(grads, float(max_norm))


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
    # Isaac Lab / skrl camera-cartpole options
    desired_kl: float | None = None     # KLAdaptiveLR: lr /1.5 when KL > 2x, x1.5 when KL < 0.5x (bounds 1e-6..1e-2)
    clip_value: bool = False            # clip_predicted_values (value_clip = clip)
    feat: int = 256                     # CNN feature width (Isaac: 512)
    activation: str = "relu"            # trunk activation after the CNN (Isaac: elu)
    qpos_dim: int | None = None         # None: from env.obs_space["qpos"]; 0: image-only policy (Isaac's camera cartpole)
    center_images: bool = False         # subtract the per-image mean (Isaac's camera cartpole observation)
    value_norm: bool = False            # skrl RunningStandardScaler on value targets
    # MPS update fast path (same math): per-image means computed once per rollout step (flattened-view reduction),
    # uint8 -> centered float as one fused elementwise pass, minibatch loss under torch.compile, compiled grad-norm
    # clip (the stock one costs 4 ms per call on MPS), fused Adam.
    # Measured update (M4 Max, Isaac camera-cartpole config): 7.2 s plain -> 5.2 s fast_update -> 3.9 s with
    # metal_conv_kernels; see docs/research/metal_cnn_update_2026-09-24.md.
    fast_update: bool = True
    # with fast_update: custom Metal kernels (metalsim/learn/metal_conv.py) for conv1's weight gradient and conv2's
    # input gradient, the two ops MPSGraph runs at 1.6 / 2.5 TFLOP/s (Isaac camera-cartpole shapes only; else aten)
    metal_conv_kernels: bool = True


class NatureCNN(nn.Module):
    """Isaac Lab / skrl camera-cartpole feature extractor: conv 32x8s4, 64x4s2, 64x3s1 (ReLU), flatten, fc."""

    def __init__(self, in_ch=3, feat=256, image_hw=(128, 128), activation="relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, *image_hw)).shape[1]
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        self.fc = nn.Sequential(nn.Linear(n, feat), act)
        self.metal_kernels = False      # set by PPO (fast_update + metal_conv_kernels)

    def forward(self, x):
        return self.fc(self.conv(x))

    def forward_u8(self, x_u8, mean=None):
        """forward((x_u8/255) - mean[:, :, None, None]) from uint8 with a precomputed mean (None: no centering).

        The conversion, centering and scale are one elementwise expression (one fused kernel under torch.compile).
        Folding the centering into conv1 instead (conv(x, W/255) + b - m @ sum_hw W) is exact algebra but costs fp32
        precision through cancellation: conv1 weight-gradient error vs fp64 3e-5 instead of 4e-6 (CPU, measured)."""
        c1, c2 = self.conv[0], self.conv[2]
        if self.metal_kernels:
            from metalsim.learn import metal_conv as mc
            conv1, conv2 = mc.conv1, mc.conv2
        else:
            conv1 = conv2 = lambda x, w, b, s: F.conv2d(x, w, b, stride=s)
        if mean is not None:
            x = (x_u8.float() - mean[:, :, None, None] * 255.0) * (1.0 / 255.0)
        else:
            x = x_u8.float() * (1.0 / 255.0)
        y = conv2(F.relu(conv1(x, c1.weight, c1.bias, c1.stride[0])), c2.weight, c2.bias, c2.stride[0])
        return self.fc(self.conv[3:](y))


class ActorCritic(nn.Module):
    def __init__(self, act_dim, qpos_dim=6, feat=256, image_hw=(128, 128), activation="relu", center_images=False):
        super().__init__()
        self.cnn = NatureCNN(3, feat, image_hw, activation)
        self.qpos_dim = qpos_dim
        self.center_images = center_images
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        if qpos_dim > 0:
            self.qpos = nn.Sequential(nn.Linear(qpos_dim, 64), nn.ReLU())
            h = feat + 64
            self.pi = nn.Sequential(nn.Linear(h, 256), act, nn.Linear(256, act_dim))
            self.v = nn.Sequential(nn.Linear(h, 256), act, nn.Linear(256, 1))
        else:                       # image-only, shared trunk, linear heads (skrl "separate: False", layers [512])
            self.pi = nn.Linear(feat, act_dim)
            self.v = nn.Linear(feat, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    @staticmethod
    def image_mean(image_u8_nchw):
        """Per-image, per-channel mean of an (N,3,H,W) uint8 image in [0,1] units, over a flattened HW view."""
        n, c = image_u8_nchw.shape[:2]
        return image_u8_nchw.reshape(n, c, -1).float().mean(-1) * (1.0 / 255.0)

    def features_fast(self, image_u8_nchw, qpos, mean=None):
        """Fast-path features from contiguous (N,3,H,W) uint8; mean = image_mean(image) (computed here if None)."""
        if self.center_images and mean is None:
            mean = self.image_mean(image_u8_nchw)
        f = self.cnn.forward_u8(image_u8_nchw, mean if self.center_images else None)
        if self.qpos_dim > 0:
            f = torch.cat([f, self.qpos(qpos)], dim=1)
        return f

    def forward_fast(self, image_u8_nchw, qpos, mean=None):
        h = self.features_fast(image_u8_nchw, qpos, mean)
        return self.pi(h), self.v(h).squeeze(-1)

    def features(self, image_u8, qpos, nchw=False):
        """image_u8: (N,H,W,3) uint8 (renderer layout) or, with nchw=True, (N,3,H,W) uint8.

        The conv input must be contiguous NCHW: a permuted view makes MPS's conv backward ~5x
        slower (measured 145 vs 28 ms per 1024 samples), so the rollout buffer is stored NCHW.
        """
        x = image_u8 if nchw else image_u8.permute(0, 3, 1, 2).contiguous()
        x = x.float() * (1.0 / 255.0)
        if self.center_images:      # Isaac Lab's camera cartpole: subtract each image's mean (per channel)
            x = x - x.mean(dim=(2, 3), keepdim=True)
        f = self.cnn(x)
        if self.qpos_dim > 0:
            f = torch.cat([f, self.qpos(qpos)], dim=1)
        return f

    def forward(self, image_u8, qpos, nchw=False):
        h = self.features(image_u8, qpos, nchw)
        return self.pi(h), self.v(h).squeeze(-1)

    def dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp().clamp(math.exp(-20.0), math.exp(2.0)))


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
        H, W, C = env.obs_space["image"]
        qdim = self.cfg.qpos_dim if self.cfg.qpos_dim is not None else int(env.obs_space.get("qpos", (0,))[0])
        self.qdim = qdim
        self.net = ActorCritic(env.act_dim, qpos_dim=qdim, feat=self.cfg.feat, image_hw=(H, W), activation=self.cfg.activation,
                               center_images=self.cfg.center_images).to(self.dev)
        self._params = list(self.net.parameters())
        self.net.cnn.metal_kernels = bool(self.cfg.fast_update and self.cfg.metal_conv_kernels and self.dev.type == "mps")
        self.ret_mean, self.ret_var, self.ret_count = 0.0, 1.0, 1e-4      # running scaler for value targets
        fused = self.cfg.fast_update and self.dev.type in ("mps", "cuda")
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5, fused=fused)
        self.lr = self.cfg.lr
        n, T = env.n, self.cfg.rollout
        self.buf_img = torch.zeros((T, n, C, H, W), dtype=torch.uint8, device=self.dev)   # NCHW, contiguous
        self.buf_mean = torch.zeros((T, n, C), device=self.dev)       # per-image channel means (fast_update)
        self._mb_loss_c = torch.compile(self._mb_loss, dynamic=False) if (self.cfg.fast_update and self.dev.type == "mps") \
            else self._mb_loss
        self.buf_q = torch.zeros((T, n, max(qdim, 1)), device=self.dev)
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
                img = obs["image"]; q = obs["qpos"] if self.qdim > 0 else self.buf_q[t]
                self.buf_img[t].copy_(img.permute(0, 3, 1, 2))
                if self.qdim > 0:
                    self.buf_q[t].copy_(q)
                if cfg.fast_update:
                    self.buf_mean[t] = self.net.image_mean(self.buf_img[t])
                    mean, v = self.net.forward_fast(self.buf_img[t], q, self.buf_mean[t])
                else:
                    mean, v = self.net(img, q)
                d = self.net.dist(mean)
                a = d.sample()
                self.buf_a[t] = a
                self.buf_logp[t] = d.log_prob(a).sum(-1)
                self.buf_v[t] = v
                if cfg.value_norm:
                    v = v * math.sqrt(self.ret_var) + self.ret_mean
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
            _, last_v = self.net(obs["image"], obs["qpos"] if self.qdim > 0 else self.buf_q[0])
            if cfg.value_norm:
                last_v = last_v * math.sqrt(self.ret_var) + self.ret_mean
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

    def _mb_loss(self, img, q, img_mean, a, logp_old, adv, ret, v_old):
        """PPO minibatch loss (fast path). Returns loss, pg, vf, logp."""
        cfg = self.cfg
        mean, v = self.net.forward_fast(img, q, img_mean)
        d = self.net.dist(mean)
        logp = d.log_prob(a).sum(-1)
        ratio = (logp - logp_old).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        if cfg.clip_value:
            v_c = v_old + (v - v_old).clamp(-cfg.clip, cfg.clip)
            vf = torch.max((v - ret) ** 2, (v_c - ret) ** 2).mean()
        else:
            vf = F.mse_loss(v, ret)
        loss = pg + cfg.vf_coef * vf
        if cfg.ent_coef != 0.0:
            loss = loss - cfg.ent_coef * d.entropy().sum(-1).mean()
        return loss, pg, vf, logp

    def update(self, adv, ret):
        cfg = self.cfg
        T, n = cfg.rollout, self.env.n
        N = T * n
        img = self.buf_img.reshape(N, *self.buf_img.shape[2:])
        img_mean = self.buf_mean.reshape(N, -1)
        q = self.buf_q.reshape(N, -1); a = self.buf_a.reshape(N, -1)
        logp_old = self.buf_logp.reshape(N); adv = adv.reshape(N); ret = ret.reshape(N); v_buf = self.buf_v.reshape(N).clone()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if cfg.value_norm:       # update the running scaler with this batch of returns, then train on normalized targets
            bm, bv, bc = float(ret.mean()), float(ret.var()), float(N)
            delta = bm - self.ret_mean; tot = self.ret_count + bc
            self.ret_var = (self.ret_var * self.ret_count + bv * bc + delta ** 2 * self.ret_count * bc / tot) / tot
            self.ret_mean += delta * bc / tot; self.ret_count = tot
            sd = math.sqrt(self.ret_var) + 1e-8
            ret = (ret - self.ret_mean) / sd; v_buf = (v_buf - self.ret_mean) / sd
        mb = N // cfg.minibatches
        stats = {"pg": 0.0, "vf": 0.0, "kl": 0.0}
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=self.dev)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                if cfg.fast_update:
                    loss, pg, vf, logp = self._mb_loss_c(img[idx], q[idx], img_mean[idx], a[idx], logp_old[idx],
                                                         adv[idx], ret[idx], v_buf[idx])
                else:
                    mean, v = self.net(img[idx], q[idx], nchw=True)
                    d = self.net.dist(mean)
                    logp = d.log_prob(a[idx]).sum(-1)
                    ratio = (logp - logp_old[idx]).exp()
                    pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
                    if cfg.clip_value:
                        v_old = v_buf[idx]; v_c = v_old + (v - v_old).clamp(-cfg.clip, cfg.clip)
                        vf = torch.max((v - ret[idx]) ** 2, (v_c - ret[idx]) ** 2).mean()
                    else:
                        vf = F.mse_loss(v, ret[idx])
                    ent = d.entropy().sum(-1).mean()
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                if cfg.desired_kl is not None:
                    with torch.no_grad():
                        kl_now = (logp_old[idx] - logp).mean().item()
                    if kl_now > 2.0 * cfg.desired_kl:
                        self.lr = max(1e-6, self.lr / 1.5)
                    elif 0.0 < kl_now < 0.5 * cfg.desired_kl:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.fast_update:
                    clip_grad_norm_fast(self._params, cfg.max_grad_norm)
                else:
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
                       f"pg {stats['pg']:.3f} vf {stats['vf']:.3f} kl {stats['kl']:.4f} lr {self.lr:.1e} | {self.timers.report()}")
                log(msg)
        return self
