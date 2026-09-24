"""PPO with the rollout in Warp (Isaac Lab pattern): the whole rollout of T steps is T replays of
one captured graph (observation gather, Warp MLP policy, buffer store, physics substeps, reward and
termination kernels); PyTorch touches the GPU once per update (GAE + minibatch epochs on the
zero-copy rollout buffers). One host synchronization per rollout.

Tasks plug in through ``WarpTask``: kernels for observations, reward/termination and reset.
``CartpoleTask`` reproduces Isaac Lab's Cartpole-Direct-v0 (reward terms, termination bounds,
reset distribution) so its rsl_rl PPO config (16 steps/env, 150 iterations, MLP 32x32) can be
run for the RL reproducibility check.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.learn.warp_policy import ActorCriticMLP, RolloutBuffers, WarpMLPPolicy
from metalsim.physics.batch import BatchSim, BatchSimOptions


# --------------------------------------------------------------------------------------------------
# Cartpole task kernels (Isaac Lab Cartpole-Direct-v0 semantics)

@wp.kernel
def cartpole_obs(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), obs: wp.array2d(dtype=float)):
    e = wp.tid()
    obs[e, 0] = qpos[e, 0]; obs[e, 1] = qpos[e, 1]; obs[e, 2] = qvel[e, 0]; obs[e, 3] = qvel[e, 1]


@wp.kernel
def cartpole_reward_done(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), t: wp.array(dtype=int),
                         max_t: int, step_idx: wp.array(dtype=int),
                         buf_rew: wp.array2d(dtype=float), buf_done: wp.array2d(dtype=float),
                         reset_mask: wp.array(dtype=wp.bool), ep_ret: wp.array(dtype=float), ep_len: wp.array(dtype=int),
                         done_ret_sum: wp.array(dtype=float), done_len_sum: wp.array(dtype=int), done_count: wp.array(dtype=int)):
    e = wp.tid()
    x = qpos[e, 0]; th = qpos[e, 1]; xd = qvel[e, 0]; thd = qvel[e, 1]
    oob = wp.abs(x) > 3.0 or wp.abs(th) > 1.5707964
    t[e] = t[e] + 1
    trunc = t[e] >= max_t
    done = oob or trunc
    r = 1.0 - 1.0 * th * th - 0.01 * wp.abs(xd) - 0.005 * wp.abs(thd)
    if oob:
        r = r - 2.0
    s = step_idx[0] - 1   # store() already advanced the index for this step
    buf_rew[s, e] = r
    buf_done[s, e] = 1.0 if done else 0.0
    ep_ret[e] = ep_ret[e] + r
    ep_len[e] = ep_len[e] + 1
    reset_mask[e] = done
    if done:
        wp.atomic_add(done_ret_sum, 0, ep_ret[e])
        wp.atomic_add(done_len_sum, 0, ep_len[e])
        wp.atomic_add(done_count, 0, 1)
        ep_ret[e] = 0.0
        ep_len[e] = 0
        t[e] = 0


@wp.kernel
def cartpole_reset(reset_mask: wp.array(dtype=wp.bool), seed: int, step_idx: wp.array(dtype=int),
                   qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float)):
    """Reset distribution of Isaac's cartpole: pole angle in ±0.25 rad, cart in ±1 m, small velocities."""
    e = wp.tid()
    if not reset_mask[e]:
        return
    rng = wp.rand_init(seed + 7, step_idx[0] * 7919 + e)
    qpos[e, 0] = wp.randf(rng, -1.0, 1.0)
    qpos[e, 1] = wp.randf(rng, -0.25, 0.25)
    qvel[e, 0] = wp.randf(rng, -0.5, 0.5)
    qvel[e, 1] = wp.randf(rng, -0.5, 0.5)


class CartpoleTask:
    obs_dim, act_dim = 4, 1
    ctrl_lo, ctrl_hi = [-1.0], [1.0]

    def __init__(self, n, seed=0, episode_seconds=5.0, decimation=2):
        from metalsim.learn.cartpole_rgb import CARTPOLE_XML
        self.model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        self.sim = BatchSim(self.model, n, options=BatchSimOptions(substeps=decimation, njmax=32, solver_iterations=10, ls_iterations=10))
        self.n, self.seed = n, seed
        self.max_t = int(episode_seconds / (self.model.opt.timestep * decimation))
        dev = "metal:0"
        self.t = wp.zeros(n, dtype=int, device=dev)
        self.ep_ret = wp.zeros(n, dtype=float, device=dev)
        self.ep_len = wp.zeros(n, dtype=int, device=dev)
        self.stats = (wp.zeros(1, dtype=float, device=dev), wp.zeros(1, dtype=int, device=dev), wp.zeros(1, dtype=int, device=dev))
        self.obs = wp.zeros((n, 4), dtype=float, device=dev)

    def launch_obs(self):
        wp.launch(cartpole_obs, dim=self.n, inputs=[self.sim.d.qpos, self.sim.d.qvel, self.obs], device="metal:0")

    def launch_reward_done_reset(self, pol, bufs):
        d = self.sim.d
        wp.launch(cartpole_reward_done, dim=self.n,
                  inputs=[d.qpos, d.qvel, self.t, self.max_t, pol.step_idx, bufs.rew, bufs.done, self.sim._reset_mask,
                          self.ep_ret, self.ep_len, *self.stats], device="metal:0")
        # MuJoCo Warp reset of the flagged worlds, then the task's state randomization, then kinematics
        import mujoco_warp as mjw
        mjw.reset_data(self.sim.m, d, reset=self.sim._reset_mask)
        wp.launch(cartpole_reset, dim=self.n, inputs=[self.sim._reset_mask, self.seed, pol.step_idx, d.qpos, d.qvel], device="metal:0")
        mjw.forward(self.sim.m, d)

    def episode_stats(self):
        """(mean return, mean length, count) of episodes finished since the last call; synchronizes."""
        s, l, c = (a.numpy()[0] for a in self.stats)
        for a in self.stats:
            a.zero_()
        return (s / c if c else 0.0, l / c if c else 0.0, int(c))


# --------------------------------------------------------------------------------------------------

@dataclass
class PPOWarpConfig:
    iterations: int = 150
    rollout: int = 16
    epochs: int = 5
    minibatches: int = 4
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 1.0
    clip_value: bool = True             # rsl_rl use_clipped_value_loss: max of clipped/unclipped value error
    max_grad_norm: float = 1.0
    hidden: tuple = (32, 32)
    desired_kl: float | None = 0.01     # rsl_rl adaptive schedule: lr x1.5 when KL < desired/2, /1.5 when > 2x desired
    seed: int = 0
    log_every: int = 10


class PPOWarp:
    def __init__(self, task, cfg: PPOWarpConfig | None = None):
        self.task, self.cfg = task, cfg or PPOWarpConfig()
        torch.manual_seed(self.cfg.seed)
        n, T = task.n, self.cfg.rollout
        self.net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=self.cfg.hidden,
                                 actor_in=getattr(task, "actor_dim", None)).to("mps")
        self.pol = WarpMLPPolicy(self.net, n, task.obs_dim, task.act_dim, task.ctrl_lo, task.ctrl_hi, seed=self.cfg.seed)
        self.bufs = RolloutBuffers(T, n, task.obs_dim, task.act_dim)
        dev = "metal:0"
        self.bufs.rew = wp.zeros((T, n), dtype=float, device=dev)
        self.bufs.done = wp.zeros((T, n), dtype=float, device=dev)
        wp.synchronize_device(dev)
        self.bufs.t_rew, self.bufs.t_done = tb.mps_tensor(self.bufs.rew), tb.mps_tensor(self.bufs.done)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.lr = self.cfg.lr
        self.graph = None
        self.learner_event = wm.SharedEvent(dev, "metalsim.ppo_warp")

    def _obs(self):
        # tasks whose observation kernels draw noise / commands keyed by the rollout step index
        if getattr(self.task, "needs_step_idx", False):
            self.task.launch_obs(self.pol.step_idx)
        else:
            self.task.launch_obs()

    def _act(self):
        task, pol = self.task, self.pol
        if hasattr(task, "launch_apply_action"):
            # the raw Gaussian sample is the action (stored for the update); the task maps it to ctrl
            pol.act(task.obs, task.action_scratch)
            task.launch_apply_action(pol.action)
        else:
            pol.act(task.obs, task.sim.d.ctrl)

    def _capture(self):
        task, pol, bufs = self.task, self.pol, self.bufs
        with wp.ScopedCapture(device="metal:0") as cap:
            self._obs()
            self._act()
            pol.store(task.obs, bufs)
            if hasattr(task, "launch_physics"):      # tasks that interleave their own work between substeps
                task.launch_physics(pol.step_idx)
            else:
                task.sim.launch_step()
            task.launch_reward_done_reset(pol, bufs)
        self.graph = cap.graph

    def rollout(self):
        if self.graph is None:
            self._capture()
        self.pol.rewind()
        for _ in range(self.cfg.rollout):
            wp.capture_launch(self.graph)
        # bootstrap value of the final state on the Warp queue, then hand over to torch
        self._obs()
        self._act()
        v = self.task.sim._signal()
        self.task.sim.after(v)

    def update(self):
        cfg, bufs, n, T = self.cfg, self.bufs, self.task.n, self.cfg.rollout
        obs, act, logp_old, val, rew, done = bufs.t_obs, bufs.t_act, bufs.t_logp, bufs.t_value, bufs.t_rew, bufs.t_done
        last_v = self.pol.t_value[:, 0]
        adv = torch.zeros_like(rew); gae = torch.zeros(n, device="mps")
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + cfg.gamma * nv * nonterm - val[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val
        N = T * n
        o = obs.reshape(N, -1).clone(); a = act.reshape(N, -1).clone(); lp0 = logp_old.reshape(N).clone()
        with torch.no_grad():   # rollout policy's mean/std for the KL schedule
            self._mean0 = self.net.actor_mean(o); self._std0 = self.net.log_std.exp().clone()
        # rows whose observation, action, log-prob or return is not finite (a world that blew up inside the
        # rollout before the task's guard reset it) are dropped from the update instead of crashing it
        ok = torch.isfinite(o).all(-1) & torch.isfinite(a).all(-1) & torch.isfinite(lp0) & torch.isfinite(ret.reshape(N)) & torch.isfinite(adv.reshape(N))
        self.dropped_rows = int((~ok).sum().item())
        if self.dropped_rows:
            keep = ok.nonzero().squeeze(-1); o = o[keep]; a = a[keep]; lp0 = lp0[keep]; adv = adv.reshape(N)[keep]; ret = ret.reshape(N)[keep]; val = val.reshape(N)[keep]; N = int(keep.numel())
            self._mean0 = self._mean0[keep]
        advn = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(N); retf = ret.reshape(N); valf = val.reshape(N).clone()
        mb = N // cfg.minibatches
        stats = {"pg": 0.0, "vf": 0.0, "kl": 0.0}
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device="mps")
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                mean = self.net.actor_mean(o[idx]); v = self.net.critic(o[idx]).squeeze(-1)
                d = torch.distributions.Normal(mean, self.net.log_std.exp())
                lp = d.log_prob(a[idx]).sum(-1)
                ratio = (lp - lp0[idx]).exp()
                pg = -torch.min(ratio * advn[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advn[idx]).mean()
                if cfg.clip_value:      # rsl_rl: value_clipped = old + clamp(v - old, -clip, clip); loss = max of the two errors
                    v_old = valf[idx]
                    v_clipped = v_old + (v - v_old).clamp(-cfg.clip, cfg.clip)
                    vf = torch.max((v - retf[idx]) ** 2, (v_clipped - retf[idx]) ** 2).mean()
                else:
                    vf = F.mse_loss(v, retf[idx])
                ent = d.entropy().sum(-1).mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                if cfg.desired_kl is not None:
                    with torch.no_grad():   # rsl_rl's KL estimate between the rollout policy and the current one
                        kl = (torch.log(self.net.log_std.exp() / self._std0 + 1e-5) + (self._std0 ** 2 + (self._mean0[idx] - mean) ** 2)
                              / (2.0 * self.net.log_std.exp() ** 2) - 0.5).sum(-1).mean().item()
                    if kl > cfg.desired_kl * 2.0:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < cfg.desired_kl / 2.0 and kl > 0.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                self.opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats["pg"] += pg.detach(); stats["vf"] += vf.detach(); stats["kl"] += (lp0[idx] - lp).mean().detach()
        # order the next rollout (Warp reads the weights) after the optimizer's writes
        v = self.learner_event.next_value()
        tb.signal_event(self.learner_event, v)
        self.task.sim.wait(self.learner_event, v)
        k = cfg.epochs * cfg.minibatches
        return {key: float(val_) / k for key, val_ in stats.items()}

    def train(self, log=print, callback=None, monitor=None):
        """``callback(it, algo)`` runs after every iteration (checkpoints, curves); ``monitor(it, algo,
        stats, (ep_ret, ep_len, count))`` runs at every log point (metalsim.learn.monitor.AnomalyMonitor)."""
        cfg = self.cfg
        t0 = time.perf_counter(); steps = 0
        hist = []
        for it in range(1, cfg.iterations + 1):
            self.rollout()
            steps += cfg.rollout * self.task.n
            stats = self.update()
            if it % cfg.log_every == 0 or it == cfg.iterations:
                torch.mps.synchronize()
                ret, length, count = self.task.episode_stats()
                sps = steps / (time.perf_counter() - t0)
                hist.append((it, steps, ret, length))
                extra = ""
                if getattr(self.task, "blown_up_episodes", 0) or getattr(self, "dropped_rows", 0):
                    extra = f" | blown {getattr(self.task, 'blown_up_episodes', 0)} dropped {getattr(self, 'dropped_rows', 0)}"
                log(f"it {it:4d} steps {steps:9d} sps {sps:8,.0f} | ep_ret {ret:7.2f} ep_len {length:6.1f} (n={count}) | "
                    f"pg {stats['pg']:.3f} vf {stats['vf']:.3f} kl {stats['kl']:.4f} lr {self.lr:.1e}{extra}")
                if monitor is not None:
                    monitor(it, self, stats, (ret, length, count))
            if callback is not None:
                callback(it, self)
        return hist


if __name__ == "__main__":
    import sys
    wp.config.quiet = True
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    task = CartpoleTask(n)
    PPOWarp(task, PPOWarpConfig()).train()
