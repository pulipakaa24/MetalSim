"""Where the PPO update (metalsim.learn.ppo_warp.PPOWarp.update, torch on MPS) spends its time for the G1 flat task
at 4096 envs: whole update, then the same update with one piece removed at a time (timing only, the results are
not used): the GAE recursion, the adaptive-KL `.item()` per minibatch, clip_grad_norm_, the optimizer step, and
the forward+backward alone. Median of 3 updates after one warm-up, on one real rollout.

usage: python scripts/diagnostics/ppo_update_profile.py [N=4096]"""
import sys, time
import numpy as np, torch, warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask, g1_ppo_config
from metalsim.learn.ppo_warp import PPOWarp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
task = G1VelocityTask(N, terrain="flat", physics_dt=0.0025, seed=0)
task.reset_all()
algo = PPOWarp(task, g1_ppo_config("flat", 1))
algo.rollout(); task.sim.synchronize(); torch.mps.synchronize()
cfg = algo.cfg
print(f"N={N}, rollout {cfg.rollout}, epochs {cfg.epochs} x minibatches {cfg.minibatches} (minibatch {cfg.rollout*N//cfg.minibatches} rows), "
      f"net {cfg.hidden}, params {sum(p.numel() for p in algo.net.parameters())}", flush=True)


def timed(label, fn, reps=3):
    state = {k: v.clone() for k, v in algo.net.state_dict().items()}
    fn(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        algo.net.load_state_dict(state)
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    algo.net.load_state_dict(state)
    print(f"  {label:58s} {sorted(ts)[1]*1e3:7.1f} ms", flush=True)
    return sorted(ts)[1]


full = timed("whole update", algo.update)
kl = cfg.desired_kl; cfg.desired_kl = None
timed("without the adaptive-KL .item() per minibatch", algo.update); cfg.desired_kl = kl
import torch.nn.utils as U
orig_clip = U.clip_grad_norm_; U.clip_grad_norm_ = lambda *a, **k: torch.zeros(())
timed("without clip_grad_norm_", algo.update); U.clip_grad_norm_ = orig_clip
orig_step = algo.opt.step; algo.opt.step = lambda *a, **k: None
timed("without the optimizer step (Adam)", algo.update); algo.opt.step = orig_step
ep, mb = cfg.epochs, cfg.minibatches; cfg.epochs = 1; cfg.minibatches = 1
timed("1 epoch x 1 minibatch (GAE + one big fwd/bwd + one step)", algo.update); cfg.epochs, cfg.minibatches = ep, mb


def gae_only():
    bufs = algo.bufs; T = cfg.rollout
    val, rew, done = bufs.t_value, bufs.t_rew, bufs.t_done
    last_v = algo.pol.t_value[:, 0]
    adv = torch.zeros_like(rew); gae = torch.zeros(N, device="mps")
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + cfg.gamma * nv * nonterm - val[t]
        gae = delta + cfg.gamma * cfg.lam * nonterm * gae
        adv[t] = gae
    return adv


timed("GAE recursion alone (24 sequential steps)", gae_only)


def fwd_bwd_only():
    o = algo.bufs.t_obs.reshape(cfg.rollout * N, -1); a = algo.bufs.t_act.reshape(cfg.rollout * N, -1)
    mbn = o.shape[0] // cfg.minibatches
    for _ in range(cfg.epochs):
        for i in range(cfg.minibatches):
            idx = slice(i * mbn, (i + 1) * mbn)
            mean = algo.net.actor_mean(o[idx]); v = algo.net.critic(o[idx]).squeeze(-1)
            d = torch.distributions.Normal(mean, algo.net.log_std.exp())
            loss = -d.log_prob(a[idx]).sum(-1).mean() + v.pow(2).mean() - 0.005 * d.entropy().sum(-1).mean()
            algo.opt.zero_grad(set_to_none=True); loss.backward()


timed("forward + backward of the 20 minibatches alone (no clip, no step)", fwd_bwd_only)
