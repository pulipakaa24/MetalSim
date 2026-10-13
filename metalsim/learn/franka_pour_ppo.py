"""PPO for Franka-Pour with the task's rsl_rl configuration, in PyTorch on MPS.

Isaac's runner config (``upstream/IsaacLab3/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/config/franka/
agents/rsl_rl_ppo_cfg.py``) on rsl_rl 5.4.1 (Isaac Lab 3.0-EA pins ``rsl-rl-lib==5.4.1``, ``uv.lock:2084``), transcribed
from that release's ``algorithms/ppo.py``, ``modules/distribution.py``, ``modules/mlp.py``, ``models/mlp_model.py``,
``storage/rollout_storage.py``, ``runners/on_policy_runner.py`` and ``utils/logger.py``:

* actor: ELU MLP 84 -> 512-256-128-64 -> (2, 8) with PyTorch's default init; output [0] is the mean, [1] the log-std,
  clamped to log([0.15, 0.75]) (``HeteroscedasticGaussianDistribution``, std_type "log"); the std head starts with
  zero weights and bias log(0.6 + 1e-7). Inputs: policy + media groups, no normalization.
* critic: ELU MLP 86 -> 512-256-128-64 -> 1 on policy + media + privileged.
* 96 steps per env, 5 epochs x 4 minibatches (one permutation reused for all epochs), clip 0.2, clipped value loss,
  value coef 1.0, entropy 3e-4, Adam lr 1.5e-4 fixed (``schedule="fixed"``: desired_kl unused), gradient norms
  clipped to 1.0 separately for actor and critic, gamma 0.99^(1/3), lambda 0.95^(1/3), advantages normalized over the
  whole batch, time-out bootstrapping r += gamma V(s_t), actions clipped to [-1, 1] by the env wrapper, episodes not
  started at random lengths.
* logging as rsl_rl's logger: mean reward / episode length over the last 100 completed episodes; ``extras["log"]``
  appended every step and averaged per iteration (``Episode_Reward/*``, ``Episode_Termination/*``, ``sampler/*``).

    python -m metalsim.learn.franka_pour_ppo train --num-envs 2048 --iterations 3000 --out runs/franka_pour/run0
    python -m metalsim.learn.franka_pour_ppo bench --num-envs 2048 --mode step|rollout|ppo
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PourPPOConfig:
    num_steps_per_env: int = 96
    max_iterations: int = 3000
    save_interval: int = 25
    clip_actions: float = 1.0
    hidden_dims: tuple = (512, 256, 128, 64)
    init_std: float = 0.60
    std_range: tuple = (0.15, 0.75)
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 3.0e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.5e-4
    schedule: str = "fixed"
    gamma: float = 0.99 ** (1.0 / 3.0)
    lam: float = 0.95 ** (1.0 / 3.0)
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0


def mlp(input_dim: int, output_dim, hidden_dims) -> nn.Sequential:
    """rsl_rl ``modules/mlp.py`` ``MLP`` (ELU hidden activations, linear last layer, optional unflatten)."""
    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ELU()]
    for a, b in zip(hidden_dims[:-1], hidden_dims[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    if isinstance(output_dim, int):
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
    else:
        layers += [nn.Linear(hidden_dims[-1], int(np.prod(output_dim))), nn.Unflatten(-1, tuple(output_dim))]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """``MLPModel`` + ``HeteroscedasticGaussianDistribution(init_std=0.6, std_type="log", std_range=(0.15, 0.75))``."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PourPPOConfig):
        super().__init__()
        self.act_dim = act_dim
        self.mlp = mlp(obs_dim, [2, act_dim], cfg.hidden_dims)
        self.log_std_range = (float(np.log(max(cfg.std_range[0], 1e-6))), float(np.log(cfg.std_range[1])))
        nn.init.zeros_(self.mlp[-2].weight[act_dim:])
        nn.init.constant_(self.mlp[-2].bias[act_dim:], float(torch.log(torch.tensor(cfg.init_std + 1e-7))))

    def forward(self, x):
        out = self.mlp(x)
        mean, log_std = torch.unbind(out, dim=-2)
        log_std = torch.clamp(log_std, self.log_std_range[0], self.log_std_range[1])
        return mean, torch.exp(log_std)


def actor_input(obs):
    return torch.cat([obs["policy"], obs["media"]], dim=-1)


def critic_input(obs):
    return torch.cat([obs["policy"], obs["media"], obs["privileged"]], dim=-1)


class PourPPO:
    def __init__(self, env, cfg: PourPPOConfig | None = None, device: str = "mps"):
        self.env, self.cfg = env, cfg or PourPPOConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.cfg.seed)
        n, T = env.num_envs, self.cfg.num_steps_per_env
        self.actor = Actor(84, 8, self.cfg).to(self.device)
        self.critic = mlp(86, 1, self.cfg.hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=self.cfg.learning_rate)
        self.learning_rate = self.cfg.learning_rate
        d = self.device
        self.obs_a = torch.zeros(T, n, 84, device=d)
        self.obs_c = torch.zeros(T, n, 86, device=d)
        self.actions = torch.zeros(T, n, 8, device=d)
        self.logp = torch.zeros(T, n, device=d)
        self.values = torch.zeros(T, n, device=d)
        self.rewards = torch.zeros(T, n, device=d)
        self.dones = torch.zeros(T, n, device=d)
        self.mu = torch.zeros(T, n, 8, device=d)
        self.sigma = torch.zeros(T, n, 8, device=d)
        # rsl_rl logger state
        self.cur_reward_sum = torch.zeros(n, device=d)
        self.cur_episode_length = torch.zeros(n, device=d)
        self.rewbuffer = collections.deque(maxlen=100)
        self.lenbuffer = collections.deque(maxlen=100)
        self.it = 0
        self.total_steps = 0

    # -- rollout ------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def act(self, obs, t: int):
        xa, xc = actor_input(obs), critic_input(obs)
        mean, std = self.actor(xa)
        dist = Normal(mean, std)
        a = dist.sample()
        self.obs_a[t], self.obs_c[t], self.actions[t] = xa, xc, a
        self.logp[t] = dist.log_prob(a).sum(-1)
        self.values[t] = self.critic(xc).squeeze(-1)
        self.mu[t], self.sigma[t] = mean, std
        return a

    @torch.inference_mode()
    def rollout(self, obs, ep_extras: list, outcomes: collections.Counter):
        cfg, env = self.cfg, self.env
        for t in range(cfg.num_steps_per_env):
            a = self.act(obs, t)
            obs, rew, term, tout, extras = env.step(a)
            dones = term | tout
            self.rewards[t] = rew + cfg.gamma * self.values[t] * tout.float()   # time-out bootstrap (ppo.py:150-155)
            self.dones[t] = dones.float()
            self.cur_reward_sum += rew
            self.cur_episode_length += 1
            ids = dones.nonzero().flatten()
            if ids.numel():
                self.rewbuffer.extend(self.cur_reward_sum[ids].cpu().tolist())
                self.lenbuffer.extend(self.cur_episode_length[ids].cpu().tolist())
                self.cur_reward_sum[ids] = 0
                self.cur_episode_length[ids] = 0
                # outcome of each finished episode (the success metric of the README: terminal particle transfer)
                td = env._term_dones
                for name in ("success", "time_out", "failure", "extreme_rigid_state", "source_receiver_overlap",
                             "spill", "particle_out_of_bounds"):
                    outcomes[name] += int(td[name][ids].sum())
                outcomes["episodes"] += int(ids.numel())
            ep_extras.append(dict(extras["log"]))
        return obs

    @torch.inference_mode()
    def compute_returns(self, obs):
        cfg = self.cfg
        last_values = self.critic(critic_input(obs)).squeeze(-1)
        T = cfg.num_steps_per_env
        self.returns = torch.zeros_like(self.values)
        advantage = 0
        for step in reversed(range(T)):
            next_values = last_values if step == T - 1 else self.values[step + 1]
            nt = 1.0 - self.dones[step]
            delta = self.rewards[step] + nt * cfg.gamma * next_values - self.values[step]
            advantage = delta + nt * cfg.gamma * cfg.lam * advantage
            self.returns[step] = advantage + self.values[step]
        adv = self.returns - self.values
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    def update(self):
        cfg = self.cfg
        N = self.obs_a.shape[0] * self.obs_a.shape[1]
        mb = N // cfg.num_mini_batches
        indices = torch.randperm(cfg.num_mini_batches * mb, device=self.device)
        flat = lambda x: x.flatten(0, 1)  # noqa: E731
        oa, oc, act, v0, ret, lp0, adv = (flat(self.obs_a), flat(self.obs_c), flat(self.actions), flat(self.values),
                                          flat(self.returns), flat(self.logp), flat(self.advantages))
        mu0, sig0 = flat(self.mu), flat(self.sigma)
        stats = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "kl": 0.0}
        for _ in range(cfg.num_learning_epochs):
            for i in range(cfg.num_mini_batches):
                idx = indices[i * mb:(i + 1) * mb]
                mean, std = self.actor(oa[idx])
                dist = Normal(mean, std)
                logp = dist.log_prob(act[idx]).sum(-1)
                entropy = dist.entropy().sum(-1)
                values = self.critic(oc[idx]).squeeze(-1)
                with torch.inference_mode():
                    kl = torch.distributions.kl_divergence(Normal(mu0[idx], sig0[idx]), Normal(mean.detach(),
                                                                                               std.detach())).sum(-1)
                    kl_mean = kl.mean()
                if cfg.desired_kl is not None and cfg.schedule == "adaptive":
                    if kl_mean > cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < cfg.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.learning_rate
                ratio = torch.exp(logp - lp0[idx])
                surrogate = -adv[idx] * ratio
                surrogate_clipped = -adv[idx] * torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                if cfg.use_clipped_value_loss:
                    value_clipped = v0[idx] + (values - v0[idx]).clamp(-cfg.clip_param, cfg.clip_param)
                    value_loss = torch.max((values - ret[idx]).pow(2), (value_clipped - ret[idx]).pow(2)).mean()
                else:
                    value_loss = (ret[idx] - values).pow(2).mean()
                loss = surrogate_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                stats["value"] += value_loss.item()
                stats["surrogate"] += surrogate_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["kl"] += kl_mean.item()
        k = cfg.num_learning_epochs * cfg.num_mini_batches
        return {key: v / k for key, v in stats.items()}

    # -- checkpoints --------------------------------------------------------------------------------------------
    _SAMPLER_TENSORS = ("_success_rates", "_success_history", "_history_pointers", "_history_sizes",
                        "_history_success_counts", "_uniform_order")

    def state_dict(self):
        """Learner state for a resume: networks, Adam moments, iteration / step counters, the curriculum sampler
        (outcome rings, rates, cyclic-replay order, cursor and credit), the RNG states (torch CPU / MPS, the env's
        sampler generator) and rsl_rl's 100-episode logger buffers. Not saved: the physics state of the worlds; a
        resume starts every env from a freshly sampled dataset row (episodes in flight at the pause are dropped
        without a curriculum outcome), so a resumed run is a faithful continuation of the learner, not bitwise of
        the trajectory."""
        s = self.env.curriculum._sampler
        d = {"actor_state_dict": self.actor.state_dict(), "critic_state_dict": self.critic.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(), "iter": self.it, "total_steps": self.total_steps,
             "cfg": asdict(self.cfg), "sampler": {k: getattr(s, k).cpu() for k in self._SAMPLER_TENSORS},
             "sampler_scalars": {"_uniform_cursor": s._uniform_cursor, "_uniform_credit": s._uniform_credit},
             "rng": {"torch_cpu": torch.get_rng_state(), "env_generator": self.env.generator.get_state()},
             "logger": {"rewbuffer": list(self.rewbuffer), "lenbuffer": list(self.lenbuffer)}}
        if self.device.type == "mps":
            d["rng"]["torch_mps"] = torch.mps.get_rng_state()
        return d

    def load_state_dict(self, d):
        self.actor.load_state_dict(d["actor_state_dict"])
        self.critic.load_state_dict(d["critic_state_dict"])
        self.optimizer.load_state_dict(d["optimizer_state_dict"])
        self.it, self.total_steps = int(d["iter"]), int(d["total_steps"])
        s = self.env.curriculum._sampler
        for k, v in d.get("sampler", {}).items():
            getattr(s, k).copy_(v.to(getattr(s, k).device))
        for k, v in d.get("sampler_scalars", {}).items():
            setattr(s, k, v)
        rng = d.get("rng", {})
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if "env_generator" in rng:
            self.env.generator.set_state(rng["env_generator"])
        if "torch_mps" in rng and self.device.type == "mps":
            torch.mps.set_rng_state(rng["torch_mps"])
        self.rewbuffer.extend(d.get("logger", {}).get("rewbuffer", []))
        self.lenbuffer.extend(d.get("logger", {}).get("lenbuffer", []))

    # -- training loop (OnPolicyRunner.learn) -------------------------------------------------------------------
    def train(self, iterations: int, out_dir: str | None = None, log=print):
        cfg, env = self.cfg, self.env
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        obs = env.reset_all() if env.obs_buf is None else env.obs_buf
        logf = open(os.path.join(out_dir, "log.jsonl"), "a") if out_dir else None
        t_start = time.perf_counter()
        steps0 = self.total_steps
        start_it = self.it
        for it in range(start_it, start_it + iterations):
            t0 = time.perf_counter()
            ep_extras, outcomes = [], collections.Counter()
            obs = self.rollout(obs, ep_extras, outcomes)
            env.physics.check_status()  # sparse-grid capacity overflow during the rollout raises here
            self.compute_returns(obs)
            t1 = time.perf_counter()
            losses = self.update()
            torch.mps.synchronize() if self.device.type == "mps" else None
            t2 = time.perf_counter()
            self.it = it + 1
            n_steps = cfg.num_steps_per_env * env.num_envs
            self.total_steps += n_steps
            with torch.inference_mode():
                _, std = self.actor(actor_input(obs))
            rec = {"it": it, "steps": self.total_steps, "collect_s": t1 - t0, "learn_s": t2 - t1,
                   "fps": n_steps / (t2 - t0),
                   "mean_reward": float(np.mean(self.rewbuffer)) if self.rewbuffer else None,
                   "mean_episode_length": float(np.mean(self.lenbuffer)) if self.lenbuffer else None,
                   "episodes": outcomes["episodes"],
                   "success_rate": outcomes["success"] / outcomes["episodes"] if outcomes["episodes"] else None,
                   "outcomes": dict(outcomes), "loss": losses, "lr": self.learning_rate,
                   "mean_std": float(std.mean()), "wall_s": time.perf_counter() - t_start}
            keys = {k for e in ep_extras for k in e}
            rec["extras"] = {k: float(np.mean([e[k] for e in ep_extras if k in e])) for k in sorted(keys)}
            if logf:
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            sr = rec["success_rate"]
            log(f"it {it:5d} steps {self.total_steps:11d} fps {rec['fps']:7.0f} | rew {rec['mean_reward'] or 0:7.3f} "
                f"len {rec['mean_episode_length'] or 0:6.1f} | eps {outcomes['episodes']:5d} success "
                f"{(sr if sr is not None else float('nan')):.3f} succ_term {rec['extras'].get('Episode_Termination/success', 0):.3f} "
                f"| v {losses['value']:.4f} s {losses['surrogate']:.4f} ent {losses['entropy']:.3f} kl {losses['kl']:.4f} "
                f"std {rec['mean_std']:.3f} | {t1 - t0:.1f}+{t2 - t1:.1f} s")
            stop = bool(out_dir) and os.path.exists(os.path.join(out_dir, "STOP"))
            if out_dir and (it % cfg.save_interval == 0 or it == start_it + iterations - 1 or stop):
                torch.save(self.state_dict(), os.path.join(out_dir, f"model_{it}.pt"))
                torch.save(self.state_dict(), os.path.join(out_dir, "model_last.pt"))
            if stop:      # clean pause: `touch <out>/STOP`; resume with --resume <out>/model_last.pt
                os.remove(os.path.join(out_dir, "STOP"))
                log(f"paused after iteration {it} (STOP file); checkpoint {out_dir}/model_last.pt")
                break
        if logf:
            logf.close()
        return (self.total_steps - steps0) / (time.perf_counter() - t_start)


# ------------------------------------------------------------------------------------------------ CLI
def _make_env(args):
    from metalsim.learn import franka_pour as P

    pcfg = P.S.PourPhysicsConfig(num_envs=args.num_envs, rigid_deterministic=args.rigid_det,
                                 metal_fast_math=args.metal_fast_math, capture_unit=args.capture_unit,
                                 mpm_grid_budget=args.mpm_grid_budget, mpm_cells_per_world=args.mpm_cells_per_world)
    return P.FrankaPourEnv(P.PourTaskConfig(num_envs=args.num_envs, seed=args.seed, physics=pcfg))


def bench(args):
    """Throughput at ``num_envs`` (synchronized wall time): ``step`` = env.step with uniform random actions in
    [-1, 1] (Isaac's benchmark_non_rl.py protocol), ``rollout`` = policy inference + env.step + storage,
    ``ppo`` = the full loop (rollout of 96 steps + update)."""
    import warp as wp

    env = _make_env(args)
    obs = env.reset_all()
    res = {"num_envs": args.num_envs, "mode": args.mode, "steps": args.steps, "warmup": args.warmup}
    g = torch.Generator(device=env.device).manual_seed(args.seed)
    if args.mode in ("step", "rollout"):
        ppo = PourPPO(env, PourPPOConfig(num_steps_per_env=max(args.steps, 1), seed=args.seed)) \
            if args.mode == "rollout" else None
        times = []
        for k in range(args.warmup + args.steps):
            t0 = time.perf_counter()
            if ppo is None:
                a = torch.rand(env.num_envs, 8, device=env.device, generator=g) * 2 - 1
            else:
                a = ppo.act(obs, max(k - args.warmup, 0))
            obs, rew, term, tout, extras = env.step(a)
            torch.mps.synchronize()
            wp.synchronize_device(env.physics.device)
            times.append(time.perf_counter() - t0)
        t = np.asarray(times[args.warmup:])
        res.update(step_s_mean=float(t.mean()), step_s_median=float(np.median(t)),
                   env_steps_per_s=float(env.num_envs / t.mean()))
    else:
        ppo = PourPPO(env, PourPPOConfig(num_steps_per_env=args.rollout_len, seed=args.seed))
        ppo.train(args.warmup)
        t0 = time.perf_counter()
        ppo.train(args.steps)
        dt = time.perf_counter() - t0
        res.update(iterations=args.steps, rollout_len=args.rollout_len, wall_s=dt,
                   env_steps_per_s=float(args.steps * args.rollout_len * env.num_envs / dt))
    res["physics"] = {"capture_unit": args.capture_unit, "rigid_deterministic": args.rigid_det,
                      "metal_fast_math": args.metal_fast_math}
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


def train(args):
    env = _make_env(args)
    ppo = PourPPO(env, PourPPOConfig(seed=args.seed, max_iterations=args.iterations))
    if args.resume and os.path.exists(args.resume):
        ppo.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False))
        print(f"resumed from {args.resume} at iteration {ppo.it}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "ppo": asdict(ppo.cfg)}, f, indent=1, default=str)
    remaining = args.iterations - ppo.it
    ppo.train(max(remaining, 0), out_dir=args.out, log=lambda s: print(s, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "bench"])
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--out", default="runs/franka_pour/run")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["step", "rollout", "ppo"], default="step")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rollout-len", type=int, default=96)
    ap.add_argument("--rigid-det", action="store_true")
    ap.add_argument("--metal-fast-math", default="off", choices=["off", "cuda", "relaxed"])
    ap.add_argument("--capture-unit", default="sim", choices=["sim", "policy"])
    ap.add_argument("--mpm-grid-budget", default="isaac", choices=["isaac", "measured"],
                    help="sparse-grid NanoVDB node capacities: Newton's automatic ones or the measured budgets")
    ap.add_argument("--mpm-cells-per-world", type=int, default=None,
                    help="MPM cell reservation per world (default: Isaac's rule, 1,024 at 735 particles)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    (train if args.cmd == "train" else bench)(args)


if __name__ == "__main__":
    main()
