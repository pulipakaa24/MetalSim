"""Train the G1 flat velocity task on MetalSim physics with the reference learner: the actual
``rsl_rl`` library (rsl-rl-lib 3.1.2, pure PyTorch on MPS) with Isaac Lab's ``G1FlatPPORunnerCfg``
(assets/isaac/g1_rsl_rl_ppo_cfg.py), through ``metalsim.learn.rslrl_adapter.G1RslRlVecEnv``.
This is the learner differential against ``metalsim.learn.ppo_warp`` on the same task and physics.

    python -m metalsim.learn.train_g1_rslrl --envs 4096 --iters 1500 --seed 0 --log runs/g1_flat_rslrl_ppo.log

Writes rsl_rl's own console output (no per-term lines) plus one summary line per iteration:

    RSLRL it <i> mean_reward <rsl_rl 100-episode mean> mean_ep_len <...> | iter_ep_ret <all episodes
    ended this iteration> iter_ep_len <...> (n=<count>, falls <k>) | kl <KL(old||new) over the batch after
    the update> lr <adaptive lr> std <mean action std> | vf <..> surr <..> ent <..> | sps <..>

Per-term episode rewards in Isaac's units go to <log>.terms.jsonl, checkpoints every 100 iterations
to --ckpt_dir in the ``ActorCriticMLP`` format ``scripts/diagnostics/g1_reward_terms.py`` reads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import warp as wp


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


class _NullWriter:
    """rsl_rl requires a summary writer when logging; tensorboard is not installed here."""
    def add_scalar(self, *a, **k): pass
    def save_file(self, *a, **k): pass
    def save_model(self, *a, **k): pass
    def log_config(self, *a, **k): pass


def isaac_g1_flat_runner_cfg(iterations: int, seed: int) -> dict:
    """G1FlatPPORunnerCfg as isaaclab_rl hands it to rsl_rl (RslRlOnPolicyRunnerCfg.to_dict())."""
    return {
        "seed": seed, "device": "mps", "num_steps_per_env": 24, "max_iterations": iterations, "save_interval": 50,
        "experiment_name": "g1_flat", "empirical_normalization": None, "clip_actions": None,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]}, "logger": "tensorboard",
        "policy": {"class_name": "ActorCritic", "init_noise_std": 1.0, "noise_std_type": "scalar",
                   "actor_obs_normalization": False, "critic_obs_normalization": False,
                   "actor_hidden_dims": [256, 128, 128], "critic_hidden_dims": [256, 128, 128], "activation": "elu"},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
                      "entropy_coef": 0.008, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1.0e-3,
                      "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0,
                      "normalize_advantage_per_mini_batch": False, "rnd_cfg": None, "symmetry_cfg": None},
    }


def to_metalsim_ckpt(policy, it: int, obs_dim: int, act_dim: int) -> dict:
    """rsl_rl ActorCritic (scalar std) -> ActorCriticMLP state dict (log_std)."""
    sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items() if k.startswith(("actor.", "critic."))}
    sd["log_std"] = policy.std.detach().clamp_min(1e-6).log().cpu().clone()
    return {"net": sd, "terrain": "flat", "iterations": it, "obs_dim": obs_dim, "act_dim": act_dim,
            "hidden": (256, 128, 128), "learner": "rsl_rl 3.1.2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--physics_dt", type=float, default=0.0025)
    ap.add_argument("--log", default="runs/g1_flat_rslrl_ppo.log")
    ap.add_argument("--run_dir", default="runs/rslrl_g1_flat")
    ap.add_argument("--ckpt_dir", default="runs/policies/g1_flat_rslrl")
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--no_random_ep_len", action="store_true", help="skip Isaac train.py's init_at_random_ep_len=True")
    ap.add_argument("--no_capture", action="store_true")
    a = ap.parse_args()
    wp.config.quiet = True

    from rsl_rl.runners import OnPolicyRunner
    from metalsim.learn.g1_velocity import G1VelocityTask
    from metalsim.learn.rslrl_adapter import G1RslRlVecEnv

    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    os.makedirs(a.run_dir, exist_ok=True); os.makedirs(a.ckpt_dir, exist_ok=True)
    logf = open(a.log, "a")
    sys.stdout = _Tee(sys.__stdout__, logf)
    termf = open(a.log.replace(".log", "") + ".terms.jsonl", "a")

    torch.manual_seed(a.seed)
    task = G1VelocityTask(a.envs, terrain="flat", seed=a.seed, physics_dt=a.physics_dt)
    env = G1RslRlVecEnv(task, capture=not a.no_capture)
    cfg = isaac_g1_flat_runner_cfg(a.iters, a.seed)
    print(f"rsl_rl G1 flat on MetalSim: N={a.envs} obs {task.obs_dim} act {task.act_dim} physics dt {task.physics_dt} "
          f"(decimation {task.decimation}) seed {a.seed} iters {a.iters} init_at_random_ep_len {not a.no_random_ep_len}")
    print("train_cfg " + json.dumps(cfg))

    class Runner(OnPolicyRunner):
        def _prepare_logging_writer(self):
            self.logger_type = "tensorboard"
            self.writer = _NullWriter()

        def log(self, locs, width=80, pad=35):
            super().log(locs, width, pad)
            it = locs["it"]
            ret, length, count = task.episode_stats()
            falls = int(round(task.last_termination_fraction * count))
            rb, lb = locs["rewbuffer"], locs["lenbuffer"]
            mr = sum(rb) / len(rb) if len(rb) else float("nan"); ml = sum(lb) / len(lb) if len(lb) else float("nan")
            ld = locs["loss_dict"]
            sps = self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"])
            print(f"RSLRL it {it:4d} mean_reward {mr:8.2f} mean_ep_len {ml:7.1f} | iter_ep_ret {ret:8.2f} iter_ep_len {length:7.1f} "
                  f"(n={count}, falls {falls}) | kl {self.alg._post_kl:.4f} lr {self.alg.learning_rate:.2e} "
                  f"std {self.alg.policy.std.mean().item():.3f} | vf {ld['value_function']:.4f} surr {ld['surrogate']:.4f} "
                  f"ent {ld['entropy']:.2f} | sps {sps:,.0f} blown {task.blown_up_episodes}")
            ts = env.pop_term_stats(); ts.update({"it": it, "mean_reward": mr, "mean_ep_len": ml, "iter_ep_ret": float(ret),
                                                  "iter_ep_len": float(length), "lr": self.alg.learning_rate,
                                                  "kl_post": self.alg._post_kl, "std": self.alg.policy.std.mean().item()})
            termf.write(json.dumps(ts) + "\n"); termf.flush()
            if (it + 1) % a.ckpt_every == 0 or it + 1 == a.iters:
                torch.save(to_metalsim_ckpt(self.alg.policy, it + 1, task.obs_dim, task.act_dim),
                           os.path.join(a.ckpt_dir, f"g1_flat_rslrl_it{it + 1}.pt"))

    runner = Runner(env, cfg, log_dir=a.run_dir, device="mps")
    runner.git_status_repos = []
    alg = runner.alg
    alg._post_kl = float("nan")
    # KL(old || new) over the whole rollout batch after the 5 x 4 minibatch updates (rsl_rl's own per-minibatch
    # estimate is local to update()); computed just before the storage is cleared
    _clear = alg.storage.clear
    def clear_with_kl():
        st = alg.storage
        with torch.no_grad():
            obs = st.observations.flatten(0, 1)
            mu0 = st.mu.flatten(0, 1); s0 = st.sigma.flatten(0, 1)
            mu = alg.policy.act_inference(obs); s = alg.policy.std.expand_as(mu)
            kl = torch.sum(torch.log(s / s0 + 1e-5) + (s0 ** 2 + (mu0 - mu) ** 2) / (2.0 * s ** 2) - 0.5, dim=-1).mean()
            alg._post_kl = float(kl.item())
        _clear()
    alg.storage.clear = clear_with_kl

    t0 = time.time()
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=not a.no_random_ep_len)
    print(f"done: {a.iters} iterations in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
