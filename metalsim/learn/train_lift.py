"""Train PPO on the GPU SO-101 lift task. Defaults reproduce the mjbatch-metal SB3 baseline config
(n_steps 64, batch 512, 8 epochs, gamma 0.9, ent 0.005, lr 3e-4, N=64) for the acceptance test.
Usage: python -m metalsim.learn.train_lift [--steps 3000000] [--envs 64] [--rollout 64] [--epochs 8] [--minibatches 8] [--log runs/x.log]
"""
import argparse, json, os, sys, time
import torch, warp as wp
from metalsim.learn.so101_lift import LiftConfig, SO101LiftEnv
from metalsim.learn.ppo import PPO, PPOConfig

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3_000_000); ap.add_argument("--envs", type=int, default=64)
ap.add_argument("--rollout", type=int, default=64); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--minibatches", type=int, default=8)
ap.add_argument("--gamma", type=float, default=0.9); ap.add_argument("--ent", type=float, default=0.005); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log", default=None); ap.add_argument("--decimate", type=int, default=2000)
args = ap.parse_args()
wp.config.quiet = True
log_path = args.log or f"runs/lift_N{args.envs}_seed{args.seed}.log"
f = open(log_path, "a")
def log(msg):
    print(msg, flush=True); f.write(msg + "\n"); f.flush()
log(f"config: {json.dumps(vars(args))}")
env = SO101LiftEnv(LiftConfig(num_envs=args.envs, decimate_faces=args.decimate, seed=args.seed))
ppo = PPO(env, PPOConfig(total_steps=args.steps, rollout=args.rollout, epochs=args.epochs, minibatches=args.minibatches,
                         gamma=args.gamma, ent_coef=args.ent, lr=args.lr, seed=args.seed, log_every=5))
t0 = time.perf_counter()
ppo.train(log=log)
dt = time.perf_counter() - t0
recent = ppo.episodes[-200:]
log(f"done: {ppo.global_step} steps in {dt/60:.1f} min ({ppo.global_step/dt:,.0f} sps); last-200-episode success {sum(e[2] for e in recent)/max(len(recent),1):.3f}, return {sum(e[0] for e in recent)/max(len(recent),1):.2f}")
torch.save(ppo.net.state_dict(), log_path.replace(".log", ".pt"))
