"""How much of the PPO update (G1 flat, 4096 envs, 24-step rollout, 5 epochs x 4 minibatches) is the per-minibatch
host read of the KL (rsl_rl's adaptive learning rate: one .item() per minibatch)? Times update() with the adaptive
schedule on and off after the same rollout (timing only; the schedule itself is unchanged in training).

usage: python scripts/diagnostics/g1_update_syncs.py"""
import time, torch, warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask, g1_ppo_config
from metalsim.learn.ppo_warp import PPOWarp

task = G1VelocityTask(4096, terrain="flat", physics_dt=0.0025)
algo = PPOWarp(task, g1_ppo_config("flat", 1))
algo.rollout(); task.sim.synchronize(); torch.mps.synchronize()
for kl in (0.01, None, 0.01, None):
    algo.cfg.desired_kl = kl
    ts = []
    for _ in range(3):
        algo.rollout(); task.sim.synchronize(); torch.mps.synchronize()
        t0 = time.perf_counter(); algo.update(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    print(f"desired_kl {kl}: update {sorted(ts)[1]*1e3:.0f} ms (median of 3)", flush=True)
