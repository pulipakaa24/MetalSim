# the failing command's configuration, 3 PPO iterations, with GPU-memory and finiteness probes
import sys, torch, numpy as np, warp as wp; wp.config.quiet = True
from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
from metalsim.learn.ppo import PPO, PPOConfig
env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=1024, tier=2, render_mode="physical", seed=42))
cfg = PPOConfig(total_steps=3 * 65536, rollout=64, epochs=4, minibatches=32, lr=1e-4, gamma=0.99, lam=0.95, clip=0.2,
                ent_coef=0.0, vf_coef=1.0, max_grad_norm=1.0, desired_kl=0.008, clip_value=True, feat=512,
                activation="elu", qpos_dim=0, center_images=True, value_norm=True, seed=42, log_every=1)
algo = PPO(env, cfg)
def log(m):
    print(m[:110], f"| mps alloc {torch.mps.current_allocated_memory()/1e9:.2f} GB driver {torch.mps.driver_allocated_memory()/1e9:.2f} GB rec max {torch.mps.recommended_max_memory()/1e9:.1f} GB", flush=True)
algo.train(log=log)
env.synchronize(); print("accum finite:", bool(np.isfinite(env.rend._accum.numpy()).all()))
