"""Camera-based RL on the Cartpole-RGB benchmark: Isaac Lab's Isaac-Cartpole-RGB-Camera-Direct-v0
(100x100 RGB tiled camera, image-only observation) with the skrl agent configuration Isaac Lab ships
for it (assets/isaac/cartpole_skrl_camera_ppo_cfg.yaml): NatureCNN 32-64-64 + 512 ELU shared trunk,
rollouts 64, 4 epochs, 32 minibatches, lr 1e-4 with KL-adaptive schedule (0.008), clip 0.2, value
clip 0.2, grad clip 1.0, entropy 0. Renderer tier selectable (0 raster, 1 hybrid RT, 2 path traced).

    python -m metalsim.learn.train_cartpole_rgb --envs 1024 --tier 0 --steps 4000000 --log runs/x.log
"""
import argparse, time
import torch, warp as wp
from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
from metalsim.learn.ppo import PPO, PPOConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024); ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--spp", type=int, default=1); ap.add_argument("--bounces", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4_000_000); ap.add_argument("--log", default=None)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    wp.config.quiet = True
    env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=a.envs, tier=a.tier, spp=a.spp, max_bounces=a.bounces, seed=a.seed))
    cfg = PPOConfig(total_steps=a.steps, rollout=64, epochs=4, minibatches=32, lr=1e-4, gamma=0.99, lam=0.95, clip=0.2,
                    ent_coef=0.0, vf_coef=1.0, max_grad_norm=1.0, desired_kl=0.008, clip_value=True, feat=512,
                    activation="elu", qpos_dim=0, seed=a.seed, log_every=1)
    f = open(a.log, "a") if a.log else None
    def log(msg):
        print(msg, flush=True)
        if f: f.write(msg + "\n"); f.flush()
    log(f"Cartpole-RGB camera PPO: {a.envs} envs, tier {a.tier} (spp {a.spp}, bounces {a.bounces}), image-only 100x100, Isaac skrl config")
    algo = PPO(env, cfg)
    t0 = time.time(); algo.train(log=log); dt = time.time() - t0
    log(f"done: {algo.global_step} steps in {dt:.0f} s -> {algo.global_step / dt:,.0f} env-steps/s incl. training; {algo.timers.report()}")
    if a.checkpoint:
        torch.save({"net": algo.net.state_dict(), "tier": a.tier, "cfg": cfg.__dict__}, a.checkpoint)

if __name__ == "__main__":
    main()
