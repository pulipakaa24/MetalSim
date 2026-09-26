"""Where the camera-cartpole rollout spends its GPU time (Isaac-Cartpole-RGB, 1024 envs, tier 0, Isaac's skrl
config), per stage, synchronized (torch.mps + Warp + the renderer context), so host enqueue times do not mislead:

  env pieces:    physics step (BatchSim graph replay), reward / reset / randomize torch ops, the render alone,
                 env.step() as a whole
  policy pieces: NHWC -> NCHW uint8 buffer copy, image_mean, forward_fast (CNN + heads), dist / sample / log_prob
                 + buffer stores, the collect() step body as a whole
  whole rollout: PPO.collect() over 64 steps (overlap between the queues shows up here), and one update()
  counters:      host syncs / command buffers / dispatches per rollout step (metalsim.interop.warp_metal)

Numbers are ms per rollout step (1024 env-steps) unless stated; K repetitions, median of 3 timings.

usage: python scripts/diagnostics/camera_rollout_profile.py [envs=1024] [tier=0]"""
import sys, time
import numpy as np, torch, warp as wp
wp.config.quiet = True
from metalsim.interop import warp_metal as wm
from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
from metalsim.learn.ppo import PPO, PPOConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
TIER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
K = 20
env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=N, tier=TIER, seed=42))
cfg = PPOConfig(total_steps=64 * N * 3, rollout=64, epochs=4, minibatches=32, lr=1e-4, gamma=0.99, lam=0.95, clip=0.2,
                ent_coef=0.0, vf_coef=1.0, max_grad_norm=1.0, desired_kl=0.008, clip_value=True, feat=512,
                activation="elu", qpos_dim=0, center_images=True, value_norm=True, seed=42, log_every=1)
algo = PPO(env, cfg)
obs = env.reset(); env.synchronize()
print(f"N={N}, tier {TIER}, image {env.obs_space['image']}, fast_update {cfg.fast_update}, metal_conv_kernels {getattr(cfg, 'metal_conv_kernels', None)}", flush=True)


def sync():
    env.synchronize()


def timed(label, fn, reps=K, unit="ms/step"):
    fn(); sync()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync(); ts.append((time.perf_counter() - t0) / reps)
    t = sorted(ts)[1]
    print(f"  {label:62s} {t*1e3:8.3f} {unit}", flush=True)
    return t


a = torch.zeros(N, 1, device="mps")
print("--- env pieces")
t_phys = timed("physics: sim.step() (graph replay, decimation substeps)", lambda: env.sim.step())
t_rend = timed("render alone: rend.render(sim) (tier 0, 1024 x 100x100 RGB)", lambda: env._render(env.sim.forward()))
t_fwd = timed("sim.forward() alone", lambda: env.sim.forward())
def reset_ops():
    done = torch.zeros(N, dtype=torch.bool, device="mps"); done[:20] = True
    v = env.sim.reset(done); env.sim.after(v); env._randomize(done); env._learner_done()
t_reset = timed("reset + randomize torch ops (20 envs flagged) + learner event", reset_ops)
t_step = timed("env.step(a) whole (physics + reward/reset + forward + render)", lambda: env.step(a))
print("--- policy pieces (collect() step body)")
obs = env._obs()
img = obs["image"]
t_copy = timed("buf_img[t].copy_(img.permute(0,3,1,2))  (NHWC uint8 -> NCHW)", lambda: algo.buf_img[0].copy_(img.permute(0, 3, 1, 2)))
t_mean = timed("image_mean (per-image channel means, flattened view)", lambda: algo.net.image_mean(algo.buf_img[0]))
mean0 = algo.net.image_mean(algo.buf_img[0])
with torch.no_grad():
    t_fwdnet = timed("forward_fast (uint8 -> centered float, NatureCNN, heads)", lambda: algo.net.forward_fast(algo.buf_img[0], None, mean0))
    mu, v = algo.net.forward_fast(algo.buf_img[0], None, mean0)
    def dist_ops():
        d = algo.net.dist(mu); a_ = d.sample(); algo.buf_a[0] = a_; algo.buf_logp[0] = d.log_prob(a_).sum(-1); algo.buf_v[0] = v
    t_dist = timed("dist / sample / log_prob + buffer stores", dist_ops)
def policy_body():
    with torch.no_grad():
        algo.buf_img[0].copy_(img.permute(0, 3, 1, 2))
        algo.buf_mean[0] = algo.net.image_mean(algo.buf_img[0])
        m_, v_ = algo.net.forward_fast(algo.buf_img[0], None, algo.buf_mean[0])
        d = algo.net.dist(m_); a_ = d.sample(); algo.buf_a[0] = a_; algo.buf_logp[0] = d.log_prob(a_).sum(-1); algo.buf_v[0] = v_
t_policy = timed("policy body whole (copy + mean + forward + dist)", policy_body)
print("--- whole rollout and update")
c0 = wm.counters(); sync()
t0 = time.perf_counter(); obs, last_v = algo.collect(obs); sync(); t_collect = time.perf_counter() - t0
c1 = wm.counters(); d = c1 - c0
print(f"  {'collect() 64 steps: whole rollout':62s} {t_collect*1e3/64:8.3f} ms/step   ({t_collect:.2f} s; sum of pieces {(t_step + t_policy)*1e3:.1f} ms/step)")
if d is not None:
    print(f"  interop counters per rollout step: syncs {d.syncs/64:.2f}, command buffers {d.flushes/64:.1f}, host ops {d.host_ops/64:.2f}, dispatches {d.dispatches/64:.0f}")
adv, ret = algo.gae(last_v)
t0 = time.perf_counter(); algo.update(adv, ret); sync(); t_upd = time.perf_counter() - t0
print(f"  {'update() (4 epochs x 32 minibatches of 2048)':62s} {t_upd*1e3:8.1f} ms/iteration  = {t_upd*1e3/64:.3f} ms per rollout step")
tot = t_collect + t_upd
print(f"  per iteration: rollout {t_collect:.2f} s ({100*t_collect/tot:.0f} %) + update {t_upd:.2f} s ({100*t_upd/tot:.0f} %) -> {64*N/tot:,.0f} env-steps/s incl. training")
print(f"  rollout budget per step: render {t_rend*1e3:.2f} + physics {t_phys*1e3:.2f} + reset {t_reset*1e3:.2f} + policy {t_policy*1e3:.2f} ms")
