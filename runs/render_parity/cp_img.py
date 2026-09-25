import numpy as np, torch, imageio.v2 as iio, warp as wp; wp.config.quiet = True
from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
ims = []
for mode, spp, ex in (("legacy", 1, 1), ("physical", 4, 1.0), ("physical", 4, 0.25), ("physical", 4, 0.15)):
    env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=4, tier=2, render_mode=mode, spp=spp, tier2_exposure=ex)); obs = env.reset()
    for _ in range(3): obs, *_ = env.step(torch.zeros(4, 1, device="mps"))
    env.synchronize(); im = obs["image"][0].cpu().numpy(); ims.append(im)
    print(mode, spp, ex, "mean", im.mean().round(1), "std", im.std().round(1), "frac>=250", (im >= 250).mean().round(3), "per-channel", im.reshape(-1, 3).mean(0).round(1))
iio.imwrite("runs/render_parity/cartpole_modes.png", np.concatenate(ims, 1))
