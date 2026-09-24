"""Video of the Cartpole benchmark: train Isaac Lab's cartpole config on the Warp rollout path, then
roll 4 envs with the trained policy and path-trace them at 512 px, 512 samples per pixel (32 spp x 16
progressive passes, 3 bounces; enough that no Monte Carlo grain is visible) into a 2x2 grid."""
import sys, time, numpy as np, torch, warp as wp, imageio.v2 as iio
wp.config.quiet = True
from metalsim.learn.ppo_warp import PPOWarp, PPOWarpConfig, CartpoleTask
from metalsim.render.tier2 import Tier2Renderer
from metalsim.interop import torch_bridge as tb

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
t0 = time.time()
train = CartpoleTask(4096, seed=0)
algo = PPOWarp(train, PPOWarpConfig(iterations=iters, rollout=16, log_every=20))
last = {}
def _log(m):
    print("  " + m[:100], flush=True); last["m"] = m
algo.train(log=_log)
import re
ret = float(re.search(r"ep_ret\s+([-\d.]+)", last["m"]).group(1))
print(f"trained in {time.time() - t0:.0f} s, final mean return {ret:.1f} / 300", flush=True)
assert ret > 270, "policy not trained enough for a video"

n = 4
task = CartpoleTask(n, seed=1)
# hero look for the video only (the benchmark scene keeps Isaac's camera spec): the same cartpole with
# a brushed-steel rail, satin plastic cart, matte pole, a wider camera that keeps the +-3 m rail in
# frame, a warm key light and a bright sky for soft fill. Physics is untouched.
import mujoco
from metalsim.learn.cartpole_rgb import CARTPOLE_XML
spec = mujoco.MjSpec.from_string(CARTPOLE_XML)
for g in spec.geoms:
    if g.name == "rail": g.rgba = [0.75, 0.76, 0.78, 1]
    if g.name == "cart": g.rgba = [0.12, 0.35, 0.85, 1]
    if g.name == "pole": g.rgba = [0.92, 0.42, 0.15, 1]
for nm, rough, metal in (("rail", 0.25, 0.9), ("cart", 0.45, 0.0), ("pole", 0.6, 0.0)):
    mm = spec.add_material(); mm.name = nm + "_mat"; mm.roughness = rough; mm.metallic = metal; mm.rgba = next(g.rgba for g in spec.geoms if g.name == nm)
    next(g for g in spec.geoms if g.name == nm).material = nm + "_mat"
for mm in spec.materials:
    if mm.name == "grid": mm.roughness = 0.8
for c in spec.cameras:
    if c.name == "cam": c.pos = [0.0, -6.8, 1.9]; c.fovy = 52
for l in spec.lights:
    l.diffuse = [1.0, 0.95, 0.85]; l.pos = [2.0, -3.0, 4.0]; l.dir = [-0.35, 0.45, -0.82]
for t in spec.textures:
    if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX: t.rgb1 = [0.62, 0.72, 0.88]; t.rgb2 = [0.93, 0.95, 0.98]
hero_model = spec.compile()
# the renderer reads camera and light poses from the sim's buffers, so the sim itself runs the hero
# model (identical physics: only materials, camera and light parameters differ)
from metalsim.physics.batch import BatchSim, BatchSimOptions
task.model = hero_model
task.sim = BatchSim(hero_model, n, options=BatchSimOptions(substeps=2, njmax=32, solver_iterations=10, ls_iterations=10))
task.sim.synchronize()
rend = Tier2Renderer(hero_model, n, width=512, height=512, camera="cam", spp=32, max_bounces=3, seed=0)
PASSES = 16
q = task.sim.t.qpos; v = task.sim.t.qvel
gen = torch.Generator(device="mps").manual_seed(1)
q[:, 0] = torch.rand(n, generator=gen, device="mps") * 2 - 1; q[:, 1] = (torch.rand(n, generator=gen, device="mps") - 0.5) * 0.5
v[:, :2] = (torch.rand(n, 2, generator=gen, device="mps") - 0.5)
task.sim.synchronize()
t_obs = tb.mps_tensor(task.obs)
frames = []
steps = 300   # 5 s at 60 Hz control (decimation 2 of 120 Hz physics)
print('rendering', steps, 'frames at 512 spp', flush=True)
for t in range(steps):
    task.launch_obs()
    vsig = task.sim._signal(); task.sim.after(vsig)
    with torch.no_grad():
        a = algo.net.actor(t_obs).clamp(-1, 1)
    task.sim.t.ctrl[:] = a
    vl = algo.learner_event.next_value(); tb.signal_event(algo.learner_event, vl); task.sim.wait(algo.learner_event, vl)
    vs = task.sim.step()
    vr = rend.render(task.sim, vs, passes=PASSES); rend.wait_sim_after_render(task.sim, vr); rend.after(vr)
    img = rend.out.rgb.cpu().numpy()                        # (4, 512, 512, 3)
    grid = img.reshape(2, 2, 512, 512, 3).transpose(0, 2, 1, 3, 4).reshape(1024, 1024, 3)
    frames.append(grid)
    if t % 60 == 0:
        task.sim.synchronize(); qq = task.sim.d.qpos.numpy(); print(f"  t={t/60:.0f}s |pole angle| mean {np.abs(qq[:,1]).mean():.3f} rad, max {np.abs(qq[:,1]).max():.3f}", flush=True)
out = "docs/gallery/cartpole_tier2_policy.mp4"
iio.mimwrite(out, frames, fps=60, codec="libx264", quality=8, macro_block_size=None)
qq = task.sim.d.qpos.numpy()
print(f"wrote {out}: {len(frames)} frames, 4 envs, tier 2 512 spp 3 bounces; final |pole angle| max {np.abs(qq[:,1]).max():.3f} rad, |cart x| max {np.abs(qq[:,0]).max():.2f} m; total {time.time() - t0:.0f} s", flush=True)
