"""Video of the G1 flat benchmark: Isaac's G1 (visual meshes + OmniPBR materials from g1_minimal.usd)
driven by the policy trained with Isaac's PPO config for 1,500 iterations, 4 envs, tier 2 path tracing."""
import sys, re, numpy as np, torch, mujoco, warp as wp, imageio.v2 as iio
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model, G1VelocityTask, INIT_POS
from metalsim.learn.warp_policy import ActorCriticMLP
from metalsim.render.tier2 import Tier2Renderer
from metalsim.interop import torch_bridge as tb
from metalsim.physics.batch import BatchSim, BatchSimOptions
ckpt = torch.load(sys.argv[1] if len(sys.argv) > 1 else "runs/policies/g1_flat_1500.pt", map_location="mps")
n = 4
task = G1VelocityTask(n, terrain="flat", seed=3)
# hero model: same physics + visual meshes, camera, key light, floor material, sky
m, info = build_g1_model("flat", visuals=True); spec = info["spec"]
def lookat_quat(pos, target):
    f = np.array(target) - np.array(pos); f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); return q.tolist()
cam = spec.worldbody.add_camera(); cam.name = "hero"; cam.pos = [3.2, -2.4, 1.4]; cam.quat = lookat_quat([3.2, -2.4, 1.4], [0.3, 0.0, 0.6]); cam.fovy = 42
l = spec.worldbody.add_light(); l.pos = [1.5, -2.0, 3.0]; l.dir = [-0.4, 0.5, -0.75]; l.diffuse = [0.9, 0.9, 0.85]; l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
tex = spec.add_texture(); tex.name = "floor"; tex.type = mujoco.mjtTexture.mjTEXTURE_2D; tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
tex.rgb1 = [0.32, 0.33, 0.35]; tex.rgb2 = [0.26, 0.27, 0.29]; tex.width = 256; tex.height = 256
mat = spec.add_material(); mat.name = "floor"; mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "floor"; mat.texrepeat = [8, 8]; mat.texuniform = True; mat.roughness = 0.7
for g in spec.geoms:
    if g.name == "ground": g.material = "floor"
sky = spec.add_texture(); sky.name = "sky"; sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX; sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
sky.rgb1 = [0.55, 0.65, 0.8]; sky.rgb2 = [0.9, 0.92, 0.95]; sky.width = 64; sky.height = 384
hero = spec.compile()
assert hero.nu == task.model.nu and hero.nq == task.model.nq
task.model = hero
task.sim = BatchSim(hero, n, options=BatchSimOptions(substeps=task.decimation, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20))
task.sim.synchronize()
net = ActorCriticMLP(ckpt["obs_dim"], ckpt["act_dim"], hidden=tuple(ckpt["hidden"])).to("mps"); net.load_state_dict(ckpt["net"]); net.eval()
# all four envs at the origin cluster so one camera sees them: 1.2 m apart along y
task.origins.assign(np.array([[0.0, -1.8 + 1.2 * i, 0.0] for i in range(n)], np.float32))
task.reset_all()
step_idx = wp.zeros(1, dtype=int, device="metal:0")
from metalsim.learn.warp_policy import bump
t_obs = tb.mps_tensor(task.obs)
rend = Tier2Renderer(hero, n, width=1024, height=576, camera="hero", spp=16, max_bounces=3, seed=0, decimate_faces=0)
frames = []; steps = 400   # 8 s at 50 Hz control
for t in range(steps):
    wp.launch(bump, dim=1, inputs=[step_idx], device="metal:0")
    task.launch_obs(step_idx)
    v = task.sim._signal(); task.sim.after(v)
    with torch.no_grad():
        a = net.actor(t_obs)
    a_wp = wp.from_torch(a.contiguous()) if False else None
    task.action_scratch.assign(a.cpu().numpy().astype(np.float32))       # host hop is fine for a 4-env video
    task.launch_apply_action(task.action_scratch)
    vs = task.sim.step()
    vr = rend.render(task.sim, vs, passes=8); rend.wait_sim_after_render(task.sim, vr); rend.after(vr)
    img = rend.out.rgb.cpu().numpy()                      # (4, 576, 1024, 3)
    frames.append(np.concatenate([np.concatenate(img[:2], 1), np.concatenate(img[2:], 1)], 0))   # 2x2 -> 2048x1152
    if t % 100 == 0:
        task.sim.synchronize(); q = task.sim.d.qpos.numpy(); print(f"  t={t/50:.0f}s pelvis z {q[:, 2].round(2)} x {q[:, 0].round(2)}", flush=True)
out = "docs/gallery/g1_tier2_policy.mp4"
iio.mimwrite(out, frames, fps=50, codec="libx264", quality=8, macro_block_size=None)
print(f"wrote {out}: {len(frames)} frames, 4 envs, 1024x576 each, tier 2 128 spp", flush=True)
