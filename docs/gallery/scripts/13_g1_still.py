"""Tier-2 still of Isaac's G1 (visual meshes + OmniPBR materials from g1_minimal.usd) standing on flat ground."""
import numpy as np, mujoco, warp as wp, imageio.v2 as iio
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.render.tier2 import Tier2Renderer
m, info = build_g1_model("flat", visuals=True)
spec = info["spec"]
# camera, key light, sky and a floor material for the picture (the physics model is untouched)
cam = spec.worldbody.add_camera(); cam.name = "hero"; cam.pos = [2.2, -1.6, 1.1]; cam.fovy = 38
# MuJoCo camera frame: looks along -z with y up; build from a look-at direction
def lookat_quat(pos, target):
    f = np.array(target) - np.array(pos); f /= np.linalg.norm(f)          # forward = -z
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); return q.tolist()
cam.quat = lookat_quat([2.2, -1.6, 1.1], [0.0, 0.0, 0.62])
l = spec.worldbody.add_light(); l.pos = [1.5, -2.0, 3.0]; l.dir = [-0.4, 0.5, -0.75]; l.diffuse = [0.9, 0.9, 0.85]; l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
tex = spec.add_texture(); tex.name = "floor"; tex.type = mujoco.mjtTexture.mjTEXTURE_2D; tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
tex.rgb1 = [0.32, 0.33, 0.35]; tex.rgb2 = [0.26, 0.27, 0.29]; tex.width = 256; tex.height = 256
mat = spec.add_material(); mat.name = "floor"; mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "floor"; mat.texrepeat = [8, 8]; mat.texuniform = True; mat.roughness = 0.7
for g in spec.geoms:
    if g.name == "ground": g.material = "floor"
sky = spec.add_texture(); sky.name = "sky"; sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX; sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
sky.rgb1 = [0.55, 0.65, 0.8]; sky.rgb2 = [0.9, 0.92, 0.95]; sky.width = 64; sky.height = 384
m = spec.compile()
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] = 0.78; mujoco.mj_forward(m, d)
rend = Tier2Renderer(m, 1, width=1024, height=768, camera="hero", spp=32, max_bounces=4, decimate_faces=0)
out = rend.render_host([d], passes=16)
iio.imwrite("docs/gallery/g1_tier2_still.png", out["rgb"][0])
print("wrote docs/gallery/g1_tier2_still.png; visual meshes", m.ngeom - 4, "faces", m.mesh_face.shape[0], "seg coverage", (out["seg"][0] > 0).mean().round(3))
