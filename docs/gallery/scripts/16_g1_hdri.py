"""Tier-2 still of Isaac's G1 lit by a Poly Haven HDRI (CC0, fetched by scripts/fetch_polyhaven_assets.sh) through
the environment map: importance-sampled image-based lighting with MIS, OmniPBR materials, RTX display transform,
no sun and no headlight (the map is the only light). 512 spp, 4 bounces, no denoiser."""
import sys
import numpy as np, mujoco, warp as wp, imageio.v2 as iio
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.render.tier2 import Tier2Renderer
from metalsim.render.hdr import load_hdr

HDR = sys.argv[1] if len(sys.argv) > 1 else "assets/polyhaven/hdri/train/lebombo.hdr"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/gallery/g1_hdri_tier2.png"
YAW = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
m, info = build_g1_model("flat", visuals=True)
spec = info["spec"]
cam = spec.worldbody.add_camera(); cam.name = "hero"; cam.pos = [2.2, -1.6, 1.1]; cam.fovy = 38
def lookat_quat(pos, target):
    f = np.array(target) - np.array(pos); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); return q.tolist()
cam.quat = lookat_quat([2.2, -1.6, 1.1], [0.0, 0.0, 0.62])
tex = spec.add_texture(); tex.name = "floor"; tex.type = mujoco.mjtTexture.mjTEXTURE_2D; tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
tex.rgb1 = [0.32, 0.33, 0.35]; tex.rgb2 = [0.26, 0.27, 0.29]; tex.width = 256; tex.height = 256
mat = spec.add_material(); mat.name = "floor"; mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "floor"; mat.texrepeat = [8, 8]; mat.texuniform = True; mat.roughness = 0.7
for g in spec.geoms:
    if g.name == "ground": g.material = "floor"; g.size = [50.0, 50.0, 0.05]
    elif g.contype != 0 or g.conaffinity != 0: g.group = 4   # collision geometry is not drawn (Isaac draws visuals only)
m = spec.compile()
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] = 0.78; mujoco.mj_forward(m, d)
rend = Tier2Renderer(m, 1, width=1024, height=768, camera="hero", spp=32, max_bounces=4, decimate_faces=0,
                     material_model="omnipbr", headlight=False, tonemap="rtx", exposure=1.0)
hdr = load_hdr(HDR)
rend.set_environment(hdr, yaw=YAW, key=HDR)
out = rend.render_host([d], passes=16)
iio.imwrite(OUT, out["rgb"][0])
print(f"wrote {OUT}; map {HDR} {hdr.shape} mean {hdr.mean():.3f} max {hdr.max():.1f}; seg coverage {(out['seg'][0] > 0).mean():.3f}; hdr mean {out['hdr'][0].mean():.3f}")
