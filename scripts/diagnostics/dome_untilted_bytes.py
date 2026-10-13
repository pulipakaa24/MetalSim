"""Untilted-dome renders for a byte comparison across two checkouts (tilted-dome change, 2026-10-13).

Renders the environment-map paths that existed before FLAG_ENVROT (sky colour, set_environment with a yaw,
set_environment_pose, a USD DomeLight with rotateZ through set_environment_from_model, a sky-only camera) and
writes every output buffer to an .npz. Run it with PYTHONPATH pointing at each checkout and compare:

    PYTHONPATH=<old> python scripts/diagnostics/dome_untilted_bytes.py --out old.npz
    PYTHONPATH=<new> python scripts/diagnostics/dome_untilted_bytes.py --out new.npz --compare old.npz
"""
import argparse
import os
import tempfile

import mujoco
import numpy as np

ENV_PLANE = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset><material name="white" rgba="1 1 1 1" specular="0.3" shininess="0.5"/></asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.2 -1" diffuse="0.5 0.5 0.5"/>
    <camera name="cam" pos="0 -1.5 1.2" xyaxes="1 0 0 0 0.6 0.8" fovy="60"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="white"/>
    <geom name="ball" type="sphere" size="0.3" pos="0 0 0.3" material="white"/>
  </worldbody>
</mujoco>
"""
SKY_CAM = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="0.7071 -0.7071 0 -0.4082 -0.4082 0.8165" fovy="100"/>
    <geom name="speck" type="sphere" size="0.001" pos="-30 -30 -30"/>
  </worldbody>
</mujoco>
"""


def _fwd(m):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return d


def _map(h=128, w=256):
    u, v = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    env = np.stack([0.2 + 0.8 * u, 0.2 + 0.8 * v, 0.5 + 0.4 * np.sin(6 * u) * np.cos(3 * v)], -1).astype(np.float32)
    env[30:38, 40:52] = (40.0, 30.0, 20.0)
    return env


def _write_rgbe(path, rgb):
    rgb = np.asarray(rgb, np.float32)
    m = rgb.max(-1)
    e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, 0)
    scale = np.where(m > 1e-32, 256.0 / np.exp2(e), 0.0)
    rgbe = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {rgb.shape[0]} +X {rgb.shape[1]}\n".encode())
        f.write(rgbe.tobytes())


def _grab(out, prefix, res):
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            res[f"{prefix}/{k}"] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    a = ap.parse_args()
    import metalsim
    from metalsim.render.tier2 import Tier2Renderer
    print("metalsim from", os.path.dirname(metalsim.__file__))
    res = {}
    m = mujoco.MjModel.from_xml_string(ENV_PLANE)
    r = Tier2Renderer(m, 2, width=64, height=48, camera="cam", spp=8, max_bounces=3, seed=7)
    _grab(r.render_host([_fwd(m), _fwd(m)], passes=2), "sky", res)
    env = _map()
    for yaw in (0.0, 1.3, -2.2):
        r = Tier2Renderer(m, 2, width=64, height=48, camera="cam", spp=8, max_bounces=3, seed=7)
        r.set_environment(env, yaw=yaw)
        _grab(r.render_host([_fwd(m), _fwd(m)], passes=2), f"env_yaw{yaw}", res)
        r.set_environment_pose(yaw + 0.5, intensity=2.0, exposure=-1.0, color=(1.0, 0.8, 0.6))
        _grab(r.render_host([_fwd(m), _fwd(m)], passes=1), f"env_pose{yaw}", res)
    ms = mujoco.MjModel.from_xml_string(SKY_CAM)
    r = Tier2Renderer(ms, 1, width=48, height=48, camera="cam", spp=1, max_bounces=1, center_sample=True, seed=1)
    r.set_environment(env, yaw=0.4)
    _grab(r.render_host([_fwd(ms)], passes=1), "skycam", res)
    # a USD stage's DomeLight with rotateZ (the importer path: yaw + KIT_DOME_AZIMUTH)
    from pxr import Usd, UsdGeom, UsdLux
    from metalsim.scene.usd_to_mjcf import load_usd
    d = tempfile.mkdtemp()
    _write_rgbe(os.path.join(d, "map.hdr"), env)
    st = Usd.Stage.CreateNew(os.path.join(d, "dome.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    w = UsdGeom.Xform.Define(st, "/World"); st.SetDefaultPrim(w.GetPrim())
    dome = UsdLux.DomeLight.Define(st, "/World/sky")
    dome.CreateTextureFileAttr("./map.hdr"); dome.CreateTextureFormatAttr("latlong"); dome.CreateIntensityAttr(1.5)
    UsdGeom.Xformable(dome.GetPrim()).AddRotateZOp().Set(30.0)
    UsdGeom.Sphere.Define(st, "/World/ball").CreateRadiusAttr(0.3)
    cam = UsdGeom.Camera.Define(st, "/World/cam")
    UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set((0.0, 0.0, 2.0))
    st.GetRootLayer().Save()
    mu = load_usd(os.path.join(d, "dome.usda")).compile()
    r = Tier2Renderer(mu, 1, width=48, height=48, camera="cam", spp=8, max_bounces=2, headlight=False, seed=2)
    r.set_environment_from_model(mu)
    _grab(r.render_host([_fwd(mu)], passes=2), "usd_rotZ30", res)
    np.savez(a.out, **res)
    print(f"wrote {len(res)} arrays to {a.out}")
    if a.compare:
        old = np.load(a.compare)
        bad = [k for k in res if k not in old or old[k].tobytes() != res[k].tobytes()]
        missing = [k for k in old.files if k not in res]
        for k in sorted(res):
            same = k in old and old[k].tobytes() == res[k].tobytes()
            print(f"  {k:28s} {res[k].dtype} {res[k].shape} {'identical' if same else 'DIFFERS'}")
        print("BYTE-IDENTICAL" if not bad and not missing else f"DIFFERENT: {bad} missing {missing}")
        raise SystemExit(0 if not bad and not missing else 1)


if __name__ == "__main__":
    main()
