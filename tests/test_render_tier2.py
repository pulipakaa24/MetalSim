"""Tier-2 path tracer acceptance: analytic radiometry (Lambertian plane under a directional light,
white furnace under a uniform sky), Monte Carlo convergence, agreement of the direct-light term
with tier 0, GPU-path ordering against BatchSim, and a full Cartpole-RGB rollout at tier 2."""
import os

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.render.tier0 import Tier0Renderer
from metalsim.render.tier2 import Tier2Renderer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "measurements")

# camera straight down onto a matte grey plane; one directional light straight down; no headlight
LAMBERT = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset><material name="matte" rgba="0.8 0.8 0.8 1" specular="0" shininess="0"/></asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.5 0.5 0.5" specular="0 0 0" ambient="0 0 0"/>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="30"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="matte"/>
  </worldbody>
</mujoco>
"""

FURNACE = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5" width="8" height="48"/>
    <material name="white" rgba="1 1 1 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="30"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="white"/>
  </worldbody>
</mujoco>
"""

PRIMS = """
<mujoco>
  <visual><headlight ambient="0 0 0"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9" ambient="0 0 0"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body pos="0 0 0.3"><joint type="free"/><geom name="b1" type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="0.15 0.1 0.2"><joint type="free"/><geom name="s1" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/></body>
    <body pos="-0.15 -0.1 0.25"><joint type="free"/><geom name="c1" type="cylinder" size="0.04 0.08" rgba="0.2 0.2 0.9 1"/></body>
    <body pos="0.05 -0.2 0.2"><joint type="free"/><geom name="k1" type="capsule" size="0.03 0.06" rgba="0.9 0.9 0.2 1"/></body>
  </worldbody>
</mujoco>
"""


def _fwd(model, seed=None):
    d = mujoco.MjData(model)
    if seed is not None:
        d.qpos[:] += np.random.default_rng(seed).uniform(-0.05, 0.05, model.nq)
    mujoco.mj_forward(model, d)
    return d


def _psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def test_lambertian_plane_analytic():
    """Radiance of a Lambertian plane (albedo 0.8) under a directional light of MuJoCo diffuse 0.5
    at normal incidence is albedo/pi * E, with E = pi * 0.5 (the stack's radiometric convention,
    shared with tier 0): 0.4 exactly, independent of bounce count (the sky is black)."""
    model = mujoco.MjModel.from_xml_string(LAMBERT)
    for bounces in (0, 3):
        rend = Tier2Renderer(model, 1, width=64, height=64, camera="cam", spp=16, max_bounces=bounces)
        out = rend.render_host([_fwd(model)], passes=4)
        hdr = out["hdr"][0]
        assert np.isfinite(hdr).all()
        print(f"bounces {bounces}: mean {hdr.mean():.4f} (expected 0.4000), pixel std {hdr.std():.4f}")
        assert abs(hdr.mean() - 0.4) < 0.01
        assert hdr.std() < 0.01          # unshadowed, unoccluded: no noise sources
    assert out["rgb"][0].mean() == pytest.approx(0.4 * 255, abs=2)
    assert np.allclose(out["depth"][0], 2.0, atol=1e-3)
    assert (out["seg"][0] > 0).all()


def test_white_furnace_uniform_sky():
    """A white Lambertian plane under a uniform sky of radiance S reflects exactly S (energy
    conservation of the cosine-sampled diffuse lobe and its pdf)."""
    model = mujoco.MjModel.from_xml_string(FURNACE)
    rend = Tier2Renderer(model, 1, width=64, height=64, camera="cam", spp=32, max_bounces=1)
    out = rend.render_host([_fwd(model)], passes=4)
    hdr = out["hdr"][0]
    print(f"furnace: mean {hdr.mean():.4f} (expected 0.5000), pixel std {hdr.std():.4f}")
    assert abs(hdr.mean() - 0.5) < 0.01
    assert hdr.std() < 0.05


ENV_PLANE = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset><material name="white" rgba="1 1 1 1" specular="0" shininess="0"/></asset>
  <worldbody>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="30"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="white"/>
  </worldbody>
</mujoco>
"""


def _env_irradiance_radiance(env):
    """Outgoing radiance of a white Lambertian, upward-facing plane under an equirectangular map (pole +z, v =
    polar angle / pi): (1/pi) * sum over upper-hemisphere pixels of L * cos(theta) * d_omega."""
    h, w = env.shape[:2]
    th = (np.arange(h) + 0.5) / h * np.pi
    d_omega = (2 * np.pi / w) * (np.pi / h) * np.sin(th)
    cos_t = np.clip(np.cos(th), 0, None)
    return (env * (cos_t * d_omega)[:, None, None]).sum((0, 1)) / np.pi


def test_env_map_furnace():
    """A constant environment map of radiance S behaves like the uniform sky: a white plane reflects S
    (the env NEE + BRDF-sampled miss with MIS weights sum to one estimator of the same integral)."""
    model = mujoco.MjModel.from_xml_string(ENV_PLANE)
    rend = Tier2Renderer(model, 1, width=64, height=64, camera="cam", spp=32, max_bounces=1)
    rend.set_environment(np.full((64, 128, 3), 0.5, np.float32))
    hdr = rend.render_host([_fwd(model)], passes=4)["hdr"][0]
    print(f"env furnace: mean {hdr.mean():.4f} (expected 0.5000), pixel std {hdr.std():.4f}")
    assert abs(hdr.mean() - 0.5) < 0.01
    # the map's USD light scale: intensity x 2^exposure x color
    rend.set_environment_pose(0.0, intensity=2.0, exposure=1.0, color=(1.0, 0.5, 0.25))
    hdr = rend.render_host([_fwd(model)], passes=4)["hdr"][0]
    assert np.allclose(hdr.reshape(-1, 3).mean(0), 0.5 * 4.0 * np.array([1.0, 0.5, 0.25]), rtol=0.02)
    # back to the sky colour
    rend.set_environment(None)
    hdr = rend.render_host([_fwd(model)], passes=1)["hdr"][0]
    assert hdr.mean() < 1e-6


def test_env_map_importance_sampling_unbiased():
    """Under a map with a small, very bright patch (a window: 0.3 % of the sphere carrying most of the
    energy) the MIS estimate (env NEE + BRDF sampling) matches the analytic plane radiance at any yaw,
    and has several times less noise than the same estimator with a uniform sampling table."""
    from metalsim.render.tier2 import env_sampling_table
    model = mujoco.MjModel.from_xml_string(ENV_PLANE)
    env = np.full((128, 256, 3), 0.05, np.float32)
    env[36:44, 40:52] = (60.0, 50.0, 40.0)                      # ~45 deg elevation
    expected = _env_irradiance_radiance(env)
    for yaw in (0.0, 1.3):
        rend = Tier2Renderer(model, 1, width=48, height=48, camera="cam", spp=64, max_bounces=1, seed=3)
        rend.set_environment(env, yaw=yaw)
        hdr = rend.render_host([_fwd(model)], passes=4)["hdr"][0]
        mean = hdr.reshape(-1, 3).mean(0)
        rel_noise = float(hdr[..., 1].std() / hdr[..., 1].mean())
        print(f"yaw {yaw}: mean {np.round(mean, 4)} expected {np.round(expected, 4)}, per-pixel rel. std {rel_noise:.3f}")
        assert np.allclose(mean, expected, rtol=0.02)
    # same map, sampling table of a constant map: still unbiased, much noisier
    flat = Tier2Renderer(model, 1, width=48, height=48, camera="cam", spp=64, max_bounces=1, seed=3)
    flat.set_environment(env)
    cdf, _, _ = env_sampling_table(np.ones_like(env))
    flat._env_cdf = flat.ctx.buffer(cdf.nbytes, cdf, "pt_env_cdf_uniform")
    hdr_u = flat.render_host([_fwd(model)], passes=4)["hdr"][0]
    rel_u = float(hdr_u[..., 1].std() / hdr_u[..., 1].mean())
    print(f"uniform table: mean {np.round(hdr_u.reshape(-1, 3).mean(0), 4)}, rel. std {rel_u:.3f} ({rel_u / rel_noise:.1f}x)")
    assert np.allclose(hdr_u.reshape(-1, 3).mean(0), expected, rtol=0.05)
    assert rel_u > 3 * rel_noise



def _rot_x(deg):
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0, 0], [0, c, -s_], [0, s_, c]])


def _env_irradiance_radiance_rotated(env, R):
    """As _env_irradiance_radiance, for the map oriented by R (dome frame -> world): every texel's world direction is
    R @ env_direction(u, v); the plane (normal +z) sees the texels with (R d).z > 0."""
    from metalsim.render.hdr import env_direction
    h, w = env.shape[:2]
    u, v = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    d = env_direction(u, v) @ np.asarray(R).T
    d_omega = (2 * np.pi / w) * (np.pi / h) * np.sin(v * np.pi)
    return (env * (np.clip(d[..., 2], 0, None) * d_omega)[..., None]).sum((0, 1)) / np.pi


SKY_CAM = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="0.7071 -0.7071 0 -0.4082 -0.4082 0.8165" fovy="100"/>
    <geom name="speck" type="sphere" size="0.001" pos="-30 -30 -30"/>
  </worldbody>
</mujoco>
"""


def test_env_map_tilted_dome_lookup():
    """A tilted dome (``set_environment(rotation=)``): the shader looks a world direction d up at R^T d, R = rotation @
    R_z(yaw). The map stores its own dome-frame direction (+2, as radiance), so a camera seeing only sky reads
    R^T d + 2 per pixel, and the un-rotated render of the same map reads d + 2 (the shader's own camera rays): the
    tilted render must equal R^T (untilted - 2) + 2 for 30 deg about x, a general rotation (with a yaw), a
    quaternion, and a y-up pole alignment. The yaw-only path stays what it was (identity rotation = yaw path)."""
    from metalsim.render.hdr import env_direction
    from metalsim.render.tier2 import dome_rotation
    model = mujoco.MjModel.from_xml_string(SKY_CAM)
    h, w = 512, 1024
    u, v = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    enc = (env_direction(u, v) + 2.0).astype(np.float32)
    rend = Tier2Renderer(model, 1, width=48, height=48, camera="cam", spp=1, max_bounces=1, center_sample=True, seed=1)
    rend.set_environment(enc)
    base = rend.render_host([_fwd(model)], passes=1)["hdr"][0].reshape(-1, 3).astype(np.float64)
    d = base - 2.0
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=5e-3)          # the shader saw sky directions
    assert d[:, 2].min() < -0.2 and d[:, 2].max() > 0.9                       # a view where a tilt matters
    q = np.array([np.cos(0.6), *(np.sin(0.6) * np.array([0.3, -0.8, 0.52]) / np.linalg.norm([0.3, -0.8, 0.52]))])
    cases = [("rotX30", _rot_x(30), 0.0), ("rotX30+yaw", _rot_x(30), 0.7), ("quat", q, -1.1),
             ("y-up pole (Rx -90) + Kit azimuth", _rot_x(-90), -np.pi / 2)]
    for name, rot, yaw in cases:
        rend.set_environment(enc, rotation=rot, yaw=yaw)
        assert rend.env_flags == 32 | 512
        R = dome_rotation(rot, yaw)
        got = rend.render_host([_fwd(model)], passes=1)["hdr"][0].reshape(-1, 3) - 2.0
        exp = d @ R                                                           # R^T d per row
        err = np.abs(got - exp).max()
        wrong = np.abs(got - d @ dome_rotation(None, yaw)).max()              # what the yaw-only lookup would give
        print(f"{name}: max |lookup - R^T d| {err:.4f} (yaw-only would be off by {wrong:.3f})")
        assert err < 0.01 and wrong > 0.2
    # back to the yaw-only path: the flag clears and the render is that of the untilted map
    rend.set_environment(enc)
    assert rend.env_flags == 32
    again = rend.render_host([_fwd(model)], passes=1)["hdr"][0].reshape(-1, 3)
    assert np.array_equal(again, base.astype(np.float32))


def test_env_map_tilted_importance_sampling():
    """Importance sampling of a tilted dome uses the same rotation as the lookup: under a bright-window map turned
    50 deg about x (and yawed), the MIS estimate (env NEE + BRDF sampling) matches the plane radiance integrated over
    the rotated map, agrees with the uniform-table estimate within noise, and a constant tilted map is a furnace."""
    from metalsim.render.tier2 import env_sampling_table, dome_rotation
    model = mujoco.MjModel.from_xml_string(ENV_PLANE)
    rot, yaw = _rot_x(50), 0.9
    rend = Tier2Renderer(model, 1, width=48, height=48, camera="cam", spp=32, max_bounces=1, seed=5)
    rend.set_environment(np.full((64, 128, 3), 0.5, np.float32), rotation=rot, yaw=yaw)
    hdr = rend.render_host([_fwd(model)], passes=4)["hdr"][0]
    print(f"tilted furnace: mean {hdr.mean():.4f} (expected 0.5)")
    assert abs(hdr.mean() - 0.5) < 0.01
    env = np.full((128, 256, 3), 0.05, np.float32)
    env[20:28, 40:52] = (60.0, 50.0, 40.0)        # ~65 deg elevation in the dome frame; ~15 deg + yawed in the world
    expected = _env_irradiance_radiance_rotated(env, dome_rotation(rot, yaw))
    upright = _env_irradiance_radiance(env)
    rend = Tier2Renderer(model, 1, width=48, height=48, camera="cam", spp=64, max_bounces=1, seed=3)
    rend.set_environment(env, rotation=rot, yaw=yaw)
    hdr = rend.render_host([_fwd(model)], passes=4)["hdr"][0]
    mean = hdr.reshape(-1, 3).mean(0); rel = float(hdr[..., 1].std() / hdr[..., 1].mean())
    flat = Tier2Renderer(model, 1, width=48, height=48, camera="cam", spp=64, max_bounces=1, seed=3)
    flat.set_environment(env, rotation=rot, yaw=yaw)
    cdf, _, _ = env_sampling_table(np.ones_like(env))
    flat._env_cdf = flat.ctx.buffer(cdf.nbytes, cdf, "pt_env_cdf_uniform")
    hdr_u = flat.render_host([_fwd(model)], passes=4)["hdr"][0]
    mean_u = hdr_u.reshape(-1, 3).mean(0); rel_u = float(hdr_u[..., 1].std() / hdr_u[..., 1].mean())
    # standard error of the image means (pixels independent): rel. std / sqrt(pixels)
    se = np.sqrt((rel ** 2 + rel_u ** 2) / hdr[..., 1].size) * expected[1]
    print(f"tilted window: importance {np.round(mean, 4)} (rel. std {rel:.3f}), uniform {np.round(mean_u, 4)} "
          f"(rel. std {rel_u:.3f}), expected {np.round(expected, 4)}; upright map would give {np.round(upright, 4)}; "
          f"|IS - uniform| {abs(mean[1] - mean_u[1]):.4f} vs 4 SE {4 * se:.4f}")
    assert not np.allclose(expected, upright, rtol=0.1)                      # the tilt changes the answer
    assert np.allclose(mean, expected, rtol=0.02)
    assert np.allclose(mean_u, expected, rtol=0.05)
    assert abs(mean[1] - mean_u[1]) < 4 * se
    assert rel_u > 2 * rel

def _write_rgbe(path, rgb):
    """Minimal Radiance RGBE writer (flat scanlines) for tests."""
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


def test_env_map_usd_dome_import(tmp_path):
    """A USD DomeLight with inputs:texture:file flows through the importer (custom text ``usd_dome``) to
    set_environment with USD's light scale (intensity x 2^exposure x color) and the prim's yaw; a furnace under
    a constant textured dome reads S x that scale, and the RGBE reader reproduces the written values."""
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics
    from metalsim.render.hdr import load_hdr
    from metalsim.scene.usd_to_mjcf import load_usd
    hdr_path = str(tmp_path / "const.hdr")
    _write_rgbe(hdr_path, np.full((8, 16, 3), 0.5, np.float32))
    assert np.allclose(load_hdr(hdr_path), 0.5, rtol=0.01)
    stage = Usd.Stage.CreateNew(str(tmp_path / "dome.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    dome = UsdLux.DomeLight.Define(stage, "/World/skyLight")
    dome.CreateTextureFileAttr("./const.hdr"); dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(2.0); dome.CreateExposureAttr(1.0); dome.CreateColorAttr((1.0, 0.5, 0.25))
    UsdGeom.Xformable(dome.GetPrim()).AddRotateZOp().Set(30.0)
    plane = UsdGeom.Cube.Define(stage, "/World/floor"); plane.CreateSizeAttr(1.0)
    UsdGeom.Xformable(plane.GetPrim()).AddScaleOp().Set((100.0, 100.0, 0.02))
    UsdGeom.Xformable(plane.GetPrim()).AddTranslateOp().Set((0.0, 0.0, -0.5))
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set((0.0, 0.0, 2.0))   # a USD camera looks down its -z: straight down
    stage.GetRootLayer().Save()
    spec = load_usd(str(tmp_path / "dome.usda"))
    spec.visual.headlight.active = 0
    mat = spec.add_material(); mat.name = "matte"; mat.rgba = [0.8, 0.8, 0.8, 1.0]; mat.specular = 0.0; mat.shininess = 0.0
    for g in spec.geoms:
        g.material = "matte"
    m = spec.compile()
    assert m.ntext == 1 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXT, 0) == "usd_dome"
    rend = Tier2Renderer(m, 1, width=32, height=32, camera="cam", spp=32, max_bounces=1, headlight=False)
    rec = rend.set_environment_from_model(m)
    print("usd dome record:", rec)
    assert os.path.realpath(rec["file"]) == os.path.realpath(hdr_path) and rec["format"] == "latlong"
    assert abs(rec["yaw"] - np.radians(30)) < 1e-6 and rec["tilt"] < 0.01
    # The shader's yaw is in the OpenUSD DomeLight_1 frame (z-up: image centre toward -y, u = 0.25 toward +x); Kit/RTX
    # renders a dome 90 deg clockwise of that (centre toward -x; measured 2026-09-26, PARITY 1.7), so a USD stage's dome
    # gets KIT_DOME_AZIMUTH added (this test's original expectation of the bare 30 deg predates that measurement).
    from metalsim.render.hdr import env_direction
    from metalsim.render.tier2 import KIT_DOME_AZIMUTH
    env_yaw = float(rend.consts.view(np.float32)[20])
    assert rend.env_flags == 32 and np.isclose(env_yaw, np.radians(30) + KIT_DOME_AZIMUTH, atol=1e-6)
    c, s_ = np.cos(np.radians(30)), np.sin(np.radians(30))
    # Kit's measured mapping, u = -(atan2(y, x) - yaw) / 2pi: the image centre faces -x, u = 0 faces +x, u = 0.25 faces
    # -y, each turned by the prim's rotateZ (+30 deg about +z)
    assert np.allclose(env_direction(0.5, 0.5, env_yaw), (-c, -s_, 0.0), atol=1e-6)
    assert np.allclose(env_direction(0.0, 0.5, env_yaw), (c, s_, 0.0), atol=1e-6)
    assert np.allclose(env_direction(0.25, 0.5, env_yaw), (s_, -c, 0.0), atol=1e-6)
    assert np.allclose(rend.consts.view(np.float32)[24:27], 2.0 * 2.0 * np.array([1.0, 0.5, 0.25]))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    hdr = rend.render_host([d], passes=2)["hdr"][0]
    top = hdr.reshape(-1, 3).mean(0)
    albedo = m.mat_rgba[m.geom_matid[0], :3]                     # the bound material's colour (0.8)
    print("usd dome furnace:", top, "expected", 0.5 * 4.0 * np.array([1.0, 0.5, 0.25]) * albedo)
    assert np.allclose(top, 0.5 * 4.0 * np.array([1.0, 0.5, 0.25]) * albedo, rtol=0.03)



@pytest.mark.parametrize("case", ["rotX30", "yup_domelight1_scene", "yup_legacy_rotX-90", "zup_domelight1_poleY"])
def test_env_map_usd_tilted_dome_import(tmp_path, case):
    """The importer records a tilted dome's full rotation (``rot`` = R_prim . P) and tier 2 uploads
    R_prim . P . R_z(KIT_DOME_AZIMUTH) (research_notes/dome_light_azimuth/tilted_dome.md): a legacy DomeLight with
    rotateX 30 deg (Kit applies it rigidly, measured); a DomeLight_1 with poleAxis "scene" on a y-up stage (pole +y);
    a legacy DomeLight on a y-up stage with the rotateX -90 a y-up author adds (Kit renders legacy domes z-up); a
    DomeLight_1 poleAxis "Y" on a z-up stage. World directions of map points match the analytic rotation."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdLux
    from metalsim.render.hdr import env_direction
    from metalsim.render.tier2 import KIT_DOME_AZIMUTH, dome_rotation
    from metalsim.scene.usd_to_mjcf import load_usd
    _write_rgbe(str(tmp_path / "const.hdr"), np.full((8, 16, 3), 0.5, np.float32))
    stage = Usd.Stage.CreateNew(str(tmp_path / "dome.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if case.startswith("yup") else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    dome = (UsdLux.DomeLight_1 if "domelight1" in case else UsdLux.DomeLight).Define(stage, "/World/skyLight")
    dome.CreateTextureFileAttr("./const.hdr"); dome.CreateTextureFormatAttr("latlong")
    if case == "rotX30":
        UsdGeom.Xformable(dome.GetPrim()).AddRotateXOp().Set(30.0)
    elif case == "yup_legacy_rotX-90":
        UsdGeom.Xformable(dome.GetPrim()).AddRotateXOp().Set(-90.0)
    elif case == "zup_domelight1_poleY":
        dome.CreatePoleAxisAttr("Y")
    UsdGeom.Cube.Define(stage, "/World/box").CreateSizeAttr(0.1)
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set((0.0, 0.0, 2.0))
    stage.GetRootLayer().Save()
    m = load_usd(str(tmp_path / "dome.usda")).compile()
    rend = Tier2Renderer(m, 1, width=8, height=8, camera="cam", spp=1, max_bounces=1, headlight=False)
    rec = rend.set_environment_from_model(m)
    print(case, "record:", {k: rec[k] for k in ("yaw", "tilt", "rot", "pole_axis") if k in rec})
    R_prim_P = {"rotX30": _rot_x(30), "yup_domelight1_scene": _rot_x(-90), "yup_legacy_rotX-90": _rot_x(-90),
                "zup_domelight1_poleY": _rot_x(-90)}[case]
    assert np.allclose(np.reshape(rec["rot"], (3, 3)), R_prim_P, atol=1e-6)
    assert rend.env_flags == 32 | 512
    R = R_prim_P @ dome_rotation(None, KIT_DOME_AZIMUTH)
    up = rend.consts.view(np.float32)[76:88].reshape(3, 4)[:, :3].T          # float4 columns -> R
    assert np.allclose(up, R, atol=1e-6)
    # pole, the image centre and u = 0.25 in the world: Kit's z-up frame (pole +z, centre -x, u = 0.25 toward -y) turned rigidly
    pts = {(0.5, 0.0): (0, 0, 1.0), (0.5, 0.5): (-1.0, 0, 0), (0.25, 0.5): (0, -1.0, 0)}
    for (u, v), dk in pts.items():
        assert np.allclose(env_direction(u, v, KIT_DOME_AZIMUTH, rotation=R_prim_P), R_prim_P @ np.array(dk), atol=1e-9)
    if case.startswith("yup") or case == "zup_domelight1_poleY":
        assert np.allclose(env_direction(0.5, 0.0, KIT_DOME_AZIMUTH, rotation=R_prim_P), (0, 1, 0), atol=1e-9)   # pole +y
    else:
        c, s_ = np.cos(np.radians(30)), np.sin(np.radians(30))
        assert np.allclose(env_direction(0.5, 0.0, KIT_DOME_AZIMUTH, rotation=R_prim_P), (0, -s_, c), atol=1e-9)
        assert abs(rec["tilt"] - 30.0) < 1e-4

def test_env_randomizer_and_helpers():
    """Replicator: ``Randomizer.environment`` picks a map per episode by key (cached GPU resources, random yaw);
    the helpers set_fovy / set_materials rewrite what they claim; the opt-in firefly clamp caps a path's
    luminance (biased by construction: a 0.5 furnace under a 0.2 cap reads 0.2)."""
    from metalsim.replicator import Randomizer
    model = mujoco.MjModel.from_xml_string(ENV_PLANE)
    rend = Tier2Renderer(model, 2, width=32, height=32, camera="cam", spp=8, max_bounces=1)
    maps = {"a": np.full((16, 32, 3), 0.5, np.float32), "b": np.full((16, 32, 3), (0.1, 0.2, 0.3), np.float32)}
    rnd = Randomizer(rend, seed=1)
    keys = [rnd.environment(maps) for _ in range(12)]
    assert set(keys) == {"a", "b"} and set(rend._env_res) == {"a", "b"} and rend.env_flags == 32
    assert rnd.environment(maps, key="b") == "b"
    datas = [_fwd(model), _fwd(model)]
    hdr = rend.render_host(datas, passes=2)["hdr"]
    assert np.allclose(hdr.reshape(-1, 3).mean(0), (0.1, 0.2, 0.3), rtol=0.03)
    assert rnd.environment(maps, key="a") == "a"          # cached: no hdr needed
    rend.set_firefly_clamp(0.2)
    hdr = rend.render_host(datas, passes=2)["hdr"]
    assert abs(hdr.mean() - 0.2) < 0.01
    rend.set_firefly_clamp(0.0)
    assert abs(rend.render_host(datas, passes=2)["hdr"].mean() - 0.5) < 0.01
    rend.set_fovy(60.0, env=1)
    f = 0.5 * 32 / np.tan(np.radians(30))
    assert np.isclose(rend.cam_spec[1, 0], f) and np.isclose(rend.t_cam_spec[1, 1].item(), f) and rend.cam_spec[0, 0] != rend.cam_spec[1, 0]
    mats = rend.tables.materials.reshape(rend.G, 16).copy(); mats[:, :3] = 0.25
    rend.set_materials(mats)
    assert abs(rend.render_host(datas, passes=2)["hdr"].mean() - 0.125) < 0.005


def test_monte_carlo_convergence():
    """Per-pixel noise between two independent renders falls as 1/sqrt(spp) in an indirectly-lit
    region (a box on the plane under the sky, 2 bounces)."""
    model = mujoco.MjModel.from_xml_string(PRIMS)
    d = _fwd(model)
    def noise(spp, seed):
        a = Tier2Renderer(model, 1, width=96, height=96, camera="cam", spp=spp, max_bounces=2, seed=seed).render_host([d])["hdr"][0]
        b = Tier2Renderer(model, 1, width=96, height=96, camera="cam", spp=spp, max_bounces=2, seed=seed + 100).render_host([d])["hdr"][0]
        return float(np.sqrt(np.mean((a - b) ** 2)))
    n4, n64 = noise(4, 1), noise(64, 2)
    print(f"noise rms: spp 4 -> {n4:.4f}, spp 64 -> {n64:.4f}, ratio {n4 / n64:.2f} (ideal 4.0)")
    assert n4 / n64 > 2.5


def test_direct_light_matches_tier0():
    """With bounces disabled the path tracer computes the same direct light + headlight + shadows
    that tier 0 rasterizes (same BRDF, same light convention): PSNR and identical depth/seg."""
    model = mujoco.MjModel.from_xml_string(PRIMS)
    t0 = Tier0Renderer(model, 1, width=192, height=192, camera="cam", outputs=("rgb", "depth", "seg"), include_planes=True)
    t2 = Tier2Renderer(model, 1, width=192, height=192, camera="cam", spp=16, max_bounces=0)
    psnrs = []
    for seed in range(3):
        d = _fwd(model, seed)
        a = t0.render_host([d]); b = t2.render_host([d], passes=4)
        psnrs.append(_psnr(a["rgb"][0], b["rgb"][0]))
        both = (a["seg"][0] > 0) & (b["seg"][0] > 0)
        dmed = np.median(np.abs(a["depth"][0][both] - b["depth"][0][both]))
        seg_iou = ((a["seg"][0] > 0) & (b["seg"][0] > 0)).sum() / ((a["seg"][0] > 0) | (b["seg"][0] > 0)).sum()
        assert dmed < 1e-3 and seg_iou > 0.97
    print("tier 2 (direct only) vs tier 0 PSNR:", np.round(psnrs, 2), "depth median", dmed, "seg IoU", round(seg_iou, 4))
    os.makedirs(OUT, exist_ok=True)
    try:
        import imageio.v2 as iio
        full = Tier2Renderer(model, 1, width=192, height=192, camera="cam", spp=64, max_bounces=4).render_host([d], passes=8)["rgb"][0]
        iio.imwrite(os.path.join(OUT, "tier2_vs_tier0.png"), np.concatenate([a["rgb"][0], b["rgb"][0], full], axis=1))
    except ImportError:
        pass
    assert min(psnrs) > 22.0


def test_gpu_path_from_batchsim_matches_host():
    from metalsim.physics.batch import BatchSim
    model = mujoco.MjModel.from_xml_string(PRIMS)
    n = 4
    sim = BatchSim(model, n)
    sim.synchronize()
    rend = Tier2Renderer(model, n, width=64, height=64, camera="cam", spp=8, max_bounces=1, seed=3)
    for _ in range(20):
        v = sim.step()
    vr = rend.render(sim, v); rend.wait_sim_after_render(sim, vr); rend.after(vr)
    gpu = rend.out.rgb.clone().cpu().numpy()
    sim.synchronize(); rend.ctx.synchronize()
    datas = []
    for e in range(n):
        d = mujoco.MjData(model); d.qpos[:] = sim.d.qpos.numpy()[e]; mujoco.mj_forward(model, d); datas.append(d)
    rend2 = Tier2Renderer(model, n, width=64, height=64, camera="cam", spp=8, max_bounces=1, seed=3)
    host = rend2.render_host(datas)["rgb"]
    assert rend.out.rgb.device.type == "mps" and gpu.shape == (n, 64, 64, 3)
    assert np.mean(np.abs(gpu.astype(int) - host.astype(int))) < 1.0   # same seeds, same state: same paths


def test_cartpole_rgb_rollout_tier2():
    """A full env rollout (physics + path-traced 100x100 observations + reward/reset) at tier 2."""
    from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
    env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=32, tier=2, spp=2, max_bounces=2))
    obs = env.reset()
    a = torch.zeros(32, 1, device="mps")
    for _ in range(30):
        obs, r, done, info = env.step(torch.rand_like(a) * 2 - 1)
    env.synchronize()
    img = obs["image"].cpu().numpy()
    assert img.shape == (32, 100, 100, 3) and img.std() > 10 and torch.isfinite(r).all()


def test_usd_units_lambertian_sun_disk_and_dome():
    """USD / RTX mode: a DistantLight of intensity I (0.53 deg disk) gives irradiance pi*I at normal incidence,
    so a Lambertian plane (OmniPBR with specular_level 0, i.e. pure Lambert) reads albedo * I; a textureless
    DomeLight of intensity S*colour is a uniform environment of that radiance (white furnace)."""
    model = mujoco.MjModel.from_xml_string(LAMBERT)
    rend = Tier2Renderer(model, 1, width=64, height=64, camera="cam", spp=16, max_bounces=0, material_model="omnipbr",
                         usd_lights=[{"intensity": 3000.0, "color": (1, 1, 1), "angle_deg": 0.53}], headlight=False)
    hdr = rend.render_host([_fwd(model)], passes=4)["hdr"][0]
    print(f"usd sun: mean {hdr.mean():.2f} (expected {0.8 * 3000:.0f})")
    assert abs(hdr.mean() / (0.8 * 3000) - 1) < 0.01
    fm = mujoco.MjModel.from_xml_string(FURNACE)
    rend = Tier2Renderer(fm, 1, width=64, height=64, camera="cam", spp=32, max_bounces=1, material_model="omnipbr",
                         dome=(400.0, (0.75, 0.8, 0.9)), headlight=False)
    hdr = rend.render_host([_fwd(fm)], passes=4)["hdr"][0]
    exp = 400 * np.array([0.75, 0.8, 0.9])
    print("usd dome furnace:", hdr.reshape(-1, 3).mean(0), "expected", exp)
    assert np.allclose(hdr.reshape(-1, 3).mean(0) / exp, 1, atol=0.02)


def test_rtx_tonemap_and_denoisers_on_furnace():
    """RTX display transform: 8-bit output = sRGB(ACES(exposure * radiance)); both denoisers keep the mean of a
    furnace image (white plane under a uniform dome) and reduce its Monte Carlo noise."""
    from metalsim.render.tier2 import rtx_exposure
    fm = mujoco.MjModel.from_xml_string(FURNACE)
    k = rtx_exposure()
    tm = lambda L: 255 * (1.055 * np.clip(L * k * (2.51 * L * k + 0.03) / (L * k * (2.43 * L * k + 0.59) + 0.14), 0, 1) ** (1 / 2.4) - 0.055)
    ran = 0
    for dn in (None, "atrous", "oidn"):
        try:
            rend = Tier2Renderer(fm, 2, width=48, height=48, camera="cam", spp=4, max_bounces=1, dome=(400.0, (0.8, 0.8, 0.8)),
                                 headlight=False, tonemap="rtx", exposure=k, denoise=dn)
        except ImportError:
            continue
        out = rend.render_host([_fwd(fm), _fwd(fm)], passes=1)
        L = out["hdr"]
        if dn is None:
            assert abs(out["rgb"].mean() - tm(L).mean()) < 1.5
            continue
        ran += 1
        noisy, clean = out["hdr_mean"], out["hdr_denoised"]
        print(f"{dn}: mean {noisy.mean():.1f} -> {clean.mean():.1f}, std {noisy.std():.1f} -> {clean.std():.1f}")
        assert abs(clean.mean() / noisy.mean() - 1) < 0.02 and clean.std() < 0.5 * noisy.std()
        assert abs(out["rgb"].mean() - tm(clean).mean()) < 1.5
    assert ran >= 1


def test_physical_preset_batch_is_finite():
    """1024 camera-RL frames under the physical tier-2 preset (the Cartpole-RGB default): every accumulated radiance
    value is finite and the observation is not degenerate."""
    from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
    env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=1024, tier=2, render_mode="physical"))
    obs = env.reset()
    for _ in range(10):
        obs, r, done, info = env.step(torch.rand(1024, 1, device="mps") * 2 - 1)
    env.synchronize()
    acc = env.rend._accum.numpy()
    assert np.isfinite(acc).all() and (acc[..., 3] > 0).all()
    img = obs["image"].float()
    assert torch.isfinite(img).all() and 40 < img.mean().item() < 200 and img.std().item() > 10
