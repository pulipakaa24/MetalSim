"""True area lights in tier 2 (metalsim/render/area_lights.py, pathtrace.metal AREA_LIGHTS): USD Rect / Disk / Cylinder /
Sphere lights imported through metalsim.scene.usd_to_mjcf, rendered onto a white Lambertian floor with direct light
only, against the analytic irradiance (Lambert's polygon formula, the sphere's pi L (R/d)^2 cos, quadrature for the
cylinder) at several distances; the light units are UsdLux's (intensity x 2^exposure x color, / area with normalize)."""
import numpy as np
import pytest
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

RHO = 0.8          # floor albedo


def light_stage(path, shape, height, size=0.4, scale=1.0):
    """One USD light of characteristic size ``size`` at (0, 0, height) facing down (-Z: the floor). Returns the stage
    path and the analytic spec (radiance per UsdLux)."""
    from pxr import Gf, Usd, UsdGeom, UsdLux
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    xf = None
    if shape == "rect":
        lt = UsdLux.RectLight.Define(stage, "/World/light"); lt.CreateWidthAttr(size); lt.CreateHeightAttr(0.5 * size)
        lt.CreateIntensityAttr(30.0); lt.CreateExposureAttr(1.0); L = 60.0
        xf = UsdGeom.Xformable(lt.GetPrim())
        xf.AddTranslateOp().Set((0.0, 0.0, height)); xf.AddRotateZOp().Set(30.0)        # rotated in plane: the X axis
        spec = dict(shape="rect", w=size, h=0.5 * size, rot=np.radians(30.0))
    elif shape == "disk":
        lt = UsdLux.DiskLight.Define(stage, "/World/light"); lt.CreateRadiusAttr(size / 2)
        lt.CreateIntensityAttr(60.0 * np.pi * (size / 2) ** 2); lt.CreateNormalizeAttr(True); L = 60.0
        UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set((0.0, 0.0, height))
        spec = dict(shape="disk", r=size / 2)
    elif shape == "cylinder":
        lt = UsdLux.CylinderLight.Define(stage, "/World/light"); lt.CreateRadiusAttr(size / 8); lt.CreateLengthAttr(size)
        lt.CreateIntensityAttr(60.0); L = 60.0
        UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set((0.0, 0.0, height))
        spec = dict(shape="cylinder", r=size / 8, len=size)
    else:
        lt = UsdLux.SphereLight.Define(stage, "/World/light"); lt.CreateRadiusAttr(size / 2)
        lt.CreateIntensityAttr(15.0); lt.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0)); L = 15.0
        UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set((0.0, 0.0, height))
        spec = dict(shape="sphere", r=size / 2)
    lt.CreateColorAttr().Set(Gf.Vec3f(scale, scale, scale)); L *= scale          # color multiplies the radiance
    stage.GetRootLayer().Save()
    spec.update(L=L, c=np.array([0.0, 0.0, height]), size=size)
    return spec


def exact_irradiance(s, p):
    """Irradiance at floor point p (normal +z) from the analytic light ``s``."""
    from metalsim.render.area_lights import disk_vertices, polygon_irradiance, rect_vertices
    n = np.array([0.0, 0.0, 1.0]); c = s["c"]
    if s["shape"] == "rect":
        x = np.array([np.cos(s["rot"]), np.sin(s["rot"]), 0.0]); y = np.array([-np.sin(s["rot"]), np.cos(s["rot"]), 0.0])
        return polygon_irradiance(s["L"], rect_vertices(c, x, y, s["w"], s["h"]), p, n)
    if s["shape"] == "disk":
        return polygon_irradiance(s["L"], disk_vertices(c, [1, 0, 0], [0, 1, 0], s["r"]), p, n)
    if s["shape"] == "sphere":
        v = c - p; d = np.linalg.norm(v)
        return np.pi * s["L"] * (s["r"] / d) ** 2 * v[2] / d
    k = 240                                    # cylinder along x, radiating from its side: midpoint quadrature
    xs = (np.arange(k) + 0.5) / k * s["len"] - s["len"] / 2; a = (np.arange(k) + 0.5) / k * 2 * np.pi
    X, A = np.meshgrid(xs, a)
    nrm = np.stack([np.zeros_like(X), np.cos(A), np.sin(A)], -1).reshape(-1, 3)
    S = c + np.stack([X, s["r"] * np.cos(A), s["r"] * np.sin(A)], -1).reshape(-1, 3)
    v = p - S; d2 = (v ** 2).sum(1); d = np.sqrt(d2)
    ce = np.clip((nrm * v).sum(1) / d, 0, None); cr = np.clip(-v[:, 2] / d, 0, None)
    return float((s["L"] * ce * cr / d2).sum() * s["len"] / k * s["r"] * 2 * np.pi / k)


def render_floor(stage_path, spec, res=24, half=None, spp=64, passes=64, area_lights=True):
    """Direct-light radiance of a white Lambertian floor under the light, seen straight down by a camera at the
    light's height (beside it is fine: the light is not geometry); returns (hdr red (res, res), floor points)."""
    import mujoco
    from metalsim.scene.usd_to_mjcf import load_usd
    from metalsim.render.tier2 import Tier2Renderer
    sp = load_usd(str(stage_path))
    mat = sp.add_material(); mat.name = "white"; mat.rgba = [RHO, RHO, RHO, 1]; mat.specular = 0.0; mat.reflectance = 0.0
    g = sp.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [0, 0, 0.05]; g.material = "white"
    H = float(spec["c"][2]) + 0.5
    half = half or max(spec["size"], spec["c"][2]) * 1.0
    fovy = float(np.degrees(2 * np.arctan(half / H)))
    cam = sp.worldbody.add_camera(); cam.name = "down"; cam.pos = [0, 0, H]; cam.fovy = fovy
    m = sp.compile()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = Tier2Renderer(m, 1, width=res, height=res, camera="down", spp=spp, max_bounces=0, headlight=False, skybox=False,
                      area_lights=area_lights)
    out = r.render_host([d], passes=passes)
    f = 0.5 * res / np.tan(np.radians(fovy) / 2)
    # floor points of a 3 x 3 grid inside each pixel (the samples are jittered over the pixel): (res, res, 9, 3)
    sub = (np.arange(3) + 0.5) / 3
    i = ((np.arange(res)[:, None] + sub[None, :]) - res / 2).reshape(-1) / f
    X, Y = np.meshgrid(i * H, -i * H)            # image x -> +x, image y (down) -> -y
    pts = np.stack([X, Y, np.zeros_like(X)], -1).reshape(res, 3, res, 3, 3).transpose(0, 2, 1, 3, 4).reshape(res, res, 9, 3)
    return out["hdr"][0, :, :, 0], pts


@pytest.mark.parametrize("shape", ["rect", "disk", "sphere", "cylinder"])
def test_area_light_irradiance(tmp_path, shape):
    """Floor radiance rho E / pi against the analytic irradiance, at 0.5, 1 and 3 light sizes: mean within 1 %, 4 x 4
    pixel blocks within 2 %; the far-field import (area_lights=False) is off by tens of percent near the light."""
    pytest.importorskip("pxr")
    for k, dist in enumerate((0.5, 1.0, 3.0)):
        spec = light_stage(tmp_path / f"{shape}{k}.usda", shape, dist * 0.4 + (0.2 if shape == "sphere" else 0.0))
        hdr, pts = render_floor(tmp_path / f"{shape}{k}.usda", spec)
        ref = np.array([[np.mean([exact_irradiance(spec, q) for q in px]) for px in row] for row in pts]) * RHO / np.pi
        b = lambda a: a.reshape(6, 4, 6, 4).mean(axis=(1, 3))
        mean_err = hdr.mean() / ref.mean() - 1
        blk = np.abs(b(hdr) / b(ref) - 1).max()
        print(f"{shape} at {dist} sizes: mean {100 * mean_err:+.2f} %, worst 4x4 block {100 * blk:.2f} %")
        assert abs(mean_err) < 0.01, (shape, dist, mean_err)
        assert blk < 0.02, (shape, dist, blk)


def test_area_light_far_field_reference(tmp_path):
    """The previous far-field import at one light size: the on-axis irradiance is too high (the documented +20-25 %),
    which the area sampling removes."""
    pytest.importorskip("pxr")
    spec = light_stage(tmp_path / "rect.usda", "rect", 0.4)
    hdr_ff, _ = render_floor(tmp_path / "rect.usda", spec, area_lights=False)
    hdr, _ = render_floor(tmp_path / "rect.usda", spec)
    ref = exact_irradiance(spec, np.zeros(3)) * RHO / np.pi
    c = slice(11, 13)
    ff, ar = hdr_ff[c, c].mean() / ref - 1, hdr[c, c].mean() / ref - 1
    print(f"rect at one size, on axis: far-field {100 * ff:+.1f} %, area {100 * ar:+.2f} %")
    assert ff > 0.1 and abs(ar) < 0.02


def render_floor_t1(stage_path, spec, res=24, occluder=None, rt_samples=16):
    """Tier 1 (raster + ray-traced shadows) of the same floor: linear rgb / 255 (headlight and ambient off)."""
    import mujoco
    from metalsim.scene.usd_to_mjcf import load_usd
    from metalsim.render.tier0 import Tier0Renderer
    sp = load_usd(str(stage_path))
    mat = sp.add_material(); mat.name = "white"; mat.rgba = [RHO, RHO, RHO, 1]; mat.specular = 0.0; mat.reflectance = 0.0
    g = sp.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [0, 0, 0.05]; g.material = "white"
    if occluder is not None:
        o = sp.worldbody.add_geom(); o.type = mujoco.mjtGeom.mjGEOM_BOX; o.pos = occluder[0]; o.size = occluder[1]
        o.material = "white"
    H = float(spec["c"][2]) + 0.5
    half = max(spec["size"], spec["c"][2])
    fovy = float(np.degrees(2 * np.arctan(half / H)))
    cam = sp.worldbody.add_camera(); cam.name = "down"; cam.pos = [0, 0, H]; cam.fovy = fovy
    sp.visual.headlight.active = 0; sp.visual.headlight.ambient = [0, 0, 0]; sp.visual.headlight.diffuse = [0, 0, 0]
    m = sp.compile()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = Tier0Renderer(m, 1, width=res, height=res, camera="down", tier=1, rt_samples=rt_samples, skybox=False)
    return r.render_host([d])["rgb"][0, :, :, 0].astype(np.float64) / 255.0, m, d


@pytest.mark.parametrize("shape", ["rect", "disk", "sphere"])
def test_tier1_area_light_irradiance(tmp_path, shape):
    """Tier 1: the unshadowed floor equals the closed-form irradiance (ratio estimator), 2 % (8-bit output)."""
    pytest.importorskip("pxr")
    spec = light_stage(tmp_path / f"{shape}.usda", shape, 0.4 + (0.2 if shape == "sphere" else 0.0),
                      scale={"rect": 0.1, "disk": 0.04, "sphere": 0.5}[shape])        # peak radiance 0.4-0.7
    t1, _, _ = render_floor_t1(tmp_path / f"{shape}.usda", spec)
    _, pts = render_floor(tmp_path / f"{shape}.usda", spec, spp=1, passes=1)
    ref = np.array([[exact_irradiance(spec, px[4]) for px in row] for row in pts]) * RHO / np.pi
    keep = ref > 0.1                               # away from the 8-bit floor
    err = np.abs(t1[keep] / ref[keep] - 1)
    print(f"tier 1 {shape}: median {100 * np.median(err):.2f} %, max {100 * err.max():.2f} % over {keep.sum()} px")
    assert ref.max() < 0.95 and keep.sum() > 100
    assert np.median(err) < 0.01 and err.max() < 0.03


def test_tier1_area_light_soft_shadow(tmp_path):
    """A box under a rect light casts a soft shadow in tier 1 that matches tier 2's (the path tracer's, validated
    above) within a few percent of the lit level, with a penumbra (intermediate pixels) where a point light has none."""
    import mujoco
    from metalsim.render.tier2 import Tier2Renderer
    pytest.importorskip("pxr")
    spec = light_stage(tmp_path / "rect.usda", "rect", 0.6, scale=0.02)
    occ = ((0.05, 0.0, 0.25), (0.06, 0.06, 0.02))
    t1, m, d = render_floor_t1(tmp_path / "rect.usda", spec, res=48, occluder=occ, rt_samples=64)
    r2 = Tier2Renderer(m, 1, width=48, height=48, camera="down", spp=64, max_bounces=0, headlight=False, skybox=False)
    t2 = r2.render_host([d], passes=32)["hdr"][0, :, :, 0]
    lit = np.percentile(t2, 99)
    diff = np.abs(t1 - t2) / lit
    pen = ((t2 > 0.1 * lit) & (t2 < 0.9 * lit)).sum()
    print(f"tier 1 vs tier 2 soft shadow: mean |diff| {100 * diff.mean():.2f} % of lit, p99 {100 * np.percentile(diff, 99):.1f} %, penumbra px {pen}")
    assert diff.mean() < 0.02 and np.percentile(diff, 99) < 0.12 and pen > 40


ROT_LIGHT = """
<mujoco>
  <option gravity="0 0 0"/>
  <asset><material name="white" rgba="0.8 0.8 0.8 1" specular="0" reflectance="0"/></asset>
  <visual><headlight active="0" ambient="0 0 0" diffuse="0 0 0"/></visual>
  <worldbody>
    <camera name="down" pos="0 0 1.2" fovy="70"/>
    <geom type="plane" size="0 0 0.05" material="white"/>
    <body name="lamp" pos="0 0 0.3"><joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0" group="3"/>
      <light name="strip" pos="0 0 0" dir="0 0 -1" type="spot" cutoff="90" exponent="1" diffuse="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>"""


def test_area_light_moving_body_gpu_path():
    """An area light on a rotating body: the emitter axes come from xmat per env on the GPU path (BatchSim) and match
    the host path; a 0.6 x 0.1 m strip turned by 0 / 90 deg lights the floor differently."""
    import json
    import mujoco
    from metalsim.physics.batch import BatchSim
    from metalsim.render.tier2 import Tier2Renderer
    spec = mujoco.MjSpec.from_string(ROT_LIGHT)
    rec = [{"light": "strip", "shape": "rect", "width": 0.6, "height": 0.1, "radiance": [2.0, 2.0, 2.0], "x_axis": [1, 0, 0]}]
    spec.add_text(name="usd_area_lights", data=json.dumps(rec))
    m = spec.compile()
    n = 4
    sim = BatchSim(m, n); sim.synchronize()
    q = np.radians([0.0, 30.0, 60.0, 90.0]).reshape(n, 1).astype(np.float32)
    sim.d.qpos.assign(q)
    v = sim.step(); sim.synchronize()
    r = Tier2Renderer(m, n, width=32, height=32, camera="down", spp=16, max_bounces=0, seed=5, skybox=False)
    vr = r.render(sim, v, passes=8); r.after(vr); r.ctx.synchronize()
    acc = r._accum.numpy(); gpu = acc[..., 0] / np.maximum(acc[..., 3], 1)
    datas = []
    for e in range(n):
        d = mujoco.MjData(m); d.qpos[:] = sim.d.qpos.numpy()[e]; mujoco.mj_forward(m, d); datas.append(d)
    r2 = Tier2Renderer(m, n, width=32, height=32, camera="down", spp=16, max_bounces=0, seed=5, skybox=False)
    host = r2.render_host(datas, passes=8)["hdr"][..., 0]
    assert np.abs(gpu - host).max() < 1e-3 * host.max(), np.abs(gpu - host).max()

    def spread(img):      # second moments of the lit patch along image x / y
        w = img / img.sum(); y, x = np.mgrid[:32, :32]
        return float((w * (x - (w * x).sum()) ** 2).sum()), float((w * (y - (w * y).sum()) ** 2).sum())
    sx0, sy0 = spread(host[0]); sx3, sy3 = spread(host[3])
    print(f"strip spread x / y: 0 deg {sx0:.1f} / {sy0:.1f}, 90 deg {sx3:.1f} / {sy3:.1f}")
    assert sx0 > 1.2 * sy0 and sy3 > 1.2 * sx3
