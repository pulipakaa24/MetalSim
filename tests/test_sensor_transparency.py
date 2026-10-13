"""Transparent and translucent geoms in the ray-traced depth camera and lidar (research_notes "transparency_area_lights",
docs/RENDER_COVERAGE.md). Analytic glass-pane-in-front-of-a-wall checks per mode, MuJoCo's mj_ray where MuJoCo defines
the behaviour (depth "surface", lidar "surface" = the rangefinder), lidar materials (clear glass, mirror ghosts), return
modes, and the plain kernels kept for scenes without transparency."""
import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.sensors.raytrace import RayTracer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

# sensor at the origin looking along +x (lidar site frame = world; camera optical axis +x)
# wall front face at x = 3.0; "pane" (alpha 0.25) front face at x = 1.995 for y in (0.1, 0.9);
# "pane2" (alpha 0.5) front face at x = 1.495 for y in (0.3, 0.7); "glass" (opaque rgba, lidar material) front face
# x = 1.995 for y in (-0.9, -0.1); "mirror" at (2, -2.5) turned 45 deg, side wall face at y = 1.95 ... see below
SCENE = """
<mujoco>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>
    <site name="lidar" pos="0 0 0"/>
    <geom name="wall" type="box" pos="3.05 0 0" size="0.05 4 2" rgba="0.8 0.8 0.8 1"/>
    <geom name="pane" type="box" pos="2 0.5 0" size="0.005 0.4 1" rgba="0.2 0.4 0.9 0.25"/>
    <geom name="pane2" type="box" pos="1.5 0.5 0" size="0.005 0.2 1" rgba="0.9 0.4 0.2 0.5"/>
    <geom name="glass" type="box" pos="2 -0.5 0" size="0.005 0.4 1" rgba="0.6 0.8 0.9 1"/>
    <geom name="mirror" type="box" pos="2 -2.5 0" size="0.005 0.5 1" euler="0 0 45" rgba="0.9 0.9 0.9 1"/>
    <geom name="side" type="box" pos="2 -4.5 0" size="2 0.05 2" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>"""
RHO = 0.5   # default reflectance (no MuJoCo material reflectance)


@pytest.fixture(scope="module")
def setup():
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rt = RayTracer(m, 2, max_range=12.0)
    return m, d, rt


def _beam(rt, d, az, **kw):
    lid = rt.make_lidar("lidar", np.asarray(az, np.float32), np.zeros(len(az), np.float32), **kw)
    return rt.trace_host([d, d], [lid])[lid.name]


def test_depth_camera_modes(setup):
    """Depth through the 25 % pane: "surface" = the pane (MuJoCo's depth and mj_ray), "behind" = the wall (Isaac
    Real-Time cutout), alpha_threshold (Isaac's opacity_threshold): the pane counts when alpha > t."""
    m, d, rt = setup
    W = H = 48
    cams = {"surface": rt.make_depth_camera("cam", W, H), "behind": rt.make_depth_camera("cam", W, H, transparent_ids="behind"),
            "opaque": rt.make_depth_camera("cam", W, H, transparent_ids="opaque"),
            "t02": rt.make_depth_camera("cam", W, H, alpha_threshold=0.2),
            "t03": rt.make_depth_camera("cam", W, H, alpha_threshold=0.3)}
    for k, c in cams.items():
        c.name = k
    out = rt.trace_host([d, d], list(cams.values()))
    assert cams["surface"].lmat is None and cams["opaque"].lmat is None and cams["behind"].lmat is not None
    # every pixel against mj_ray (rays through the pixel centres; z-depth = distance * cos to the optical axis)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    f = 0.5 * H / np.tan(np.deg2rad(60) / 2)
    R = d.cam_xmat[cid].reshape(3, 3)
    ref = np.zeros((H, W))
    gid = np.zeros(1, np.int32)
    for y in range(H):
        for x in range(W):
            dl = np.array([(x + 0.5 - W / 2) / f, -(y + 0.5 - H / 2) / f, -1.0])
            v = R @ (dl / np.linalg.norm(dl))
            t = mujoco.mj_ray(m, d, d.cam_xpos[cid], v, None, 1, -1, gid)
            ref[y, x] = t * (-(R.T @ v)[2]) if gid[0] >= 0 else 0.0
    sfc = out["surface"][0]
    assert np.abs(sfc - ref).max() < 2e-4, np.abs(sfc - ref).max()
    assert np.array_equal(out["surface"][0], out["opaque"][0])
    assert np.array_equal(out["surface"][0], out["t02"][0])
    # pane pixels (alpha 0.25 and 0.5 at 1.995 / 1.495 m): behind and t03 see the wall (or pane2 for t03)
    left = np.arange(W)[None, :] < W // 2                   # image left = world +y (the "glass" pane is on the right)
    pane = np.isclose(sfc, 1.995, atol=1e-3) & left
    pane2 = np.isclose(sfc, 1.495, atol=1e-3) & left
    assert pane.sum() > 50 and pane2.sum() > 20
    assert np.allclose(out["behind"][0][pane | pane2], 3.0, atol=1e-3)
    assert np.allclose(out["t03"][0][pane], 3.0, atol=1e-3)          # 0.25 <= 0.3: cut out
    assert np.allclose(out["t03"][0][pane2], 1.495, atol=1e-3)       # 0.5 > 0.3: kept
    assert np.array_equal(out["behind"][0][~(pane | pane2)], sfc[~(pane | pane2)])
    assert np.allclose(out["behind"][0], out["behind"][1], atol=1e-5)   # envs alike (offsets in the shared world: float32)


def test_lidar_modes_analytic(setup):
    """Lidar through the 25 % pane at several angles: per mode, range and intensity against the analytic values:
    pane P a rho cos / r^2, wall P (1 - a) rho cos / r^2 (transmit); surface = mj_ray (MuJoCo's rangefinder)."""
    m, d, rt = setup
    y = np.array([0.15, 0.2, 0.35])     # where the beam crosses x = 1.995 (the pane; pane2 is crossed at y > 0.3 only past 0.4)

    az = np.arctan2(y, 1.995)
    c = np.cos(az)
    r_p, r_w = 1.995 / c, 3.0 / c
    p_p, p_w = 0.25 * RHO * c / r_p ** 2, 0.75 * RHO * c / r_w ** 2
    two = _beam(rt, d, az, max_returns=2)
    assert np.allclose(two["range"][0], np.stack([r_p, r_w], 1), atol=2e-4)
    assert np.allclose(two["intensity"][0], np.stack([p_p, p_w], 1), rtol=1e-4)
    assert two["n_returns"][0].tolist() == [2] * len(az)
    one = _beam(rt, d, az)                                   # transmit, strongest
    strongest = np.where(p_p > p_w, r_p, r_w)
    assert np.allclose(one["range"][0], strongest, atol=2e-4)
    for mode, want in (("surface", r_p), ("behind", r_w), ("opaque", r_p)):
        o = _beam(rt, d, az, transparency=mode)
        assert np.allclose(o["range"][0], want, atol=2e-4), mode
        assert o["range"].shape == (2, len(az))
    # "surface" is MuJoCo's rangefinder: mj_ray from the site
    o = _beam(rt, d, az, transparency="surface")
    gid = np.zeros(1, np.int32)
    ref = [mujoco.mj_ray(m, d, np.zeros(3), np.array([np.cos(a), np.sin(a), 0.0]), None, 1, -1, gid) for a in az]
    assert np.allclose(o["range"][0], ref, atol=2e-4)


def test_lidar_return_modes(setup):
    """Two panes and the wall on one beam (three echoes): first / last / strongest and the dual modes."""
    m, d, rt = setup
    az = np.array([np.arctan2(0.5, 1.995)])                   # through pane2 (x 1.495) and pane (x 1.995)
    c = np.cos(az[0])
    r = np.array([1.495, 1.995, 3.0]) / c
    a = np.array([0.5, 0.25, 1.0])
    P = np.array([1.0, 0.5, 0.5 * 0.75])                      # power reaching each surface
    p = P * a * RHO * c / r ** 2
    order = np.argsort(-p)
    got = {}
    for mode, R in (("strongest", 1), ("first", 1), ("last", 1), ("strongest_last", 2), ("first_last", 2),
                    ("first_strongest", 2), ("strongest", 3)):
        o = _beam(rt, d, az, return_mode=mode, max_returns=R)
        got[(mode, R)] = o["range"][0].reshape(-1)
        assert np.allclose(o["intensity"][0].reshape(-1)[:o["n_returns"][0, 0]],
                           [p[np.argmin(np.abs(r - x))] for x in got[(mode, R)][:o["n_returns"][0, 0]]], rtol=1e-4)
    assert np.allclose(got[("strongest", 1)], [r[order[0]]], atol=2e-4)
    assert np.allclose(got[("first", 1)], [r[0]], atol=2e-4)
    assert np.allclose(got[("last", 1)], [r[2]], atol=2e-4)
    second = r[[k for k in order if k != 2][0]]
    assert np.allclose(got[("strongest_last", 2)], sorted([second, r[2]]), atol=2e-4)
    assert np.allclose(got[("first_last", 2)], [r[0], r[2]], atol=2e-4)
    assert np.allclose(got[("first_strongest", 2)], sorted([r[0], r[[k for k in order if k != 0][0]]]), atol=2e-4)
    assert np.allclose(got[("strongest", 3)], r, atol=2e-4)
    # detection threshold (Isaac minReflectance at minReflectionRangeM): drops the weakest echo
    thr = (np.sort(p)[0] + np.sort(p)[1]) / 2
    o = _beam(rt, d, az, max_returns=3, min_reflectance=thr * 4.0, min_reflectance_range=2.0)
    assert o["n_returns"][0, 0] == 2 and np.allclose(np.sort(o["intensity"][0].reshape(-1)[:2]), np.sort(p)[1:], rtol=1e-4)


def test_clear_glass_and_mirror(setup):
    """clear_glass: the specular return only near normal incidence (analytic Beckmann backscatter), ~92 % transmitted;
    mirror: the echo behind it at the reflected path length (a ghost), with the Fresnel-scaled power."""
    m, d, rt = setup
    mats = {"glass": "clear_glass", "mirror": "mirror"}
    az = np.array([np.arctan2(-0.5, 1.995), np.arctan2(-0.15, 1.995)])     # 14.0 deg and 4.3 deg off normal
    two = _beam(rt, d, az, max_returns=2, lidar_materials=mats)
    c = np.cos(az)
    F = 0.077 + (1 - 0.077) * (1 - c) ** 5
    assert two["n_returns"][0].tolist() == [1, 1]            # the pane returns nothing a few degrees off normal
    assert np.allclose(two["range"][0, :, 0], 3.0 / c, atol=2e-4)
    assert np.allclose(two["intensity"][0, :, 0], (1 - F) * RHO * c / (3.0 / c) ** 2, rtol=1e-4)
    # normal incidence: the strong specular return (pi F D G / 4 at cos = 1, D = 1 / (pi alpha^2), alpha = 0.1^2)
    lid = rt.make_lidar("lidar", np.array([0.0], np.float32), np.array([0.0], np.float32), max_returns=2,
                        lidar_materials=mats)
    lid_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    dd = mujoco.MjData(m); mujoco.mj_forward(m, dd)
    dd.site_xpos[lid_site] = (0.0, -0.5, 0.0)                # in front of the glass, beam along its normal
    o = rt.trace_host([dd, dd], [lid])["lidar"]
    spec = np.pi * 0.077 / (np.pi * 1e-4) / 4.0
    assert o["n_returns"][0, 0] == 2
    assert np.allclose(o["range"][0, 0], [1.995, 3.0], atol=2e-4)
    assert np.allclose(o["intensity"][0, 0], [spec / 1.995 ** 2, 0.923 * RHO / 9.0], rtol=2e-3)
    # mirror at 45 deg: a beam along +x from (0, -2.5) reflects to -y onto the side wall (face y = -4.45)
    dd.site_xpos[lid_site] = (0.0, -2.5, 0.0)
    o = rt.trace_host([dd, dd], [lid])["lidar"]
    x_hit = 2.0 - 0.005 * np.sqrt(2.0)                        # the mirror's front face on the beam
    path = x_hit + (4.45 - 2.5)
    F45 = 0.95 + 0.05 * (1 - np.cos(np.pi / 4)) ** 5
    assert o["n_returns"][0, 0] == 1
    assert np.isclose(o["range"][0, 0, 0], path, atol=5e-4)
    assert np.allclose(o["points"][0, 0, 0], (path, -2.5, 0.0), atol=5e-4)   # reported along the beam: behind the mirror
    assert np.isclose(o["intensity"][0, 0, 0], F45 * RHO / path ** 2, rtol=1e-3)


def test_divergence_with_transparency_and_determinism(setup):
    """Sub-rays over a 0.5 deg spot through the pane: the echo powers sum to the single-ray values (the spot lies on
    the pane), and two traces are bitwise equal (deterministic)."""
    m, d, rt = setup
    az = np.array([np.arctan2(0.2, 1.995)])
    lid = rt.make_lidar("lidar", az.astype(np.float32), np.zeros(1, np.float32), max_returns=2, divergence_deg=(0.5, 0.5),
                        spot_rays=16)
    a = rt.trace_host([d, d], [lid])["lidar"]
    b = rt.trace_host([d, d], [lid])["lidar"]
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    one = _beam(rt, d, az, max_returns=2)
    assert np.allclose(a["intensity"][0], one["intensity"][0], rtol=2e-3)
    assert np.allclose(a["range"][0], one["range"][0], atol=2e-3)


def test_plain_kernels_without_transparency():
    """A scene without transparent geoms keeps the plain kernels (no SENSOR_TR library is built), and the SENSOR_TR
    kernels forced onto it with an all-opaque table give the same outputs bit for bit."""
    xml = SCENE.replace(' rgba="0.2 0.4 0.9 0.25"', ' rgba="0.2 0.4 0.9 1"').replace(' rgba="0.9 0.4 0.2 0.5"', ' rgba="0.9 0.4 0.2 1"')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rt = RayTracer(m, 2, max_range=12.0)
    assert not rt.has_transparency
    az = np.linspace(-1.2, 1.2, 64).astype(np.float32); el = np.linspace(-0.3, 0.3, 64).astype(np.float32)
    plain = [rt.make_lidar("lidar", az, el, name="l1"), rt.make_lidar("lidar", az, el, name="l2", max_returns=2),
             rt.make_depth_camera("cam", 32, 32)]
    assert not plain[0].tr and not plain[1].tr and plain[2].lmat is None
    out = rt.trace_host([d, d], plain)
    assert rt._tr_pipes is None
    forced = [rt.make_lidar("lidar", az, el, name="l1"), rt.make_lidar("lidar", az, el, name="l2", max_returns=2),
              rt.make_depth_camera("cam", 32, 32)]
    for s in forced:
        s.lmat = rt.sensor_material_table()
        s.lmat_buf = rt.ctx.buffer(s.lmat.nbytes, s.lmat, "forced")
        if hasattr(s, "tr"):
            s.tr = True
    out2 = rt.trace_host([d, d], forced)
    for k in out:
        if isinstance(out[k], dict):
            for f in out[k]:
                assert np.array_equal(out[k][f], out2[k][f]), (k, f)
        else:
            assert np.array_equal(out[k], out2[k]), k


def test_streaming_kernel_matches_lidar_ext(setup):
    """One sub-ray and one return run the streaming kernel (lidar_tr1, no candidate arrays): equal, bit for bit
    (points to float rounding), to the general lidar_ext over a fan of beams through panes, glass, the mirror and the walls, per return mode."""
    m, d, rt = setup
    A, E = np.meshgrid(np.linspace(-2.0, 2.0, 181), np.deg2rad(np.linspace(-20, 20, 9)))
    az, el = A.ravel().astype(np.float32), E.ravel().astype(np.float32)
    mats = {"glass": "clear_glass", "mirror": "mirror"}
    for kw in (dict(), dict(return_mode="first"), dict(return_mode="last"), dict(lidar_materials=mats),
               dict(lidar_materials=mats, return_mode="first", min_reflectance=0.02, min_reflectance_range=4.0)):
        a = rt.make_lidar("lidar", az, el, name="a", **kw)
        b = rt.make_lidar("lidar", az, el, name="b", **kw)
        assert a.stream and b.stream
        b.stream = False
        o = rt.trace_host([d, d], [a, b])
        assert (o["a"]["n_returns"] > 0).mean() > 0.5
        for f in o["a"]:       # the streaming kernel puts the point at the echo's range on the beam: float rounding
            if f == "points":
                assert np.allclose(o["a"][f], o["b"][f], atol=2e-6), (kw, f)
            else:
                assert np.array_equal(o["a"][f], o["b"][f]), (kw, f)
