"""Lidar extras on the Metal ray tracer (beam divergence, multi-return, per-geom reflectance),
modelled on Isaac Sim's RTX lidar attributes (divergenceHorDeg/VerDeg, maxReturns,
minDistBetweenEchosM, material reflectance). The default lidar (single infinitesimal ray) is
unchanged and checked against mj_ray in test_sensors_rt.py / test_lidar_nav.py."""
import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.sensors.raytrace import Lidar, RayTracer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

SCENE = """
<mujoco>
  <worldbody>
    <body name="sensor" pos="0 0 0.5"><site name="lidar"/></body>
    <!-- edge scene (+x): box front face at x = 2.0 covering y < 0, wall face at x = 2.2 -->
    <geom name="box" type="box" pos="2.5 -0.5 0.5" size="0.5 0.5 0.5"/>
    <geom name="wall" type="box" pos="2.3 0 0.5" size="0.1 2 1"/>
    <!-- pole scene (-x): 6 mm pole at x = -1.0 (front surface 0.997 m) in front of a wall at x = -2.2 -->
    <geom name="pole" type="cylinder" pos="-1.0 0 0.5" size="0.003 1"/>
    <geom name="wall2" type="box" pos="-2.3 0 0.5" size="0.1 2 1"/>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def setup():
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rt = RayTracer(m, 2, max_range=10.0)
    return m, d, rt


def test_divergence_mixed_pixel_at_edge(setup):
    """A beam whose 1 deg spot straddles the box edge (2.0 m) and the wall (2.2 m) returns an
    intermediate, power-weighted range (echoes closer than minDistBetweenEchosM merge); beams whose
    spot lies on one surface return that surface; the infinitesimal default returns one of the two."""
    m, d, rt = setup
    az = np.deg2rad([-2.0, 0.0, 2.0])      # -2: box, 0: edge, +2: wall
    base = rt.make_lidar("lidar", az, np.zeros(3))
    div = rt.make_lidar("lidar", az, np.zeros(3), divergence_deg=(1.0, 1.0), spot_rays=32)
    rt.trace_host([d, d], [base, div])
    rb, rd = base.numpy()["range"], div.numpy()["range"]
    assert rb.shape == (2, 3) and rd.shape == (2, 3)
    assert abs(rd[0, 0] - 2.0 / np.cos(np.deg2rad(2.0))) < 5e-3 and abs(rd[0, 2] - 2.2 / np.cos(np.deg2rad(2.0))) < 5e-3
    assert 2.03 < rd[0, 1] < 2.17, rd[0, 1]
    # power weights: half the spot at 2.0 m, half at 2.2 m -> (2.0/2.0^2 + 2.2/2.2^2) / (1/2.0^2 + 1/2.2^2) ~ 2.09
    assert abs(rd[0, 1] - 2.09) < 0.03, rd[0, 1]
    assert rb[0, 1] in (pytest.approx(2.0, abs=1e-3), pytest.approx(2.2, abs=1e-3))
    # with a finer echo separation the two surfaces are two returns instead of a mixed pixel
    two = rt.make_lidar("lidar", az, np.zeros(3), divergence_deg=(1.0, 1.0), spot_rays=32, max_returns=2, hits_per_ray=1,
                        min_echo_sep=0.1)
    rt.trace_host([d, d], [two])
    t = two.numpy()
    assert t["n_returns"][0].tolist() == [1, 2, 1]
    assert np.allclose(t["range"][0, 1], [2.0, 2.2], atol=5e-3)


def test_multi_return_thin_pole_in_front_of_wall(setup):
    """maxReturns = 2: a 6 mm pole at ~1 m in front of a wall at 2.2 m gives two returns, by re-casting
    past the pole (single ray) and by a divergent spot the pole covers only partly (no re-cast)."""
    m, d, rt = setup
    az = np.array([np.pi])
    base = rt.make_lidar("lidar", az, [0.0])
    recast = rt.make_lidar("lidar", az, [0.0], max_returns=2)
    spot = rt.make_lidar("lidar", az, [0.0], max_returns=2, hits_per_ray=1, divergence_deg=(2.0, 2.0), spot_rays=32)
    rt.trace_host([d, d], [base, recast, spot])
    b, r, s = base.numpy(), recast.numpy(), spot.numpy()
    assert abs(b["range"][0, 0] - 0.997) < 2e-3                           # default: first hit only
    assert r["range"].shape == (2, 1, 2) and r["n_returns"][0, 0] == 2
    assert abs(r["range"][0, 0, 0] - 0.997) < 2e-3 and abs(r["range"][0, 0, 1] - 2.2) < 2e-3
    assert s["n_returns"][0, 0] == 2
    assert abs(s["range"][0, 0, 0] - 1.0) < 5e-3 and abs(s["range"][0, 0, 1] - 2.2) < 5e-3
    # the pole covers a small part of a 2 deg spot at 1 m (6 mm of 35 mm): its echo is the weaker one
    assert s["intensity"][0, 0, 0] < s["intensity"][0, 0, 1] * 2.2 ** 2
    # slots: first return is the pole, second the wall
    geoms = np.asarray(rt.tables.geoms)
    gname = lambda sl: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(geoms[sl]))
    assert gname(r["slot"][0, 0, 0]) == "pole" and gname(r["slot"][0, 0, 1]) == "wall2"


def test_reflectance_table_and_default_equivalence(setup):
    """Per-geom reflectance scales intensity = reflectance cos(theta) / r^2; with no divergence and
    one return the extended kernel reproduces the default kernel's ranges and points."""
    m, d, rt = setup
    az = np.deg2rad(np.linspace(-170, 170, 64))
    el = np.deg2rad(np.linspace(-20, 20, 64))
    base = rt.make_lidar("lidar", az, el)
    ref = rt.make_lidar("lidar", az, el, reflectance={"wall": 0.9, "box": 0.1, "wall2": 0.3, "pole": 0.7})
    rt.trace_host([d, d], [base, ref])
    b, x = base.numpy(), ref.numpy()
    assert np.allclose(b["range"], x["range"], rtol=0, atol=1e-6) and np.array_equal(b["slot"], x["slot"])   # 1 ulp: fast-math order of the direction
    assert np.allclose(b["points"], x["points"], atol=1e-6)
    geoms = np.asarray(rt.tables.geoms)
    tab = {"wall": 0.9, "box": 0.1, "wall2": 0.3, "pole": 0.7}
    hit = x["slot"][0] >= 0
    assert hit.sum() > 20
    for k in np.nonzero(hit)[0]:
        g = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(geoms[x["slot"][0, k]]))
        dirv = x["points"][0, k] - d.site_xpos[0]; dirv /= np.linalg.norm(dirv)
        cos = max(-np.dot(x["normals"][0, k], dirv), 0.0)
        assert abs(x["intensity"][0, k] - tab[g] * cos / x["range"][0, k] ** 2) < 1e-4 * max(1.0, x["intensity"][0, k])


def test_spot_pattern():
    s = Lidar.spot_pattern((2.0, 1.0), 64)
    assert abs(s[:, 2].sum() - 1) < 1e-6
    assert np.abs(s[:, 0]).max() <= np.tan(np.deg2rad(1.0)) + 1e-7 and np.abs(s[:, 1]).max() <= np.tan(np.deg2rad(0.5)) + 1e-7
    assert np.allclose(Lidar.spot_pattern((1.0, 1.0), 1)[0], [0, 0, 1, 0])
