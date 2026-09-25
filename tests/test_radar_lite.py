"""Radar-lite (metalsim.sensors.radar): Isaac RTX radar point-cloud fields (radial distance, azimuth,
elevation, radial velocity, RCS in dBsm) from the Metal ray tracer and MuJoCo body velocities.
Radial velocity is checked against MuJoCo C's mj_objectVelocity at each hit point (moving and
spinning target, moving sensor); the Lambertian RCS proxy sums to 4 rho A on a plate."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.sensors.radar import Radar
from metalsim.sensors.raytrace import RayTracer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

XML = """
<mujoco><option gravity="0 0 0"/>
  <worldbody>
    <body name="sensor" pos="0 0 0.5"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <site name="radar"/></body>
    <body name="target" pos="3 0.3 0.5"><freejoint/><geom name="tgt" type="box" size="0.3 0.4 0.3" contype="0" conaffinity="0"/></body>
    <geom name="plate" type="box" pos="-4 0 0.5" size="0.01 0.4 0.3" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def scene():
    m = mujoco.MjModel.from_xml_string(XML)
    n = 2
    sim = BatchSim(m, n, options=BatchSimOptions(njmax=16))
    rt = RayTracer(m, n, max_range=10.0)
    return m, sim, rt


def test_radial_velocity_matches_mujoco(scene):
    m, sim, rt = scene
    radar = Radar(rt, sim, "radar", fov_deg=(40.0, 20.0), n_az=64, n_el=16)
    q = np.tile(m.qpos0, (sim.n, 1)).astype(np.float32)
    qv = np.zeros((sim.n, m.nv), np.float32)
    qv[:, 0:3] = [0.7, 0.2, 0.0]                  # sensor translating
    qv[:, 6:9] = [-2.0, 0.5, 0.1]; qv[:, 9:12] = [0.0, 0.4, 3.0]   # target: translating and spinning (world ang. vel. for free joints)
    qv[1, 6] = 1.5                                 # world 1: target receding
    sim.set_state(q, qv); sim.synchronize()
    v = radar.trace(); sim.synchronize()
    o = radar.numpy()
    res = np.zeros(6)
    for w in range(sim.n):
        d = sim.get_world(w)
        site = radar.site
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, site, res, 0)
        vs = res[3:].copy()
        valid = o["valid"][w]
        assert valid.sum() > 100
        geoms = np.asarray(rt.tables.geoms)
        err = []
        for k in np.nonzero(valid)[0]:
            p = o["points"][w, k]; g = int(geoms[o["slot"][w, k]]); b = m.geom_bodyid[g]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
            vp = res[3:] + np.cross(res[:3], p - d.xpos[b])
            dirv = (p - d.site_xpos[site]); rng = np.linalg.norm(dirv); dirv /= rng
            err.append(abs(o["radial_velocity"][w, k] - np.dot(vp - vs, dirv)))
            assert abs(o["range"][w, k] - rng) < 1e-4
        err = np.array(err)
        print(f"world {w}: {len(err)} detections, |vr - mujoco| max {err.max():.2e} m/s, "
              f"vr range [{o['radial_velocity'][w][valid].min():.2f}, {o['radial_velocity'][w][valid].max():.2f}]")
        assert err.max() < 1e-4
    assert o["radial_velocity"][0][o["valid"][0]].mean() < 0 < o["radial_velocity"][1][o["valid"][1]].mean()  # approaching / receding
    # azimuth / elevation are the ray angles in the sensor frame
    assert np.all(np.abs(o["azimuth"]) <= np.deg2rad(20.0) + 1e-6) and np.all(np.abs(o["elevation"]) <= np.deg2rad(10.0) + 1e-6)
    noisy = Radar.apply_noise(radar)
    assert torch.equal(noisy["valid"], radar.out["valid"])


def test_rcs_plate_sums_to_lambertian(scene):
    m, sim, rt = scene
    rho = 0.5
    radar = Radar(rt, sim, "radar", fov_deg=(30.0, 20.0), n_az=256, n_el=128, reflectance={"plate": rho, "tgt": rho})
    q = np.tile(m.qpos0, (sim.n, 1)).astype(np.float32)
    q[:, 3:7] = [0, 0, 0, 1]                      # sensor yawed 180 deg: faces the plate at x = -4
    sim.set_state(q, np.zeros((sim.n, m.nv), np.float32)); sim.synchronize()
    radar.trace(); sim.synchronize()
    o = radar.numpy()
    geoms = np.asarray(rt.tables.geoms)
    on_plate = o["valid"][0] & (geoms[np.maximum(o["slot"][0], 0)] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "plate"))
    total = (10 ** (o["rcs_dbsm"][0][on_plate] / 10)).sum()
    ref = 4 * rho * 0.8 * 0.6
    print(f"plate: {on_plate.sum()} rays, sum sigma {total:.4f} m^2 vs 4 rho A {ref:.4f}")
    assert abs(total - ref) / ref < 0.03
    assert np.allclose(o["range"][0][on_plate].min(), 4 - 0.01, atol=2e-3)
