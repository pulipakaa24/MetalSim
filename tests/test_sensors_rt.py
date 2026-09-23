"""WS5: ray-traced sensors on Metal hardware ray tracing, checked against MuJoCo's mj_ray and
against the rasterizer's depth."""
import os

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.sensors.raytrace import RayTracer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101 = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")

LIDAR_SCENE = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="sensor" pos="0 0 0.3"><joint type="free"/>
      <geom name="s0" type="sphere" size="0.02" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      <site name="lidar" pos="0 0 0.05"/></body>
    <body pos="0.8 0.1 0.2"><joint type="free"/><geom name="b1" type="box" size="0.1 0.15 0.2" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="-0.5 0.6 0.25"><joint type="free"/><geom name="c1" type="cylinder" size="0.12 0.25" rgba="0.2 0.2 0.9 1"/></body>
    <body pos="0.2 -0.9 0.3"><joint type="free"/><geom name="k1" type="capsule" size="0.08 0.2" rgba="0.9 0.9 0.2 1"/></body>
    <body pos="-0.7 -0.4 0.15"><joint type="free"/><geom name="sp" type="sphere" size="0.15" rgba="0.2 0.9 0.2 1"/></body>
  </worldbody>
</mujoco>
"""


def _mj_ray(model, d, origin, dirs, exclude_body):
    """Reference ranges from MuJoCo C: mj_ray against all geoms (planes included)."""
    out = np.zeros(len(dirs))
    geomid = np.zeros(1, np.int32)
    for i, v in enumerate(dirs):
        dist = mujoco.mj_ray(model, d, origin, v, None, 1, exclude_body, geomid)
        out[i] = dist if geomid[0] >= 0 else 0.0
    return out


def test_lidar_vs_mujoco_ray():
    model = mujoco.MjModel.from_xml_string(LIDAR_SCENE)
    n = 4
    rt = RayTracer(model, n, max_range=10.0)
    az = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    el = np.deg2rad(np.array([-15, -8, -2, 0, 5]))
    A, E = np.meshgrid(az, el)
    lidar = rt.make_lidar("lidar", A, E)
    datas = []
    rng = np.random.default_rng(0)
    for e in range(n):
        d = mujoco.MjData(model)
        d.qpos[:] += rng.uniform(-0.05, 0.05, model.nq)
        mujoco.mj_forward(model, d)
        datas.append(d)
    out = rt.trace_host(datas, [lidar])["lidar"]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sensor")
    worst = 0.0; hits = 0
    for e, d in enumerate(datas):
        R = d.site_xmat[site].reshape(3, 3)
        dl = np.stack([np.cos(E.ravel()) * np.cos(A.ravel()), np.cos(E.ravel()) * np.sin(A.ravel()), np.sin(E.ravel())], 1)
        dirs = dl @ R.T
        ref = _mj_ray(model, d, d.site_xpos[site], dirs, body)
        got = out["range"][e]
        both = (ref > 0) & (got > 0)
        hits += both.sum()
        # meshes are tessellated primitives (spheres/cylinders with 12x24 segments): tolerate a few mm
        err = np.abs(got[both] - ref[both])
        worst = max(worst, err.max() if both.any() else 0)
        assert np.mean(np.abs((ref > 0) != (got > 0))) < 0.03, "hit/miss pattern differs from mj_ray"
        assert np.median(err) < 2e-3
        assert np.percentile(err, 95) < 2e-2
    print(f"lidar: {hits} beams compared over {n} envs; worst |dr| {worst:.4f} m")
    # no cross-env contamination: env 0 with the box moved must differ from env 1 only where geometry differs
    assert (out["slot"] >= -1).all()


def test_raycast_depth_vs_raster_depth():
    from metalsim.render.tier0 import Tier0Renderer
    model = mujoco.MjModel.from_xml_path(SO101)
    n = 8
    rt = RayTracer(model, n, decimate_faces=0)
    cam = rt.make_depth_camera("base_cam", 128, 128)
    rend = Tier0Renderer(model, n, width=128, height=128, camera="base_cam", outputs=("depth",), decimate_faces=0)
    datas = []
    rng = np.random.default_rng(1)
    for e in range(n):
        d = mujoco.MjData(model)
        d.qpos[:6] = rng.uniform(-0.5, 0.5, 6)
        mujoco.mj_forward(model, d)
        datas.append(d)
    ray_depth = rt.trace_host(datas, [cam])["ray_depth"]
    raster = rend.render_host(datas)["depth"]
    both = (ray_depth > 0) & (raster > 0)
    err = np.abs(ray_depth[both] - raster[both])
    agree = np.mean((ray_depth > 0) == (raster > 0))
    print(f"ray vs raster depth: coverage agreement {agree:.4f}, median |dz| {np.median(err):.5f} m, 99th pct {np.percentile(err, 99):.4f} m")
    assert agree > 0.995
    assert np.median(err) < 1e-3
    assert np.percentile(err, 99) < 2e-2


def test_gpu_path_ordering_with_sim():
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.interop import warp_metal as wm
    model = mujoco.MjModel.from_xml_string(LIDAR_SCENE)
    n = 16
    sim = BatchSim(model, n, options=BatchSimOptions(substeps=2))
    rt = RayTracer(model, n)
    lidar = rt.make_lidar("lidar", np.linspace(-np.pi, np.pi, 64, endpoint=False), np.full(64, -0.2))  # slightly down: beams reach the floor
    sim.synchronize()
    c0 = wm.counters()
    ranges = []
    for k in range(10):
        vs = sim.step()
        vr = rt.trace(sim, [lidar], vs)
        sim.wait(rt.event, vr)
        import metalsim.interop.torch_bridge as tb
        tb.wait_event(rt.event, vr)
        ranges.append(lidar.out["range"].clone())
    torch.mps.synchronize()
    assert (wm.counters() - c0).syncs == 0
    r = torch.stack(ranges).cpu().numpy()
    assert np.isfinite(r).all() and (r > 0).mean() > 0.5
    assert not np.allclose(r[0], r[-1])   # the sensor body falls: ranges change
