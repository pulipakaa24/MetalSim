"""Byte-identity of the ray-traced sensors on scenes without transparent geoms, between two checkouts.

    cd <checkout A> && python <this file> --out a.npz
    cd <checkout B> && python <this file> --out b.npz
    python <this file> --compare a.npz b.npz

The metalsim package is imported from the current directory, so one script measures any checkout. Scenes: SO-101
(scene_box_rl), the lidar test scene, ANYmal C (menagerie scene); sensors: a 16 x 360 rotary lidar
(default kernel), the same with divergence and two returns (extended kernel), a 96 x 72 depth camera. 8 envs with
perturbed states. The transparency work (SENSOR_TR kernels) must leave every output bit-identical here.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "assets")
MENAGERIE = os.environ.get("MENAGERIE", os.path.join(os.path.dirname(os.path.dirname(HERE)), "upstream", "mujoco_menagerie"))

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
</mujoco>"""


def scenes():
    import mujoco
    yield "lidar_scene", mujoco.MjModel.from_xml_string(LIDAR_SCENE), "lidar", "cam"
    so = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "so101", "scene_box_rl.xml"))
    yield "so101", so, None, "base_cam"
    spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "anybotics_anymal_c", "scene.xml"))
    spec.worldbody.add_site(name="ext_lidar", pos=[1.5, 0.0, 1.0])
    cam = spec.worldbody.add_camera(); cam.name = "ext_cam"; cam.pos = [2.5, 0, 1.0]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float).ravel()); cam.quat = q   # looks along -x
    yield "anymal_c", spec.compile(), "ext_lidar", "ext_cam"


def run(out):
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root
    import mujoco
    from metalsim.sensors.raytrace import RayTracer
    import metalsim
    print("metalsim from", os.path.dirname(metalsim.__file__))
    res = {}
    n = 8
    for name, m, site, cam in scenes():
        rng = np.random.default_rng(0)
        datas = []
        for e in range(n):
            d = mujoco.MjData(m)
            d.qpos[:] += rng.uniform(-0.05, 0.05, m.nq)
            mujoco.mj_forward(m, d)
            datas.append(d)
        rt = RayTracer(m, n, max_range=8.0)
        sens = []
        if site is None:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar") if m.nsite else -1
            site = int(sid) if sid >= 0 else 0
        if m.nsite:
            A, E = np.meshgrid(np.linspace(-np.pi, np.pi, 360, endpoint=False), np.deg2rad(np.linspace(-15, 15, 16)))
            sens.append(rt.make_lidar(site, A.ravel(), E.ravel(), name="lidar"))
            sens.append(rt.make_lidar(site, A.ravel(), E.ravel(), name="lidar_ext", divergence_deg=(0.3, 0.3), spot_rays=4,
                                      max_returns=2))
        dc = rt.make_depth_camera(cam, 96, 72)
        dc.name = "depth"
        sens.append(dc)
        o = rt.trace_host(datas, sens)
        print(name, "transparency:", getattr(rt, "has_transparency", "n/a"))
        for k, v in o.items():
            if isinstance(v, dict):
                for f, a in v.items():
                    res[f"{name}/{k}/{f}"] = a
            else:
                res[f"{name}/{k}"] = v
    np.savez(out, **res)
    print("wrote", out, len(res), "arrays")


def compare(a, b):
    A, B = np.load(a), np.load(b)
    keys = sorted(set(A.files) | set(B.files))
    bad = 0
    for k in keys:
        if k not in A.files or k not in B.files:
            print("missing", k); bad += 1; continue
        same = A[k].shape == B[k].shape and A[k].tobytes() == B[k].tobytes()
        if not same:
            bad += 1
        print(f"{'identical' if same else 'DIFFERENT'}  {k}  {A[k].shape}")
    print(f"{len(keys) - bad} of {len(keys)} arrays byte-identical")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(1 if compare(*a.compare) else 0)
    run(a.out)
