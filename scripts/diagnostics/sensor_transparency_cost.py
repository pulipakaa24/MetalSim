"""GPU time of the ray-traced lidar and depth camera with and without transparent geoms (SENSOR_TR kernels).

    python scripts/diagnostics/sensor_transparency_cost.py --n 1024 [--out runs/render_sensor_tr/cost_1024.json]

Scene: the lidar test scene (a floating sensor among a box, cylinder, capsule and sphere on a floor), and the same with
three glass panes (alpha 0.3) around the sensor covering ~40 % of the beams ("glass"). Sensors, each timed alone (GPU
time of command buffers holding only its kernel, median of 20, after the acceleration structures are built): a
16 x 360 lidar (default kernel; transparency modes), the same with two returns, a 64 x 64 depth camera (modes).
Run each env count alone (scripts/gpu_run.sh, kind timing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

BASE = """
<mujoco>
  <worldbody>
    <camera name="cam" pos="0 0 0.35" xyaxes="0 -1 0 0 0 1" fovy="70"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="sensor" pos="0 0 0.3"><joint type="free"/>
      <geom name="s0" type="sphere" size="0.02" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      <site name="lidar" pos="0 0 0.05"/></body>
    <body pos="0.8 0.1 0.2"><joint type="free"/><geom name="b1" type="box" size="0.1 0.15 0.2" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="-0.5 0.6 0.25"><joint type="free"/><geom name="c1" type="cylinder" size="0.12 0.25" rgba="0.2 0.2 0.9 1"/></body>
    <body pos="0.2 -0.9 0.3"><joint type="free"/><geom name="k1" type="capsule" size="0.08 0.2" rgba="0.9 0.9 0.2 1"/></body>
    <body pos="-0.7 -0.4 0.15"><joint type="free"/><geom name="sp" type="sphere" size="0.15" rgba="0.2 0.9 0.2 1"/></body>
    <geom name="wall" type="box" pos="2.0 0 0.5" size="0.05 2 0.5" rgba="0.8 0.8 0.8 1"/>
    PANES
  </worldbody>
</mujoco>"""
PANES = """<geom name="pane1" type="box" pos="0.5 0 0.4" size="0.004 0.6 0.4" rgba="0.6 0.8 0.9 0.3"/>
    <geom name="pane2" type="box" pos="0 0.5 0.4" size="0.6 0.004 0.4" rgba="0.6 0.8 0.9 0.3"/>
    <geom name="pane3" type="box" pos="-0.3 -0.3 0.4" size="0.004 0.4 0.4" euler="0 0 45" rgba="0.6 0.8 0.9 0.3"/>"""


def gpu_ms(rt, sensor, reps=20):
    c = rt.ctx
    t = []
    for _ in range(reps + 2):
        cb = c.command_buffer()
        sensor.encode(cb, sim=None)
        cb.commit(); cb.waitUntilCompleted()
        if cb.error() is not None:
            raise RuntimeError(cb.error())
        t.append((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3)
    return float(np.median(t[2:]))


def main(n, out):
    import mujoco
    import warp as wp
    wp.config.quiet = True
    from metalsim.sensors.raytrace import RayTracer
    A, E = np.meshgrid(np.linspace(-np.pi, np.pi, 360, endpoint=False), np.deg2rad(np.linspace(-15, 15, 16)))
    az, el = A.ravel().astype(np.float32), E.ravel().astype(np.float32)
    res = {"n_envs": n, "beams": int(az.size), "depth": "64x64"}
    for scene, xml in (("plain", BASE.replace("PANES", "")), ("glass", BASE.replace("PANES", PANES))):
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        datas = [d] * n
        rt = RayTracer(m, n, max_range=8.0)
        t0 = time.perf_counter()
        rt.trace_host(datas, [])                                 # builds and refits the structures
        res[f"{scene}/build_s"] = time.perf_counter() - t0
        cfgs = [("lidar", dict()), ("lidar_r2", dict(max_returns=2))]
        if scene == "glass":
            cfgs += [("lidar_surface", dict(transparency="surface")), ("lidar_behind", dict(transparency="behind")),
                     ("lidar_opaque", dict(transparency="opaque")), ("lidar_r2_dual", dict(max_returns=2, return_mode="strongest_last"))]
        for name, kw in cfgs:
            s = rt.make_lidar("lidar", az, el, **kw)
            rt.trace_host(datas, [s])
            o = s.numpy()
            res[f"{scene}/{name}_ms"] = gpu_ms(rt, s)
            res[f"{scene}/{name}_hit_frac"] = float((np.asarray(o["range"]) > 0).mean())
            print(f"[cost] n {n} {scene} {name}: {res[f'{scene}/{name}_ms']:.3f} ms (tr={s.tr}, ext={s.extended})", flush=True)
            del s, o
        dcfg = [("depth", dict())] + ([("depth_behind", dict(transparent_ids="behind"))] if scene == "glass" else [])
        for name, kw in dcfg:
            s = rt.make_depth_camera("cam", 64, 64, **kw)
            rt.trace_host(datas, [s])
            res[f"{scene}/{name}_ms"] = gpu_ms(rt, s)
            print(f"[cost] n {n} {scene} {name}: {res[f'{scene}/{name}_ms']:.3f} ms (tr={s.lmat is not None})", flush=True)
            del s
        del rt
    print(json.dumps(res, indent=1))
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    main(a.n, a.out)
