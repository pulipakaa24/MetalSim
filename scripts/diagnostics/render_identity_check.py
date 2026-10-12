"""Byte-identity of renders across two checkouts (the default-off guarantee of a renderer change).

    # in each checkout (PYTHONPATH pointing at it), through the GPU queue:
    python scripts/diagnostics/render_identity_check.py render --out base.npz
    python scripts/diagnostics/render_identity_check.py render --out new.npz
    python scripts/diagnostics/render_identity_check.py compare base.npz new.npz

``render`` draws a fixed set of scenes (repo assets, synthetic MJCF, MuJoCo Menagerie) with every renderer at its
defaults (tier 0 RGB / depth / segmentation / normals, tier 1, tier 2 at 4 spp, the ray-traced depth camera and a
lidar) from MjData states and saves every output array; ``compare`` reports, per scene and output, whether the bytes
are identical (and the differing pixel fraction when not). Only public constructor arguments that existed before the
change are used, so the same script runs on both checkouts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MENAGERIE = os.environ.get("MENAGERIE", "/Users/aditya/robosim/upstream/mujoco_menagerie")

PRIMS = """<mujoco><worldbody><light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
  <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/><geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
  <body pos="0 0 0.3"><joint type="free"/><geom type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
  <body pos="0.15 0.1 0.2"><joint type="free"/><geom type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/></body>
  <body pos="-0.15 -0.1 0.25"><joint type="free"/><geom type="cylinder" size="0.04 0.08" rgba="0.2 0.2 0.9 1"/></body>
  <body pos="0.05 -0.2 0.2"><joint type="free"/><geom type="capsule" size="0.03 0.06" rgba="0.9 0.9 0.2 1"/>
  <site name="hidden_site" group="3" size="0.05"/><site name="clear_site" size="0.05" rgba="1 0 0 0"/></body>
  </worldbody></mujoco>"""

TEXTURED = """<mujoco><asset>
  <texture name="checker" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.9 0.9 0.9" width="64" height="64"/>
  <material name="checker" texture="checker" texrepeat="4 4"/>
  <hfield name="hf" nrow="8" ncol="8" size="0.3 0.3 0.1 0.02"
    elevation="0 0 0 0 0 0 0 0  0 1 1 1 1 1 1 0  0 1 2 2 2 2 1 0  0 1 2 3 3 2 1 0  0 1 2 3 3 2 1 0  0 1 2 2 2 2 1 0  0 1 1 1 1 1 1 0  0 0 0 0 0 0 0 0"/>
  </asset><worldbody><light directional="true" pos="0 0 3" dir="0 0 -1"/><light pos="0.3 0 1" dir="0 0 -1" cutoff="40"/>
  <camera name="cam" pos="0 -0.9 0.9" xyaxes="1 0 0 0 0.707 0.707"/>
  <geom name="floor" type="plane" size="1 1 0.1" material="checker"/>
  <geom type="hfield" hfield="hf" pos="0.3 0.2 0"/>
  <geom name="b1" type="box" pos="0 0 0.15" size="0.15 0.15 0.15" rgba="0.8 0.5 0.2 1"/></worldbody></mujoco>"""


def scenes():
    out = [("prims", mujoco.MjModel.from_xml_string(PRIMS)), ("textured_hfield", mujoco.MjModel.from_xml_string(TEXTURED))]
    if os.environ.get("IDENTITY_EXTRAS"):       # a scene with flexes, a skin, tendons and sites (tests/test_render_extras.py)
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from test_render_extras import EXTRAS, DRAPE
        out += [("extras", mujoco.MjModel.from_xml_string(EXTRAS)), ("drape", mujoco.MjModel.from_xml_string(DRAPE))]
    so = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")
    if os.path.exists(so):
        out.append(("so101", mujoco.MjModel.from_xml_path(so)))
    for name, rel in (("panda", "franka_emika_panda/scene.xml"), ("shadow_hand", "shadow_hand/scene_right.xml"),
                      ("go2_with_site", "unitree_go2/scene.xml")):
        p = os.path.join(MENAGERIE, rel)
        if os.path.exists(p):
            spec = mujoco.MjSpec.from_file(p)
            spec.worldbody.add_camera(name="idcam", pos=[0.9, -0.9, 0.7], xyaxes=[0.707, 0.707, 0, -0.35, 0.35, 0.87])
            out.append((name, spec.compile()))
    return out


def render(out_path):
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    from metalsim.sensors.raytrace import RayTracer
    arrays = {}
    for name, m in scenes():
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        for _ in range(5):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        cam = m.ncam - 1
        datas = [d, d]
        r0 = Tier0Renderer(m, 2, width=96, height=72, camera=cam, outputs=("rgb", "depth", "seg", "normal"))
        for k, v in r0.render_host(datas).items():
            arrays[f"{name}/tier0/{k}"] = v
        r1 = Tier0Renderer(m, 2, width=96, height=72, camera=cam, outputs=("rgb", "depth", "seg"), tier=1)
        for k, v in r1.render_host(datas).items():
            arrays[f"{name}/tier1/{k}"] = v
        r2 = Tier2Renderer(m, 2, width=64, height=48, camera=cam, spp=4, max_bounces=2)
        o2 = r2.render_host(datas)
        for k in ("rgb", "depth", "seg", "normal"):
            arrays[f"{name}/tier2/{k}"] = o2[k]
        rt = RayTracer(m, 2, max_range=20.0)
        dc = rt.make_depth_camera(cam, 64, 48)
        sensors = [dc]
        if m.nsite:
            az, el = np.meshgrid(np.linspace(-np.pi, np.pi, 64, endpoint=False), np.linspace(-0.3, 0.3, 8))
            sensors.append(rt.make_lidar(0, az, el))
        o = rt.trace_host(datas, sensors)
        arrays[f"{name}/raydepth"] = o["ray_depth"]
        if "lidar" in o:
            for k, v in o["lidar"].items():
                arrays[f"{name}/lidar/{k}"] = v
        print(name, "rendered", flush=True)
    np.savez_compressed(out_path, **arrays)


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    res = {}
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            res[k] = "missing"
            continue
        x, y = a[k], b[k]
        if x.shape == y.shape and x.tobytes() == y.tobytes():
            res[k] = "identical"
        else:
            diff = (x != y).reshape(x.shape[0], -1).mean() if x.shape == y.shape else float("nan")
            mx = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.shape == y.shape else float("nan")
            res[k] = f"DIFFERENT ({diff:.4f} of values, max abs difference {mx:.3g})"
    for k, v in res.items():
        print(f"{k}: {v}")
    n_same = sum(v == "identical" for v in res.values())
    print(json.dumps({"identical": n_same, "total": len(res)}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("render", "compare"))
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", default="identity.npz")
    a = ap.parse_args()
    if a.mode == "render":
        render(a.out)
    else:
        compare(*a.paths)


if __name__ == "__main__":
    sys.exit(main())
