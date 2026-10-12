"""Scenes shared by the Newton bridge tests (synthetic MJCF and the SO-101; no Metal imports)."""
import os

import mujoco
import numpy as np

from metalsim.render.newton_bridge import Camera, Light, lookat_quat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRIMS = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.08 0 0  0 0.08 0  0 0 0.08  0.08 0.08 0.08"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
    <camera name="cam" pos="0.7 -0.7 0.6" xyaxes="0.707 0.707 0 -0.4 0.4 0.82"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.3 0.3 0.35 1"/>
    <geom name="post" type="box" pos="-0.3 0.3 0.15" size="0.03 0.03 0.15" rgba="0.6 0.6 0.6 1"/>
    <body name="box" pos="0 0 0.3"><joint name="j_box" type="free"/><geom type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
    <body name="ball" pos="0.15 0.1 0.2"><joint name="j_ball" type="free"/><geom type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/></body>
    <body name="cyl" pos="-0.15 -0.1 0.25"><joint name="j_cyl" type="free"/><geom type="cylinder" size="0.04 0.08" rgba="0.2 0.2 0.9 1"/></body>
    <body name="cap" pos="0.05 -0.2 0.2"><joint name="j_cap" type="free"/><geom type="capsule" size="0.03 0.06" rgba="0.9 0.9 0.2 1"/></body>
    <body name="ell" pos="-0.2 0.15 0.2"><joint name="j_ell" type="free"/><geom type="ellipsoid" size="0.07 0.04 0.03" rgba="0.8 0.3 0.8 1"/></body>
    <body name="msh" pos="0.2 -0.05 0.25"><joint name="j_msh" type="free"/><geom type="mesh" mesh="tet" rgba="0.3 0.8 0.8 1"/></body>
    <body name="arm0" pos="0.25 0.25 0.05">
      <joint name="j_a0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" rgba="0.7 0.5 0.2 1"/>
      <body name="arm1" pos="0 0 0.2">
        <joint name="j_a1" type="hinge" axis="0 1 0"/>
        <geom type="box" pos="0.08 0 0" size="0.08 0.015 0.015" rgba="0.5 0.7 0.2 1"/>
        <site name="tip" pos="0.16 0 0" group="4"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def so101_xml():
    spec = mujoco.MjSpec.from_file(os.path.join(ROOT, "assets", "so101", "scene.xml"))
    spec.meshdir = os.path.join(ROOT, "assets", "so101", "assets")
    p = (0.45, -0.35, 0.35)
    spec.worldbody.add_camera(name="overview", pos=list(p), quat=list(lookat_quat(p, (0.1, 0.0, 0.12))), fovy=50.0)
    return spec.to_xml()


def mirror(mj):
    """``Camera`` / ``Light`` lists reproducing the MJCF's cameras and lights on a Newton-shape twin."""
    cams = []
    for c in range(mj.ncam):
        body = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(mj.cam_bodyid[c])) if mj.cam_bodyid[c] else None
        kw = {}
        if mj.cam_sensorsize[c][0] > 0:
            kw = dict(sensorsize=tuple(mj.cam_sensorsize[c]), focal=tuple(mj.cam_intrinsic[c][:2]),
                      principal=tuple(mj.cam_intrinsic[c][2:4]), resolution=tuple(mj.cam_resolution[c]))
        cams.append(Camera(name=mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_CAMERA, c), pos=tuple(mj.cam_pos[c]),
                           quat=tuple(mj.cam_quat[c]), lookat=None, fovy=float(mj.cam_fovy[c]), body=body, **kw))
    lights = [Light(pos=tuple(mj.light_pos[i]), dir=tuple(mj.light_dir[i]),
                    directional=int(mj.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
                    diffuse=tuple(mj.light_diffuse[i]), specular=tuple(mj.light_specular[i]),
                    ambient=tuple(mj.light_ambient[i]), castshadow=bool(mj.light_castshadow[i])) for i in range(mj.nlight)]
    return cams, lights


def random_qpos(mj, n, seed):
    rng = np.random.default_rng(seed)
    out = np.tile(mj.qpos0, (n, 1))
    for e in range(n):
        for j in range(mj.njnt):
            a = int(mj.jnt_qposadr[j]); t = int(mj.jnt_type[j])
            if t == mujoco.mjtJoint.mjJNT_FREE:
                out[e, a:a + 3] += rng.uniform(-0.05, 0.05, 3)
                q = rng.normal(size=4); out[e, a + 3:a + 7] = q / np.linalg.norm(q)
            elif t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                lo, hi = mj.jnt_range[j] if mj.jnt_limited[j] else (-0.8, 0.8)
                out[e, a] = rng.uniform(max(lo, -0.8), min(hi, 0.8))
    return out
