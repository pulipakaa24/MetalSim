"""Newton -> renderer bridge coverage: a scene imported into Newton (``add_mjcf``) and rendered through
``metalsim.render.newton_bridge`` against the same MJCF rendered through the MuJoCo path, at the same state.

Per scene and renderer: coverage of the MuJoCo-path silhouette per body (seg), foreground IoU, depth error on the
common foreground, RGB mean absolute difference; both twin kinds (Newton shapes, source MJCF). Synthetic MJCF and the
SO-101 (Menagerie-style asset in assets/so101) only.

    scripts/gpu_run.sh newton_bridge check 10 -- .venv/bin/python scripts/diagnostics/newton_render_bridge.py --out runs/newton_bridge
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import numpy as np
import warp as wp

import newton

from metalsim.render.newton_bridge import Camera, Light, NewtonRenderSource, lookat_quat, mujoco_qpos_to_joint_q
from metalsim.render.tier0 import SEG_BODY, Tier0Renderer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def so101_xml():
    """The SO-101 scene with a world overview camera added in front of the wrist camera (MJCF string, absolute meshdir)."""
    spec = mujoco.MjSpec.from_file(os.path.join(ROOT, "assets", "so101", "scene.xml"))
    spec.meshdir = os.path.join(ROOT, "assets", "so101", "assets")
    spec.worldbody.add_camera(name="overview", pos=[0.45, -0.35, 0.35], quat=list(lookat_quat((0.45, -0.35, 0.35), (0.1, 0.0, 0.12))),
                              fovy=50.0)
    return spec.to_xml()


def newton_from_mjcf(xml: str, n_worlds: int, spacing=0.0):
    """Newton model of ``n_worlds`` copies of the MJCF (a path or an XML string; one world each)."""
    env = newton.ModelBuilder()
    env.add_mjcf(xml)
    b = newton.ModelBuilder()
    if n_worlds == 1:
        b.add_world(env)
    else:
        b.replicate(env, n_worlds, spacing=(spacing, 0.0, 0.0))
    return b.finalize(device="metal:0")


def coverage(ref_seg, ref_depth, seg, depth, ref_ids, ids, rgb_ref=None, rgb=None):
    """Per-object coverage of the reference silhouette (same object id), foreground IoU, depth error (mm)."""
    fg_r, fg = ref_seg > 0, seg > 0
    covs = {}
    for k, (a, b) in enumerate(zip(ref_ids, ids)):
        m = ref_seg == a
        if m.sum() >= 10:
            covs[k] = float((seg[m] == b).mean())
    both = fg_r & fg & (ref_depth > 0) & (depth > 0)
    dz = np.abs(ref_depth[both] - depth[both]) * 1000
    out = {"fg_iou": float((fg_r & fg).sum() / max((fg_r | fg).sum(), 1)),
           "min_object_coverage": min(covs.values()) if covs else None,
           "mean_object_coverage": float(np.mean(list(covs.values()))) if covs else None,
           "depth_median_mm": float(np.median(dz)) if dz.size else None,
           "depth_p99_mm": float(np.percentile(dz, 99)) if dz.size else None}
    if rgb is not None:
        out["rgb_mad"] = float(np.abs(rgb.astype(int) - rgb_ref.astype(int)).mean())
    return out


def run_scene(name, xml, n_worlds, out_dir, res=(160, 120), seeds=3, from_file=False, cam=0):
    mj = mujoco.MjModel.from_xml_path(xml) if from_file else mujoco.MjModel.from_xml_string(xml)
    model = newton_from_mjcf(xml, n_worlds)
    state = model.state()
    W, H = res
    results = {}
    for twin_kind in ("newton", "mjcf"):
        if twin_kind == "newton":
            cams = [Camera(name=mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_CAMERA, c) or f"cam{c}",
                           pos=tuple(mj.cam_pos[c]), quat=tuple(mj.cam_quat[c]), lookat=None, fovy=float(mj.cam_fovy[c]),
                           body=mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(mj.cam_bodyid[c])) if mj.cam_bodyid[c] else None,
                           **(dict(sensorsize=tuple(mj.cam_sensorsize[c]), focal=tuple(mj.cam_intrinsic[c][:2]),
                                   principal=tuple(mj.cam_intrinsic[c][2:4]), resolution=tuple(mj.cam_resolution[c]))
                              if mj.cam_sensorsize[c][0] > 0 else {}))
                    for c in range(mj.ncam)]
            lights = [Light(pos=tuple(mj.light_pos[i]), dir=tuple(mj.light_dir[i]),
                            directional=int(mj.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
                            diffuse=tuple(mj.light_diffuse[i]), specular=tuple(mj.light_specular[i]),
                            ambient=tuple(mj.light_ambient[i]), castshadow=bool(mj.light_castshadow[i]))
                      for i in range(mj.nlight)]
            src = NewtonRenderSource(model, cameras=cams, lights=lights, env_origins=None)
        else:
            src = NewtonRenderSource(model, twin=mj, env_origins=None)
        if not mj.ncam:
            raise SystemExit("scene needs a camera")
        rs = Tier0Renderer(src.m, src.n, width=W, height=H, camera=cam, outputs=("rgb", "depth", "seg"), seg_mode=SEG_BODY)
        rr = Tier0Renderer(mj, src.n, width=W, height=H, camera=cam, outputs=("rgb", "depth", "seg"), seg_mode=SEG_BODY)
        # body ids: twin body name -> MuJoCo body id
        tb_names = [mujoco.mj_id2name(src.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(src.m.nbody)]
        recs = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            datas = []
            for e in range(src.n):
                d = mujoco.MjData(mj)
                d.qpos[:] = mj.qpos0
                for j in range(mj.njnt):
                    a = int(mj.jnt_qposadr[j])
                    if int(mj.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                        d.qpos[a:a + 3] += rng.uniform(-0.05, 0.05, 3)
                        q = rng.normal(size=4); d.qpos[a + 3:a + 7] = q / np.linalg.norm(q)
                    elif int(mj.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                        lo, hi = (mj.jnt_range[j] if mj.jnt_limited[j] else (-0.8, 0.8))
                        d.qpos[a] = rng.uniform(max(lo, -0.8), min(hi, 0.8))
                mujoco.mj_forward(mj, d)
                datas.append(d)
            model.joint_q.assign(mujoco_qpos_to_joint_q(mj, np.stack([d.qpos for d in datas]), model))
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            v = src.update(state)
            vr = rs.render(src, v); rs.ctx.synchronize()
            got = {k: a.numpy().copy() for k, a in rs._out_arrays.items()}
            ref = rr.render_host(datas)
            for e in range(src.n):
                ref_ids = [b + 1 for b in range(mj.nbody)]
                ids = []
                for b in range(mj.nbody):
                    nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b)
                    ids.append((tb_names.index(nm) + 1) if nm in tb_names else (1 if b == 0 else -99))
                c = coverage(ref["seg"][e], ref["depth"][e], got["seg"][e], got["depth"][e], ref_ids, ids, ref["rgb"][e], got["rgb"][e])
                recs.append(c)
                if seed == 0 and e == 0 and out_dir:
                    from PIL import Image
                    Image.fromarray(np.concatenate([ref["rgb"][e], got["rgb"][e]], 1)).save(os.path.join(out_dir, f"{name}_{twin_kind}.png"))
        agg = {k: (min(r[k] for r in recs if r[k] is not None) if "cov" in k or "iou" in k else
                   max(r[k] for r in recs if r[k] is not None)) for k in recs[0] if any(r[k] is not None for r in recs)}
        results[twin_kind] = agg
        print(name, twin_kind, json.dumps(agg))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/newton_bridge")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"newton": newton.__version__, "mujoco": mujoco.__version__}
    res["prims_1"] = run_scene("prims_1", PRIMS, 1, a.out)
    res["prims_4"] = run_scene("prims_4", PRIMS, 4, a.out)
    res["so101_2_overview"] = run_scene("so101_2_overview", so101_xml(), 2, a.out, cam="overview")
    res["so101_2_wrist"] = run_scene("so101_2_wrist", so101_xml(), 2, a.out, cam="wrist_cam")
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
