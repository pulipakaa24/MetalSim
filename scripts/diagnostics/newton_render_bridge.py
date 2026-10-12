"""Newton -> renderer bridge coverage: a scene imported into Newton (``add_mjcf``) and rendered through
``metalsim.render.newton_bridge`` against the same MJCF rendered through the MuJoCo path, at the same state.

Per scene, twin (Newton shapes / source MJCF) and renderer (tier 0 raster, ray-traced depth camera): the minimum over
bodies and envs of the coverage of the MuJoCo-path silhouette (body segmentation), foreground IoU, depth error on the
common foreground, RGB mean absolute difference (appearance: the Newton-shape twin carries colours only). Worlds are
replicated 2 m apart (env origins exercised). Scenes: synthetic MJCF, the SO-101 (assets/so101), and Menagerie's G1
and Panda when a mujoco_menagerie checkout is found (``--menagerie``, default upstream/mujoco_menagerie).

    scripts/gpu_run.sh newton_bridge check 15 -- .venv/bin/python scripts/diagnostics/newton_render_bridge.py --out runs/newton_bridge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

import newton

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from newton_bridge_scenes import PRIMS, mirror, random_qpos, so101_xml  # noqa: E402

from metalsim.render.newton_bridge import NewtonRenderSource, lookat_quat, mujoco_qpos_to_joint_q  # noqa: E402
from metalsim.render.tier0 import SEG_BODY, Tier0Renderer  # noqa: E402
from metalsim.sensors.raytrace import RayTracer  # noqa: E402


def with_overview(path: str, pos, target, fovy=50.0) -> str:
    """An MJCF file with a world camera "overview" added (string, absolute asset dirs)."""
    spec = mujoco.MjSpec.from_file(path)
    base = os.path.dirname(os.path.abspath(path))
    spec.meshdir = os.path.join(base, spec.meshdir or "")
    spec.texturedir = os.path.join(base, spec.texturedir or "")
    spec.worldbody.add_camera(name="overview", pos=list(pos), quat=list(lookat_quat(pos, target)), fovy=fovy)
    return spec.to_xml()


def coverage(ref, got, ids):
    covs = []
    for a, b in zip(*ids):
        m = ref["seg"] == a
        if m.sum() >= 10:
            covs.append(float((got["seg"][m] == b).mean()))
    fr, fg = ref["seg"] > 0, got["seg"] > 0
    both = fr & fg & (ref["depth"] > 0) & (got["depth"] > 0)
    dz = np.abs(ref["depth"][both] - got["depth"][both]) * 1000
    out = {"min_body_coverage": min(covs), "fg_iou": float((fr & fg).sum() / max((fr | fg).sum(), 1)),
           "depth_median_mm": float(np.median(dz)) if dz.size else 0.0, "depth_p99_mm": float(np.percentile(dz, 99)) if dz.size else 0.0}
    if "rgb" in ref:
        out["rgb_mad"] = float(np.abs(ref["rgb"].astype(int) - got["rgb"].astype(int)).mean())
    return out


def depth_cov(ref, got):
    hr, hg = ref > 0, got > 0
    both = hr & hg
    return {"hit_agreement": float((hr == hg).mean()),
            "depth_p99_mm": float(np.percentile(np.abs(ref[both] - got[both]), 99) * 1000) if both.any() else 0.0}


def run_scene(name, xml, n, cam, out_dir, res=(160, 120), seeds=3, spacing=2.0):
    mj = mujoco.MjModel.from_xml_string(xml)
    env = newton.ModelBuilder(); env.add_mjcf(xml)
    b = newton.ModelBuilder(); b.replicate(env, n, spacing=(spacing, spacing, 0.0))
    model = b.finalize(device="metal:0")
    state = model.state()
    W, H = res
    results = {"worlds": n, "camera": cam}
    for twin in ("newton", "mjcf"):
        t0 = time.perf_counter()
        if twin == "newton":
            cams, lights = mirror(mj)
            src = NewtonRenderSource(model, cameras=cams, lights=lights)
        else:
            src = NewtonRenderSource(model, twin=mj)
        t_build = time.perf_counter() - t0
        rs = Tier0Renderer(src.m, n, width=W, height=H, camera=cam, outputs=("rgb", "depth", "seg"), seg_mode=SEG_BODY)
        rr = Tier0Renderer(mj, n, width=W, height=H, camera=cam, outputs=("rgb", "depth", "seg"), seg_mode=SEG_BODY)
        ts = RayTracer(src.m, n, max_range=20.0); cs = ts.make_depth_camera(cam, W, H)
        tr = RayTracer(mj, n, max_range=20.0); cr = tr.make_depth_camera(cam, W, H)
        names = [mujoco.mj_id2name(src.m, mujoco.mjtObj.mjOBJ_BODY, k) for k in range(src.m.nbody)]
        ids = ([], [])
        for k in range(mj.nbody):
            nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, k)
            if k == 0 or nm in names:
                ids[0].append(k + 1); ids[1].append(1 if k == 0 else names.index(nm) + 1)
        recs, rrecs = [], []
        for seed in range(seeds):
            qpos = random_qpos(mj, n, seed)
            datas = []
            for q in qpos:
                d = mujoco.MjData(mj); d.qpos[:] = q; mujoco.mj_forward(mj, d); datas.append(d)
            model.joint_q.assign(mujoco_qpos_to_joint_q(mj, qpos, model, offsets=src.origins))
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            v = src.update(state)
            rs.render(src, v); ts.trace(src, [cs], v); rs.ctx.synchronize(); ts.ctx.synchronize()
            got = {k: a.numpy().copy() for k, a in rs._out_arrays.items()}
            ref = rr.render_host(datas)
            rref = tr.trace_host(datas, [cr])["ray_depth"]
            rgot = cs.numpy()
            for e in range(n):
                recs.append(coverage({k: x[e] for k, x in ref.items()}, {k: x[e] for k, x in got.items()}, ids))
                rrecs.append(depth_cov(rref[e], rgot[e]))
                if seed == 0 and e == n - 1 and out_dir:
                    from PIL import Image
                    Image.fromarray(np.concatenate([ref["rgb"][e], got["rgb"][e]], 1)).save(os.path.join(out_dir, f"{name}_{twin}.png"))
        worst = lambda rs_, k: (min if ("cov" in k or "iou" in k or "agree" in k) else max)(r[k] for r in rs_)
        results[twin] = {"tier0": {k: worst(recs, k) for k in recs[0]}, "raydepth": {k: worst(rrecs, k) for k in rrecs[0]},
                         "twin_geoms": int(src.m.ngeom), "drawn_slots": int(rs.G), "build_s": round(t_build, 3)}
        print(name, twin, json.dumps(results[twin]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/newton_bridge")
    ap.add_argument("--menagerie", default=os.path.join(os.path.dirname(ROOT.rstrip("/")), "upstream", "mujoco_menagerie"))
    a = ap.parse_args()
    if not os.path.isdir(a.menagerie):
        a.menagerie = "/Users/aditya/robosim/upstream/mujoco_menagerie"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"newton": newton.__version__, "mujoco": mujoco.__version__}
    res["prims"] = run_scene("prims", PRIMS, 4, "cam", a.out)
    res["so101_overview"] = run_scene("so101_overview", so101_xml(), 2, "overview", a.out)
    res["so101_wrist"] = run_scene("so101_wrist", so101_xml(), 2, "wrist_cam", a.out)
    g1 = os.path.join(a.menagerie, "unitree_g1", "scene.xml")
    if os.path.exists(g1):
        res["menagerie_g1"] = run_scene("menagerie_g1", with_overview(g1, (2.2, -1.6, 1.4), (0.0, 0.0, 0.7)), 2, "overview", a.out)
    panda = os.path.join(a.menagerie, "franka_emika_panda", "scene.xml")
    if os.path.exists(panda):
        res["menagerie_panda"] = run_scene("menagerie_panda", with_overview(panda, (1.4, -1.0, 1.0), (0.3, 0.0, 0.4)), 2, "overview", a.out)
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
