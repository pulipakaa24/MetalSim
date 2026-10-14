"""Cost of drawing deformables / tendons in the batched renderers (metalsim.render.dynamic_mesh), per frame.

For each scene and env count, renders the GPU path straight from the simulator's buffers and reports milliseconds per
frame (``--frames`` renders committed back to back, one synchronize) for: the scene with its extra objects (MuJoCo's
default visual option) and the same renderer with the extras off (``visual=False``: the deformables invisible, the
previous behaviour), so the difference is the price of drawing them; plus the vertex generation alone and, for the ray
tracers, a refit-only and a rebuild-every-frame variant of the dynamic acceleration structures.

Scenes (synthetic MJCF and MuJoCo Menagerie only):
  cloth     MuJoCo Warp flex (DeformableSim): 10x10 cloth + 12-vertex cable + 4x4x4 soft cube dropping on a box
  hand      Menagerie tetheria_aero_hand_open (20 spatial tendons) on BatchSim

    scripts/gpu_run.sh deform_bench timing 20 -- .venv/bin/python scripts/diagnostics/deformable_render_bench.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

EKW = {}   # tube_slices / cap_stacks for every renderer (--tube-slices, --cap-stacks)
CAM = '<camera name="cam" pos="0 -1.1 0.75" xyaxes="1 0 0 0 0.55 0.835"/>'


def cloth_scene():
    from metalsim.physics import deformable as dfm
    xml = dfm.box_scene_xml(dfm.ClothCfg(), dfm.CableCfg(), soft=dfm.SoftCfg())
    xml = xml.replace("<worldbody>", "<worldbody>" + CAM, 1)
    m = mujoco.MjModel.from_xml_string(xml)
    return m, "deformable"


def hand_scene():
    p = os.path.join(ROOT, "upstream", "mujoco_menagerie", "tetheria_aero_hand_open", "scene_right.xml")
    for cand in (os.environ.get("MENAGERIE", ""), os.path.join(ROOT, "upstream", "mujoco_menagerie")):
        q = os.path.join(cand, "tetheria_aero_hand_open", "scene_right.xml")
        if cand and os.path.exists(q):
            p = q
            break
    spec = mujoco.MjSpec.from_file(p)
    spec.worldbody.add_camera(name="cam", pos=[0.0, -0.45, 0.25], xyaxes=[1, 0, 0, 0, 0.45, 0.89])
    return spec.compile(), "batch"


def make_sim(m, kind, n):
    if kind == "deformable":
        from metalsim.physics.deformable import DeformableSim
        sim = DeformableSim(m, n)
        sim.randomize(seed=0)
        for _ in range(30):
            sim.step()
        sim.synchronize()
        return sim
    from metalsim.physics.batch import BatchSim
    sim = BatchSim(m, n)
    sim.step()
    sim.synchronize()
    return sim


def timed(fn, frames, sync):
    fn(); sync()                                  # warm-up (pipelines, first build)
    fn(); sync()
    t0 = time.perf_counter()
    for _ in range(frames):
        fn()
    sync()
    return 1e3 * (time.perf_counter() - t0) / frames


def bench(scene, n, frames, res, renderers):
    m, kind = scene()
    sim = make_sim(m, kind, n)
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    from metalsim.sensors.raytrace import RayTracer
    out = {"scene": scene.__name__, "envs": n, "res": res, "frames": frames}
    for which in renderers:
        row = {}
        for label, vis in (("with_extras", None), ("geoms_only", False)):
            kw = {}
            if which in ("tier0", "tier1"):
                r = Tier0Renderer(m, n, width=res, height=res, camera="cam", outputs=("rgb", "depth", "seg"), tier=0 if which == "tier0" else 1, visual=vis, **EKW)
                fn = lambda: r.render(sim)
            elif which == "tier2":
                r = Tier2Renderer(m, n, width=res, height=res, camera="cam", spp=1, max_bounces=1, visual=vis, **EKW)
                fn = lambda: r.render(sim)
            else:
                from metalsim.render.dynamic_mesh import camera_view
                from metalsim.render.scene_tables import CameraIntrinsics
                ci = CameraIntrinsics.from_model(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam"), res, res)
                tv = None if (EKW.get("tube_slices") or EKW.get("cap_stacks")) else camera_view(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam"), ci.fy)
                r = RayTracer(m, n, max_range=20.0, visual=mujoco.MjvOption() if vis is None else vis, tube_view=tv, **EKW)
                cam = r.make_depth_camera("cam", res, res)
                fn = lambda: r.trace(sim, [cam])
            row[label] = round(timed(fn, frames, r.ctx.synchronize), 3)
            if vis is None:
                t = r.tables
                dyn = getattr(r, "dyn", None) or getattr(getattr(r, "rt", None), "dyn", None)
                row["dyn_tris_per_env"] = int(sum(len(o["tris"]) for o in (dyn.objects if dyn is not None else [])))
                row["extras"] = [e["kind"] for e in t.extras]

                if dyn is not None:
                    src = r._dyn_src if which in ("tier0", "raydepth") else r.rt._dyn_src

                    def gen():
                        cb = r.ctx.command_buffer()
                        dyn.encode(cb, src)
                        r.ctx.commit(cb)
                    row["generate_only"] = round(timed(gen, frames, r.ctx.synchronize), 3)
                rt = r if which == "raydepth" else getattr(r, "rt", None)
                if rt is not None and rt.dyn is not None:
                    for pol, every in (("refit_only", 0), ("rebuild_every_frame", 1)):
                        rt.dyn_rebuild_every = every
                        rt._dyn_built = False
                        row[pol] = round(timed(fn, frames, r.ctx.synchronize), 3)
                    rt.dyn_rebuild_every = 32
        row["overhead_ms"] = round(row["with_extras"] - row["geoms_only"], 3)
        out[which] = row
        print(json.dumps({"scene": out["scene"], "envs": n, which: row}), flush=True)
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--scenes", nargs="+", default=["cloth", "hand"])
    ap.add_argument("--renderers", nargs="+", default=["tier0", "tier1", "raydepth", "tier2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tube-slices", type=int, default=None)
    ap.add_argument("--cap-stacks", type=int, default=None)
    a = ap.parse_args()
    import warp as wp
    wp.config.quiet = True
    EKW.update(tube_slices=a.tube_slices, cap_stacks=a.cap_stacks)
    rows = []
    for s in a.scenes:
        for n in a.envs:
            rend = [r for r in a.renderers if not (r == "tier2" and n > 1024)]
            rows.append(bench({"cloth": cloth_scene, "hand": hand_scene}[s], n, a.frames, a.res, rend))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
