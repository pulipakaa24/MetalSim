"""Where the ray-traced cost of deformables goes: acceleration-structure work alone vs. ray traversal, by block size and
build policy. Cloth scene of deformable_render_bench.py at --envs, tier 1 and tier 2, ms per frame.

    scripts/gpu_run.sh deform_profile timing 15 -- .venv/bin/python scripts/diagnostics/deformable_render_profile.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deformable_render_bench import cloth_scene, make_sim, timed   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--res", type=int, default=64)
    a = ap.parse_args()
    import warp as wp
    wp.config.quiet = True
    from metalsim.interop import warp_metal as wm
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    m, kind = cloth_scene()
    n = a.envs
    sim = make_sim(m, kind, n)
    gx, gm = wm.buffer_of(sim.d.geom_xpos), wm.buffer_of(sim.d.geom_xmat)
    for which in ("tier1", "tier2"):
        for label, kw in (("geoms_only", dict(visual=False)), ("blocks_auto", {}), ("blocks_1env", dict(dyn_block_envs=1)),
                          ("one_structure", dict(dyn_block_envs=n))):
            if which == "tier1":
                if "dyn_block_envs" in kw:
                    continue          # Tier0Renderer builds its tracer with the default blocks
                r = Tier0Renderer(m, n, width=a.res, height=a.res, camera="cam", outputs=("rgb",), tier=1, **kw)
                rt = r.rt
            else:
                r = Tier2Renderer(m, n, width=a.res, height=a.res, camera="cam", spp=1, max_bounces=1, **kw)
                rt = r.rt
            row = {"which": which, "variant": label, "envs": n}
            row["render_ms"] = round(timed(lambda: r.render(sim), a.frames, r.ctx.synchronize), 3)

            def as_only():
                cb = r.ctx.command_buffer()
                rt._encode_refit_and_build(cb, gx.buffer, gx.offset, gm.buffer, gm.offset)
                r.ctx.commit(cb)
            row["as_only_ms"] = round(timed(as_only, a.frames, r.ctx.synchronize), 3)
            if rt.dyn is not None:
                row["structures"] = len(rt.dyn_as)
                for pol, every in (("refit", 0), ("rebuild", 1)):
                    rt.dyn_rebuild_every = every
                    rt._dyn_built = False
                    row[f"render_{pol}_ms"] = round(timed(lambda: r.render(sim), a.frames, r.ctx.synchronize), 3)
                    row[f"as_{pol}_ms"] = round(timed(as_only, a.frames, r.ctx.synchronize), 3)
            print(json.dumps(row), flush=True)
            if label == "blocks_auto" and which == "tier1":
                twin = static_twin(r)                # the same surfaces as static mesh geoms: the cost of the triangles
            del r
        rt_twin = (Tier0Renderer(twin[0], n, width=a.res, height=a.res, camera="cam", outputs=("rgb",), tier=1, visual=False)
                   if which == "tier1" else
                   Tier2Renderer(twin[0], n, width=a.res, height=a.res, camera="cam", spp=1, max_bounces=1, visual=False))
        print(json.dumps({"which": which, "variant": "static_mesh_twin", "envs": n,
                          "render_ms": round(timed(lambda: rt_twin.render(twin[1]), a.frames, rt_twin.ctx.synchronize), 3)}), flush=True)


def static_twin(r):
    """The scene with the dynamic objects' current env-0 surfaces baked into static mesh geoms (rigid, instanced per env),
    and a BatchSim of it: the same triangles without the dynamic path."""
    import mujoco
    import numpy as np
    from metalsim.physics import deformable as dfm
    from metalsim.physics.batch import BatchSim
    from deformable_render_bench import CAM
    xml = dfm.box_scene_xml(None, None).replace("<worldbody>", "<worldbody>" + CAM, 1)
    spec = mujoco.MjSpec.from_string(xml)
    dyn = r.rt.dyn if r.rt is not None else r.dyn
    r.ctx.synchronize()
    rec = dyn.records()
    for k, ob in enumerate(dyn.objects):
        v = rec[ob["obj_base"]: ob["obj_base"] + ob["nvert"], :3].astype(np.float64)   # env 0: no grid offset
        mesh = spec.add_mesh(name=f"twin{k}")
        mesh.uservert = v.ravel().tolist()
        mesh.userface = ob["tris"].ravel().tolist()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"twin{k}", contype=0, conaffinity=0,
                                rgba=[0.2, 0.5, 0.8, 1])
    m = spec.compile()
    sim = BatchSim(m, r.n)
    sim.step(); sim.synchronize()
    return m, sim


if __name__ == "__main__":
    main()
