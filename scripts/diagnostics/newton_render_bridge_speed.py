"""Cost of the Newton -> renderer bridge per frame: the pose kernels (``NewtonRenderSource.update``, graph replay)
against the tier-0 render they feed, for the SO-101 scene at several world counts.

    scripts/gpu_run.sh newton_bridge_speed timing 15 -- .venv/bin/python scripts/diagnostics/newton_render_bridge_speed.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np
import warp as wp

import newton

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from newton_bridge_scenes import mirror, so101_xml  # noqa: E402

from metalsim.render.newton_bridge import NewtonRenderSource  # noqa: E402
from metalsim.render.tier0 import Tier0Renderer  # noqa: E402


def bench(fn, sync, iters):
    for _ in range(5):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="16,256,1024")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--out", default="runs/newton_bridge/speed.json")
    a = ap.parse_args()
    xml = so101_xml()
    mj = mujoco.MjModel.from_xml_string(xml)
    cams, lights = mirror(mj)
    rows = []
    for n in [int(x) for x in a.worlds.split(",")]:
        env = newton.ModelBuilder(); env.add_mjcf(xml)
        b = newton.ModelBuilder(); b.replicate(env, n, spacing=(1.0, 1.0, 0.0))
        t0 = time.perf_counter()
        model = b.finalize(device="metal:0")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        src = NewtonRenderSource(model, cameras=cams, lights=lights)
        t_build = time.perf_counter() - t0
        r = Tier0Renderer(src.m, n, width=a.res, height=a.res, camera="overview", outputs=("rgb", "depth", "seg"))
        sync = lambda: wp.synchronize_device("metal:0")

        def upd():
            src.update(state)

        def both():
            r.render(src, src.update(state))

        def rsync():
            r.ctx.synchronize()

        ms_update = bench(upd, sync, 200)
        ms_render = bench(lambda: r.render(src, src.event.value), rsync, 50)
        ms_both = bench(both, rsync, 50)
        row = dict(worlds=n, geoms=int(src.m.ngeom), slots=int(r.G), build_s=round(t_build, 2),
                   update_ms=round(ms_update, 4), render_ms=round(ms_render, 3), update_plus_render_ms=round(ms_both, 3))
        print(json.dumps(row))
        rows.append(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"newton": newton.__version__, "res": a.res, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
