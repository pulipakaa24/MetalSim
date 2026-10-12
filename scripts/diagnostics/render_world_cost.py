"""Speed impact of the world-extent / per-environment render fixes (docs/RENDER_COVERAGE.md, workstream C).

ms per frame (median of ``--reps`` back-to-back GPU-path renders of a fixed physics state, one synchronize each):

  g1_flat   G1 flat task model (infinite ground plane) + a 128 x 128 head camera, tier 0, 1024 envs
  g1_rough  G1 rough task (heightfield + per-world terrain slots), same camera, tier 0, 1024 envs
  prims     the render_bench primitives scene, 64 x 64, tier 0, 4096 envs; tier 2 (1 spp, 2 bounces), 1024 envs

Each with the renderer's defaults, plus (when the renderer has them) far = 100 m and all fixes off. Runs on the code
before and after the fixes (keywords are passed only when accepted), so the same script gives both columns.

    scripts/gpu_run.sh render_world_cost timing 15 -- .venv/bin/python scripts/diagnostics/render_world_cost.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _kw(cls, **kw):
    ok = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kw.items() if k in ok}


OFF = dict(infinite_planes=False, hfield_skirts=False, per_world_size=False, model_fields=False, reverse_z=False)


def _time(r, sim, reps):
    import torch
    v = sim.event.value
    for _ in range(3):
        r.render(sim, v)
    r.ctx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r.render(sim, v); r.ctx.synchronize(); ts.append(time.perf_counter() - t0)
    torch.mps.synchronize()
    return round(1e3 * float(np.median(ts)), 3)


def g1_scene(kind, n):
    import mujoco
    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    if kind == "rough":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from terrain_render_cost import rough_scene
        m, sim, _, _ = rough_scene(n)
        return m, sim
    m, info = build_g1_model("flat", None)
    spec = info["spec"]
    tb = next(b for b in spec.bodies if b.name == "torso_link")
    a = np.deg2rad(40.0)
    cam = tb.add_camera(); cam.name = "head"; cam.pos = [0.1, 0.0, 0.35]; cam.fovy = 70.0
    x = np.array([0.0, -1.0, 0.0]); y = np.array([np.sin(a), 0.0, np.cos(a)]); R = np.stack([x, y, np.cross(x, y)], 1)
    qc = np.zeros(4); mujoco.mju_mat2Quat(qc, R.reshape(-1)); cam.quat = qc.tolist()
    m = spec.compile()
    sim = BatchSim(m, n, options=BatchSimOptions(njmax=256, nconmax=128))
    q = np.tile(m.key_qpos[0] if m.nkey else m.qpos0, (n, 1)).astype(np.float32)
    q[:, 0] = np.arange(n) % 32 * 2.0; q[:, 1] = np.arange(n) // 32 * 2.0
    sim.set_state(q, np.zeros((n, m.nv), np.float32)); sim.forward(); sim.synchronize()
    return m, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    import mujoco
    import warp as wp
    wp.config.quiet = True
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    from metalsim.physics.batch import BatchSim
    from metalsim.bench.render_bench import PRIMS
    res = {"far_kw": "far" in inspect.signature(Tier0Renderer.__init__).parameters}
    cfgs = {"default": {}, "far100": dict(far=100.0), "fixes_off": OFF, "no_skirts": dict(hfield_skirts=False),
            "no_sizes": dict(per_world_size=False), "no_sync": dict(model_fields=False), "standard_z": dict(reverse_z=False)}
    jobs = []
    for kind in ("flat", "rough"):
        jobs.append((f"g1_{kind}_t0_1024x128", lambda kind=kind: g1_scene(kind, 1024), Tier0Renderer, dict(width=128, height=128, camera="head")))
    pm = mujoco.MjModel.from_xml_string(PRIMS)

    def prims(n):
        s = BatchSim(pm, n); s.forward(); s.synchronize(); return pm, s
    jobs.append(("prims_t0_4096x64", lambda: prims(4096), Tier0Renderer, dict(width=64, height=64, camera="cam")))
    jobs.append(("prims_t2_1024x64", lambda: prims(1024), Tier2Renderer, dict(width=64, height=64, camera="cam", spp=1, max_bounces=2)))
    for name, mk, cls, base in jobs:
        if a.only and a.only not in name:
            continue
        m, sim = mk()
        row = {}
        for cname, extra in cfgs.items():
            if cname != "default" and not set(extra) <= set(inspect.signature(cls.__init__).parameters):
                continue
            kw = dict(base, **_kw(cls, **extra))
            r = cls(m, sim.n, **kw)
            row[cname] = _time(r, sim, a.reps)
            del r
        res[name] = row
        print(name, json.dumps(row), flush=True)
        del sim
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
