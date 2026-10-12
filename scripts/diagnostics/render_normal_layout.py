"""Tier-0 normal layouts and shadow-view culling: same image, less work (docs/RENDER_COVERAGE.md §3b).

normal_layout:
  split      a vertex per distinct (position, normal, uv): hard edges duplicate vertices (619 K vs 162 K on SO-101)
  corner_fs  vertices shared by position; the fragment stage reads the triangle's three corner normals / uvs by
             primitive_id and blends them with the hardware barycentrics
  corner_vs  non-indexed draw; the vertex stage pulls position (by index) and corner normal / uv per corner
shadow_cull: extra shadow views (spot / point / further directional lights) drawn only for instances whose bounding
sphere meets the view's frustum, per env.

Correctness: every layout and culling variant against split / no culling on the same frames (rgb, depth, seg,
normal: max |diff| and the fraction of differing pixels). Timing (relative; ``--timing``): render-only ms per frame,
best of 3 x ``--frames``.

    scripts/gpu_run.sh normal_layout check 10 -- .venv/bin/python scripts/diagnostics/render_normal_layout.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compare(a, b):
    out = {}
    for k in a:
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        d = np.abs(x - y)
        diffpix = d.reshape(d.shape[0], d.shape[1], d.shape[2], -1).max(-1) > (0 if k in ("rgb", "seg") else 1e-4)
        out[k] = {"max": round(float(d.max()), 5), "frac_px": round(float(diffpix.mean()), 6)}
    return out


def scenes():
    import render_appearance_cost as rac
    import render_coverage_audit as rca
    ms = rac.scene_models()
    out = {"so101": ms["so101"], "panda": ms["panda"], "glass": ms["glass"],
           "meshbox": (mujoco.MjModel.from_xml_string(rca.MESHBOX), "cam")}
    pl = rca.SHADOW.replace("{LIGHT}", '<light type="point" pos="0 0 1.5" castshadow="true" diffuse="0.9 0.9 0.9"/>'
                            '<light pos="0.3 0 1.2" dir="0 0 -1" cutoff="40" castshadow="true" diffuse="0.5 0.5 0.5"/>')
    out["point_spot"] = (mujoco.MjModel.from_xml_string(pl), "cam")
    import tempfile
    out["textures"] = (mujoco.MjModel.from_xml_string(rca.texture_xml(tempfile.mkdtemp())), "cam")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--tn", type=int, default=256)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--out", default="runs/render_appearance/normal_layout.jsonl")
    a = ap.parse_args()
    from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for name, (m, cam) in scenes().items():
        rng = np.random.default_rng(0)
        datas = []
        for _ in range(a.n):
            d = mujoco.MjData(m); d.qpos[:] += rng.uniform(-0.05, 0.05, m.nq) if m.nq else 0; mujoco.mj_forward(m, d)
            datas.append(d)
        for tier in (0, 1):
            ref = None
            for lay, cull in (("split", False), ("corner_fs", False), ("corner_vs", False), ("split", True), ("corner_fs", True)):
                if tier == 1 and cull:
                    continue
                r = Tier0Renderer(m, a.n, width=96, height=96, camera=cam, outputs=("rgb", "depth", "seg", "normal"),
                                  seg_mode=SEG_GEOM, tier=tier, normal_layout=lay, shadow_cull=cull)
                o = r.render_host(datas)
                if ref is None:
                    ref = o
                    row = dict(scene=name, tier=tier, variant=f"{lay}/cull={cull}", vertices=int(len(r.tables.vertices)),
                               extra_shadow_views=int(r.n_views))
                else:
                    row = dict(scene=name, tier=tier, variant=f"{lay}/cull={cull}", vertices=int(len(r.tables.vertices)),
                               vs_split=compare(o, ref))
                rows.append(row); print(json.dumps(row), flush=True)
                del r
    if a.timing:
        from metalsim.physics.batch import BatchSim
        import torch
        for name in ("so101", "panda"):
            m, cam = scenes()[name]
            sim = BatchSim(m, a.tn); sim.forward(); sim.synchronize(); ev = sim.event.value
            for lay, cull in (("split", False), ("corner_fs", False), ("corner_vs", False), ("corner_fs", True), ("split", True)):
                r = Tier0Renderer(m, a.tn, width=128, height=128, camera=cam, outputs=("rgb", "depth"), normal_layout=lay, shadow_cull=cull)
                for _ in range(2):
                    r.render(sim, ev)
                r.ctx.synchronize()
                best = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(a.frames):
                        r.render(sim, ev)
                    r.ctx.synchronize()
                    best.append((time.perf_counter() - t0) / a.frames * 1e3)
                row = dict(scene=name, timing=True, n=a.tn, variant=f"{lay}/cull={cull}", ms=round(min(best), 3))
                rows.append(row); print(json.dumps(row), flush=True)
                del r; torch.mps.synchronize()
    with open(a.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
