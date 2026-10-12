"""Speed impact of the 2026-10 appearance fixes on batched envs (docs/RENDER_COVERAGE.md).

Each scene is rendered from a fixed physics state (no steps between frames; every frame waits on the same sim event),
``--frames`` back-to-back renders, one synchronize, best of 3. Configurations:

  before   appearance="legacy", transparency=False, shadow_lights="caster", max_lights=8, skybox=False (the renderer
           before the fixes; tier 2: appearance="legacy", transparency=False, skybox=False)
  after    the defaults (MuJoCo geometry / normals / texture mapping, skybox, transparency, every light, all shadows;
           normal_layout="corner_fs", shadow_cull=True)
  after_split / after_corner_vs / after_nocull (tier 0): the same with the split-vertex layout and no shadow-view culling
           (the first version of the fix), with the vertex-pulling layout, or with corner_fs but no culling

Scenes: SO-101 box task (base_cam, Cartpole-like workload of the learning tasks), Franka Panda (Menagerie; a
shadow-casting spot light -> one extra shadow pass in tier 0), G1 rough terrain head camera (the G1 task's camera
path), and a transparency probe (one 25 % glass pane).

    scripts/gpu_run.sh appearance_cost timing 15 -- .venv/bin/python scripts/diagnostics/render_appearance_cost.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np
import torch
import warp as wp

wp.config.quiet = True
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BEFORE0 = dict(appearance="legacy", transparency=False, shadow_lights="caster", max_lights=8, skybox=False)
BEFORE2 = dict(appearance="legacy", transparency=False, skybox=False)
AFTER0 = {"after_split": dict(normal_layout="split", shadow_cull=False),       # the first fix: split vertices, no culling
          "after_corner_vs": dict(normal_layout="corner_vs"), "after_nocull": dict(shadow_cull=False)}   # after = defaults
LOD = {"so101_lod3000": dict(decimate_faces=3000)}      # the SO-101 lift task's training setting (LiftConfig.decimate_faces)


def timed(r, sim, ev, frames, tier):
    for _ in range(2):
        r.render(sim, ev)
    r.ctx.synchronize()
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(frames):
            r.render(sim, ev)
        r.ctx.synchronize()
        best.append((time.perf_counter() - t0) / frames * 1e3)
    return min(best)


def scene_models():
    import render_coverage_audit as rca
    out = {"so101": (mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")), "base_cam")}
    out["so101_lod3000"] = out["so101"]
    men, p = None, ROOT                              # upstream/mujoco_menagerie of this checkout or an enclosing one
    while p and p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "upstream", "mujoco_menagerie")):
            men = os.path.join(p, "upstream", "mujoco_menagerie"); break
        p = os.path.dirname(p)
    panda = os.path.join(men or "", "franka_emika_panda", "scene.xml")
    if os.path.exists(panda):
        spec = mujoco.MjSpec.from_file(panda)
        cam = spec.worldbody.add_camera(); cam.name = "cam"; cam.pos = [1.6, 0.0, 0.9]
        x = np.array([0.0, 1.0, 0.0]); y = np.array([-0.45, 0.0, 0.9]); y /= np.linalg.norm(y)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, np.cross(x, y)], 1).ravel())
        cam.quat = q.tolist()
        out["panda"] = (spec.compile(), "cam")
    out["glass"] = (mujoco.MjModel.from_xml_string(rca.TRANSPARENT), "cam")
    out["point_spot"] = (mujoco.MjModel.from_xml_string(rca.SHADOW.replace("{LIGHT}",       # 6 + 1 extra shadow views
        '<light type="point" pos="0 0 1.5" castshadow="true" diffuse="0.6 0.6 0.6"/>'
        '<light pos="0.3 0 1.2" dir="0 0 -1" cutoff="40" castshadow="true" diffuse="0.5 0.5 0.5"/>')), "cam")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--n1", type=int, default=256)
    ap.add_argument("--n2", type=int, default=64)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--scenes", default="so101,so101_lod3000,panda,point_spot,glass,g1_rough")
    ap.add_argument("--out", default="runs/render_appearance/cost.jsonl")
    a = ap.parse_args()
    from metalsim.physics.batch import BatchSim
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    scenes = a.scenes.split(",")
    models = scene_models()
    rows = []
    for name in scenes:
        if name == "g1_rough":
            from terrain_render_cost import rough_scene
            m, sim, _, _ = rough_scene(a.n)
            cam, tiers = "head", ((0, a.n),)
        elif name in models:
            m, cam = models[name]
            tiers = ((0, a.n), (1, a.n1), (2, a.n2))
            sim = None
        else:
            continue
        for tier, n in tiers:
            if name != "g1_rough":
                sim = BatchSim(m, n); sim.forward(); sim.synchronize()
            ev = sim.event.value
            for cfg in (("before", "after_split", "after_corner_vs", "after_nocull", "after") if tier == 0 else ("before", "after")):
                if tier < 2:
                    kw = dict(BEFORE0) if cfg == "before" else dict(AFTER0.get(cfg, {}))
                    kw.update(LOD.get(name, {}))
                    if tier == 1:
                        kw.pop("shadow_lights", None)
                    r = Tier0Renderer(m, n if name != "g1_rough" else a.n, width=a.size, height=a.size, camera=cam,
                                      outputs=("rgb", "depth"), tier=tier, **kw)
                else:
                    kw = dict(BEFORE2) if cfg == "before" else {}
                    kw.update(LOD.get(name, {}))
                    r = Tier2Renderer(m, n, width=a.size, height=a.size, camera=cam, spp=4, max_bounces=2, **kw)
                ms = timed(r, sim, ev, a.frames, tier)
                extra = {}
                if tier == 0:
                    extra = dict(extra_shadow_views=int(r.n_views), transparent_slots=int(r.n_trans))
                row = dict(scene=name, tier=tier, config=cfg, n=n if name != "g1_rough" else a.n, size=a.size, ms=round(ms, 3),
                           env_frames_per_s=round((n if name != "g1_rough" else a.n) / ms * 1e3), vertices=int(len(r.tables.vertices)),
                           indices=int(len(r.tables.indices)), **extra)
                rows.append(row); print(json.dumps(row), flush=True)
                del r; torch.mps.synchronize()
            if name != "g1_rough":
                del sim
    with open(a.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
