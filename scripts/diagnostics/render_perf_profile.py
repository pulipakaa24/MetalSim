"""Where tier 1 / tier 2 frame time goes, per feature and per GPU pass (docs/RENDER_COVERAGE.md "Render cost by feature").

For one scene, renderer and env count: the frame time with the defaults, with each fidelity feature switched off on its
own (its documented switch), and with all of them off (the configuration before the 2026-10 render workstreams); and
for the defaults and all-off, the GPU time of every pass (metalsim.render.gpu_profile: one command buffer per encoder).

Scenes (synthetic MJCF):
  cloth   deformable_render_bench's MuJoCo Warp flex scene (cloth + cable + soft cube on a box, one spot light), 64x64
  glass   render_coverage_audit.TRANSPARENT (a 25 % glass pane before a red box, one directional light), 128x128

    scripts/gpu_run.sh perf_t1 timing 15 -- .venv/bin/python scripts/diagnostics/render_perf_profile.py \\
        --scene cloth --renderer tier1 --envs 4096 --out runs/render_gaps2/perf.jsonl

Frame time: ``--frames`` renders committed back to back, one synchronize, best of 3 (render only, the simulator's
buffers already written). Each configuration runs in this process one after the other; the device is checked idle by
gpu_run.sh at start and end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--root"), ROOT)   # metalsim tree to time
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402

ALL_OFF = dict(appearance="legacy", transparency=False, skybox=False, infinite_planes=False, model_fields=False,
               hfield_skirts=False, per_world_size=False, haze=False, tendons="straight")
ALL_OFF_T1 = dict(ALL_OFF, shadow_lights="caster", max_lights=8, reverse_z=False)
# feature -> (renderer kwargs, environment overrides) that switch it off alone
FEATURES_T1 = {
    "shadow_lights=caster": (dict(shadow_lights="caster"), {}),
    "transparency=False": (dict(transparency=False), {}),
    "infinite_planes=False": (dict(infinite_planes=False), {}),
    "model_fields=False (no per-env light records)": (dict(model_fields=False), {}),
    "env_layout=legacy (no cell clipping)": ({}, {"METALSIM_ENV_LAYOUT": "legacy"}),
    "reverse_z=False": (dict(reverse_z=False), {}),
    "appearance=legacy": (dict(appearance="legacy"), {}),
    "max_lights=8": (dict(max_lights=8), {}),
    "dyn_layout=sbvh": ({}, {"METALSIM_DYN_LAYOUT": "sbvh"}),
}
FEATURES_T2 = {
    "transparency=False": (dict(transparency=False), {}),
    "infinite_planes=False": (dict(infinite_planes=False), {}),
    "model_fields=False (no per-env light records)": (dict(model_fields=False), {}),
    "env_layout=legacy (no cell clipping)": ({}, {"env_layout": "legacy"}),
    "appearance=legacy": (dict(appearance="legacy"), {}),
    "dyn_layout=per_env": (dict(dyn_layout="per_env"), {}),
}


def scene(name, n):
    if name == "cloth":
        from deformable_render_bench import cloth_scene, make_sim
        m, kind = cloth_scene()
        return m, make_sim(m, kind, n), "cam", 64
    if name == "cloth_inf":            # the same with the floor an infinite plane (horizon-continued in the ray tracers)
        from deformable_render_bench import make_sim
        from metalsim.physics import deformable as dfm
        from deformable_render_bench import CAM
        xml = dfm.box_scene_xml(dfm.ClothCfg(), dfm.CableCfg(), soft=dfm.SoftCfg()).replace("<worldbody>", "<worldbody>" + CAM, 1)
        xml = xml.replace('type="plane" size="2 2 0.1"', 'type="plane" size="0 0 0.1"')
        m = mujoco.MjModel.from_xml_string(xml)
        return m, make_sim(m, "deformable", n), "cam", 64
    if name == "glass":
        import render_coverage_audit as rca
        from metalsim.physics.batch import BatchSim
        m = mujoco.MjModel.from_xml_string(rca.TRANSPARENT)
        sim = BatchSim(m, n); sim.forward(); sim.synchronize()
        return m, sim, "cam", 128
    raise ValueError(name)


def make(which, m, n, cam, res, kw, env):
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    old = {k: os.environ.get(k) for k in env if k.startswith("METALSIM_")}
    for k, v in env.items():
        if k.startswith("METALSIM_"):
            os.environ[k] = v
    try:
        if which == "tier1":
            kw = {k: v for k, v in kw.items() if k not in ("dyn_layout",)}
            return Tier0Renderer(m, n, width=res, height=res, camera=cam, outputs=("rgb", "depth"), tier=1, **kw)
        kw = dict(kw)
        if "env_layout" in env:
            kw["env_layout"] = env["env_layout"]
        for k in ("shadow_lights", "max_lights", "reverse_z"):
            kw.pop(k, None)
        spp, b = (1, 1) if res <= 64 else (4, 2)
        return Tier2Renderer(m, n, width=res, height=res, camera=cam, spp=spp, max_bounces=b, **kw)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def frame_ms(r, sim, frames):
    for _ in range(2):
        r.render(sim)
    r.ctx.synchronize()
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(frames):
            r.render(sim)
        r.ctx.synchronize()
        best.append((time.perf_counter() - t0) / frames * 1e3)
    return min(best)


def _gpu_profile():
    """This tree's metalsim.render.gpu_profile, also when timing an older tree (--root) that predates it."""
    try:
        from metalsim.render import gpu_profile
        return gpu_profile
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gpu_profile", os.path.join(ROOT, "metalsim", "render", "gpu_profile.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def passes(r, sim, frames):
    gpu_profile = _gpu_profile()
    prof = gpu_profile.PassProfiler(r.ctx)
    with prof:
        for _ in range(frames + 1):
            r.render(sim)
            prof.next_frame()
        r.ctx.synchronize()
    return prof.table(skip_frames=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cloth")
    ap.add_argument("--renderer", default="tier1", choices=["tier1", "tier2"])
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--features", default="all", help="'all', 'none' or comma-separated names")
    ap.add_argument("--profile", type=int, default=1, help="per-pass table for defaults and all-off")
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=ROOT, help="metalsim tree to time (an older commit's export)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--all-off", type=int, default=1, help="0: skip the all-off configuration (older trees lack its switches)")
    a = ap.parse_args()
    import warp as wp
    wp.config.quiet = True
    _gpu_profile().install()
    m, sim, cam, res = scene(a.scene, a.envs)
    feats = FEATURES_T1 if a.renderer == "tier1" else FEATURES_T2
    if a.features == "none":
        feats = {}
    elif a.features != "all":
        feats = {k: v for k, v in feats.items() if any(k.startswith(f) for f in a.features.split(","))}
    configs = [("defaults", {}, {})] + [(k, kw, env) for k, (kw, env) in feats.items()]
    if a.all_off:
        configs.append(("all_off", ALL_OFF_T1 if a.renderer == "tier1" else ALL_OFF,
                        {"METALSIM_ENV_LAYOUT": "legacy", "env_layout": "legacy"}))
    rows = []
    for label, kw, env in configs:
        r = make(a.renderer, m, a.envs, cam, res, kw, env)
        row = dict(scene=a.scene, renderer=a.renderer, envs=a.envs, res=res, config=label, tree=a.tag or a.root,
                   ms=round(frame_ms(r, sim, a.frames), 3))
        if a.profile and label in ("defaults", "all_off"):
            row["passes"] = passes(r, sim, min(a.frames, 5))
        print(json.dumps(row), flush=True)
        rows.append(row)
        del r
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
