"""A composed scene as a film task (``metalsim.render.film``): ``film_task(cfg)`` returns a ``FilmTask`` whose recorder
adapter steps the scene's ``BatchSim`` with the scene policy and records ``qpos``; the render twin is the scene's own
``MjSpec`` (proxies and sensor sites are in groups the path tracer does not draw)."""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np

from metalsim.scene import cfg as C
from metalsim.scene import compose as CP


def make_adapter(scene_cfg: C.SceneCfg):
    class SceneAdapter:
        backend = "mujoco"

        def __init__(self, task, film, checkpoint, viewer=None):
            cfg = replace(scene_cfg, num_envs=int(film.num_envs))
            t0 = time.perf_counter()
            self.scene = CP.ComposedScene(cfg, renderers=False)
            self.build_s = time.perf_counter() - t0
            self.num_envs = self.scene.n
            self.frame_dt = self.scene.control_dt
            self.cfg = cfg

        def reset(self):
            self.scene.reset()

        def step(self):
            t0 = time.perf_counter()
            self.scene.step()
            self.scene.sim.synchronize()
            sim_s = time.perf_counter() - t0
            q = self.scene.sim.d.qpos.numpy().astype(np.float64).reshape(self.num_envs, -1)
            return dict(qpos=q), np.zeros(self.num_envs, bool), {}, dict(sim_s=sim_s)

        def rank(self, first):
            return list(range(self.num_envs))

        def twin_spec(self):
            return CP.build_spec(self.cfg)[0]

        def meta(self):
            return dict(policy=f"scene policy: {self.cfg.policy.type}", env_build_s=self.build_s,
                        physics="MuJoCo Warp BatchSim (metalsim.scene.compose)")

    return SceneAdapter


def film_task(cfg: C.SceneCfg, doc: str | None = None):
    from metalsim.render.film.cfg import AssetRef, FilmCfg, FilmTask, LookCfg, ViewerCfg

    if cfg.background.splat is not None:
        raise NotImplementedError("filming a splat background: the offline path tracer does not draw splats yet; "
                                  "record the composited camera instead (python -m metalsim.run <scene> --save DIR)")
    f = dict(cfg.film or {})
    v = dict(f.get("viewer") or {})
    viewer = ViewerCfg(**{k: tuple(x) if isinstance(x, list) else x for k, x in v.items()})
    film = FilmCfg(num_envs=min(int(cfg.num_envs), 4), envs=[0], grid=1, seed=int(cfg.policy.seed),
                   max_seconds=float(f.get("seconds", 5.0)), hold_s=0.0, passes=int(f.get("passes", 16)),
                   spp=int(f.get("spp", 4)), bounces=int(f.get("bounces", 3)), output_filename_prefix=cfg.name)
    dome = cfg.background.dome
    look = LookCfg(hdri=CP.resolve(cfg, dome.texture_file) if dome is not None and dome.texture_file else None,
                   ground_texture=None, materials={})
    assets = [AssetRef(a.get("name", ""), a.get("source", ""), a.get("url", ""), a.get("licence", ""), a.get("fetch", ""))
              for a in cfg.assets]
    return FilmTask(name=cfg.name, adapter=make_adapter(cfg), venv="main", checkpoint=None,
                    title=cfg.name.replace("_", " "), summary=f"Scene `{cfg.name}` composed from its scene file.",
                    viewer=viewer, film=film, look=look, assets=assets, doc=doc or f.get("doc"))


def film(cfg: C.SceneCfg, out: str, stages=("record", "render", "encode", "doc")):
    """Film a composed scene with the film tool's stages (all in this venv)."""
    from metalsim.render.film import doc as D
    from metalsim.render.film import encode as E
    from metalsim.render.film import record as R
    from metalsim.render.film import render as RR

    task = film_task(cfg)
    rec = os.path.join(out, "rec")
    if "record" in stages:
        R.record(task, task.film, task.viewer, rec, None)
    if "render" in stages:
        RR.render(task, rec, os.path.join(out, "hero"))
    if "encode" in stages:
        import json
        fps = json.load(open(os.path.join(rec, "rec.json")))["fps"]
        E.encode(os.path.join(out, "hero"), os.path.join(out, f"{cfg.name}.mp4"), fps, task.film.crf, 1280,
                 task.film.stills[1:2])
    if "doc" in stages and task.doc:
        D.write_doc(task, rec, [("hero", os.path.join(out, "hero"))], task.doc, ["python -m metalsim.run <scene> --film"])
    return task
