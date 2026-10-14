"""Run a scene file: ``python -m metalsim.run <scene.yaml> [options]``.

    python -m metalsim.run configs/scenes/go2_boxes.yaml --check                 # parse + build the model (CPU only)
    python -m metalsim.run configs/scenes/go2_boxes.yaml --steps 200 --save runs/scenes/go2_boxes
    python -m metalsim.run configs/scenes/go2_boxes.yaml --film                  # offline tier-2 film + doc (metalsim.render.film)
    mjpython -m metalsim.run configs/scenes/go2_boxes.yaml --viewer              # interactive (MuJoCo viewer, env 0)
    python -m metalsim.run configs/scenes/go2_interiorgs.yaml --rooms            # InteriorGS room centres (placement)

Headless (default): reset, step ``--steps`` control steps with the scene policy, render every camera and lidar every
``--save-every`` steps into ``--save`` (PNG per camera for ``--env``, ``.npz`` of every data type, lidar ranges), then
encode each camera's RGB to mp4, and print timings. A scene with ``task:`` (e.g. franka_pour) hands over to the film
tool. The scene schema (Isaac Lab names) is in ``metalsim/scene/cfg.py`` and docs/SCENES.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _overrides(cfg, a):
    from dataclasses import replace

    if a.num_envs:
        cfg = replace(cfg, num_envs=a.num_envs)
    return cfg


def rooms(cfg):
    from metalsim.scene import compose as CP

    sp = cfg.background.splat
    js = sp and sp.collision.structure_json
    if not js:
        print("no structure_json in background.splat.collision"); return 1
    s = json.load(open(CP.resolve(cfg, js)))
    for i, r in enumerate(s.get("rooms", [])):
        p = np.asarray(r.get("profile", []), float)
        if len(p) >= 3:
            c = p[:, :2].mean(0)
            print(f"room {i:2d} {str(r.get('name', r.get('type', ''))):24s} centre ({c[0]:.2f}, {c[1]:.2f})  "
                  f"extent {np.ptp(p[:, 0]):.1f} x {np.ptp(p[:, 1]):.1f} m")
    return 0


def colorize_depth(d, far=None):
    d = np.asarray(d, np.float32)
    far = far or float(np.percentile(d[d > 0], 99)) if (d > 0).any() else 1.0
    x = np.clip(d / max(far, 1e-6), 0, 1)
    return (np.stack([1 - x, 1 - x ** 0.5, x * 0.4 + 0.2], -1) * 255 * (d > 0)[..., None]).astype(np.uint8)


def headless(cfg, a):
    import imageio.v3 as iio

    from metalsim.scene import compose as CP

    scene = CP.compose(cfg)
    scene.reset()
    out = a.save
    if out:
        os.makedirs(out, exist_ok=True)
    t_step, t_render, t_sense = [], {}, []
    frames = {}
    for k in range(a.steps):
        t0 = time.perf_counter()
        scene.step()
        scene.sim.synchronize()
        t_step.append(time.perf_counter() - t0)
        if a.save_every and k % a.save_every == 0:
            datas = scene.host_datas()
            for name in scene.cameras:
                t0 = time.perf_counter()
                o = scene.render([name], datas)[name]
                t_render.setdefault(name, []).append(time.perf_counter() - t0)
                if out:
                    for dt, v in o.items():
                        img = v[a.env]
                        if dt in ("rgb", "rgba"):
                            p = os.path.join(out, name, "frames", f"frame_{len(frames.setdefault(name, [])):04d}.png")
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            iio.imwrite(p, np.asarray(img)[..., :3].astype(np.uint8))
                            frames[name].append(p)
                        elif dt in ("distance_to_image_plane", "distance_to_camera", "depth") and k == 0:
                            iio.imwrite(os.path.join(out, f"{name}_{dt}_0.png"), colorize_depth(img))
                    if k == 0:
                        np.savez_compressed(os.path.join(out, f"{name}_step0.npz"), **{d: np.asarray(v) for d, v in o.items()})
            if scene.lidars:
                t0 = time.perf_counter()
                hits = scene.sense(datas)
                t_sense.append(time.perf_counter() - t0)
                if out and k == 0:
                    np.savez_compressed(os.path.join(out, "lidar_step0.npz"),
                                        **{f"{n}_{key}": np.asarray(v) for n, r in hits.items() for key, v in r.items()})
    summary = dict(scene=cfg.name, num_envs=scene.n, steps=a.steps, control_dt=scene.control_dt,
                   step_ms_mean=float(np.mean(t_step) * 1e3), step_ms_median=float(np.median(t_step) * 1e3),
                   render_ms={n: float(np.median(v) * 1e3) for n, v in t_render.items()},
                   lidar_ms=float(np.median(t_sense) * 1e3) if t_sense else None, build_s=scene.build_s,
                   sim_build_s=scene.sim_s, cameras={n: dict(kind=c.kind, outputs=list(c.cfg.data_types),
                                                              size=[c.cfg.width, c.cfg.height]) for n, c in scene.cameras.items()},
                   proxy=scene.info.proxy)
    q = scene.sim.d.qpos.numpy()
    summary["final_qpos_finite"] = bool(np.isfinite(q).all())
    if scene.info.robot_free:
        import mujoco
        j = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, scene.info.robot_free)
        summary["robot_root_z"] = q[:, scene.model.jnt_qposadr[j] + 2].round(3).tolist()
    if out:
        from metalsim.render.film.encode import encode
        for name, fl in frames.items():
            if len(fl) >= 2:
                fps = 1.0 / (scene.control_dt * a.save_every)
                tmp = os.path.join(out, name)
                try:
                    encode(tmp, os.path.join(out, f"{name}.mp4"), fps, 22, None, ())
                except Exception as e:  # noqa: BLE001
                    print(f"[run] encode {name}: {e}")
        json.dump(summary, open(os.path.join(out, "run.json"), "w"), indent=1)
    print("[run] " + json.dumps(summary))
    if not summary["final_qpos_finite"]:
        return 2
    return 0


def viewer(cfg, a):
    import mujoco
    import mujoco.viewer

    from metalsim.scene import compose as CP

    scene = CP.compose(cfg, renderers=False)
    scene.reset()
    d = scene.sim.get_world(a.env)
    with mujoco.viewer.launch_passive(scene.model, d) as v:
        while v.is_running():
            t0 = time.perf_counter()
            scene.step()
            scene.sim.get_world(a.env, d)
            v.sync()
            time.sleep(max(0.0, scene.control_dt - (time.perf_counter() - t0)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--check", action="store_true", help="parse and build the model only (CPU)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--save", default=None, help="output directory for camera frames / lidar / run.json")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--env", type=int, default=0, help="env whose camera images are written as PNG / mp4")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--viewer", action="store_true", help="MuJoCo's interactive viewer on one env (macOS: mjpython)")
    ap.add_argument("--film", action="store_true", help="film with metalsim.render.film (offline path tracing)")
    ap.add_argument("--film-out", default=None)
    ap.add_argument("--rooms", action="store_true", help="list the splat scene's rooms (InteriorGS structure.json)")
    a = ap.parse_args(argv)
    from metalsim.scene import cfg as C

    cfg = _overrides(C.load(a.scene), a)
    if a.rooms:
        return rooms(cfg)
    if cfg.task:
        if not a.film:
            print(f"scene {cfg.name} is the task {cfg.task!r} (its own scene builder): film it with --film")
            return 0
        from metalsim.render.film.__main__ import main as film_main
        f = cfg.film or {}
        args = [cfg.task]
        if f.get("checkpoint"):
            args += ["--checkpoint", f["checkpoint"]]
        v = f.get("viewer") or {}
        for k, flag in (("eye", "--eye"), ("lookat", "--lookat")):
            if k in v:
                args += [flag] + [str(x) for x in v[k]]
        if "fovy" in v:
            args += ["--fovy", str(v["fovy"])]
        if a.film_out:
            args += ["--out", a.film_out]
        return film_main(args)
    if a.check:
        from metalsim.scene import compose as CP
        t0 = time.perf_counter()
        spec, info = CP.build_spec(cfg)
        m = spec.compile()
        print(f"[run] {cfg.name}: ok ({time.perf_counter() - t0:.2f} s): nbody {m.nbody} ngeom {m.ngeom} nq {m.nq} "
              f"nu {m.nu} ncam {m.ncam} nsite {m.nsite}; robot joints {len(info.robot_joints)}, objects "
              f"{len(info.objects)}, cameras {list(info.cameras)}, lidars {list(info.lidars)}, proxy {info.proxy}")
        return 0
    if a.viewer:
        return viewer(cfg, a)
    if a.film:
        from metalsim.scene.film import film
        film(cfg, a.film_out or os.path.join("runs", "film", cfg.name))
        return 0
    return headless(cfg, a)


if __name__ == "__main__":
    sys.exit(main())
