"""Record stage: roll out a task with its policy, then write a portable recording that the render stage reads in any
virtual environment:

* ``twin.xml`` (+ ``twin_tex_*.png``): the render twin as MJCF. Newton tasks: ``newton_bridge.NewtonRenderSource``'s
  twin of a one-world copy of the scene with the visual shapes loaded (meshes inline, textures as PNG); MuJoCo tasks:
  the task's render model. MJCF rather than a compiled ``.mjb`` because the two venvs carry different MuJoCo versions.
* ``rec.npz``: per frame and filmed env, the world pose of every twin geom (``xpos`` (F, E, G, 3), ``xmat``
  (F, E, G, 9), in each env's own frame), the particles (``particles`` (F, E, P, 3) or empty), the tracked body
  (``track`` (F, E, 3)), each env's first-episode end frame (``ends``), and the twin's compiled geom frames.
* ``rec.json``: task, envs and their outcomes, timings, provenance (the policy checkpoint's repo-relative path and
  sha256 under ``checkpoint``).

A filmed env whose first episode ends by a termination (the adapter's ``info["terminated"]``: a fall, not a time-out
or a success) makes the recorder refuse the film unless ``FilmCfg.allow_early_end`` (``--allow-early-end``); with
``envs="auto"`` envs that did not terminate are ranked first.

    python -m metalsim.render.film.record --task franka_pour --out runs/film/franka_pour/rec
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy as np

from metalsim.render.film import provenance


def write_png(path: str, img: np.ndarray) -> None:
    """Minimal PNG writer (8-bit RGB / RGBA; no imageio in every venv)."""
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    ctype = {1: 0, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def export_twin(spec, out_dir: str) -> str:
    """Write ``spec`` as ``twin.xml`` in ``out_dir``: in-memory textures become PNG files next to it (MJCF has no
    inline texture data); unnamed geoms get names (the recording maps poses by geom name)."""
    import mujoco

    fix_mesh_corners(spec)
    written = []
    for i, t in enumerate(spec.textures):
        data = bytes(t.data) if t.data is not None else b""
        if data and not t.file:
            h, w, c = int(t.height), int(t.width), int(t.nchannel or 3)
            img = np.frombuffer(data, np.uint8)[: h * w * c].reshape(h, w, c)
            name = f"twin_tex_{i}.png"
            full = os.path.abspath(os.path.join(out_dir, name))
            write_png(full, img)
            t.data = b""
            t.file = full                   # to_xml compiles: an absolute path while writing, relative in the file
            written.append((full, name))
    xml = spec.to_xml()
    for full, name in written:
        xml = xml.replace(f'"{full}"', f'"{name}"')
    path = os.path.join(out_dir, "twin.xml")
    with open(path, "w") as f:
        f.write(xml)
    del mujoco
    return path


def fix_mesh_corners(spec) -> dict:
    """MJCF inline meshes carry per-vertex texcoords / normals only (no per-face-corner indices). Meshes with corner
    indices (USD imports) are rewritten: texcoords dropped when no geom using the mesh has a textured material,
    otherwise the mesh is unwelded (one vertex per face corner) with the original smooth vertex normals kept."""
    textured = {m.name for m in spec.materials if any(t for t in m.textures)}
    uses = {}
    for g in spec.geoms:
        if g.meshname:
            uses.setdefault(g.meshname, set()).add(g.material in textured)
    dropped = unwelded = 0
    for me in spec.meshes:
        v = np.asarray(me.uservert, np.float64).reshape(-1, 3)
        nv = len(v)
        tc, ftc = np.asarray(me.usertexcoord, np.float64), np.asarray(me.userfacetexcoord, np.int64)
        fn = np.asarray(me.userfacenormal, np.int64)
        corner = (len(ftc) and len(ftc) != 0) or (len(tc) and len(tc) != 2 * nv) or len(fn)
        if not nv or not corner:
            continue
        if not any(uses.get(me.name, ())) or not len(tc):
            me.usertexcoord = []; me.userfacetexcoord = []; me.usernormal = []; me.userfacenormal = []
            dropped += 1
            continue
        f = np.asarray(me.userface, np.int64).reshape(-1, 3)
        # area-weighted smooth normals on the welded mesh, then one vertex per corner
        n = np.zeros_like(v)
        fnrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for k in range(3):
            np.add.at(n, f[:, k], fnrm)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        uv = tc.reshape(-1, 2)
        ti = ftc.reshape(-1, 3) if len(ftc) else f
        me.uservert = v[f.reshape(-1)].ravel().tolist()
        me.usernormal = n[f.reshape(-1)].ravel().tolist()
        me.usertexcoord = uv[ti.reshape(-1)].ravel().tolist()
        me.userface = np.arange(f.size).tolist()
        me.userfacetexcoord = []; me.userfacenormal = []
        unwelded += 1
    return dict(texcoords_dropped=dropped, unwelded=unwelded)


def name_geoms(spec) -> None:
    for i, g in enumerate(spec.geoms):
        if not g.name:
            g.name = f"_film_geom_{i}"


def prefer_visuals(model) -> dict:
    """Shape visibility for the render twin of a Newton model (in place): a body that has visual-only shapes
    (no rigid or particle collision) hides its colliders; a body with no visible shape shows its rigid colliders (any
    geometry, unlike the bridge's primitive-only ``collider_fallback``). Static (world) shapes are left as they are."""
    from newton import ShapeFlags as F

    flags = model.shape_flags.numpy().copy()
    body = model.shape_body.numpy()
    coll = int(F.COLLIDE_SHAPES) | int(F.COLLIDE_PARTICLES)
    vis = int(F.VISIBLE)
    hid = shown = 0
    for b in np.unique(body[body >= 0]):
        s = np.flatnonzero(body == b)
        visual_only = [i for i in s if flags[i] & vis and not flags[i] & coll]
        if visual_only:
            for i in s:
                if flags[i] & coll and flags[i] & vis:
                    flags[i] &= ~vis; hid += 1
        elif not any(flags[i] & vis for i in s):
            for i in s:
                if flags[i] & int(F.COLLIDE_SHAPES):
                    flags[i] |= vis; shown += 1
    model.shape_flags.assign(flags)
    return dict(colliders_hidden=hid, colliders_shown=shown)


def mujoco_visibility(spec) -> dict:
    """Render-twin visibility for a MuJoCo spec (in place), the MuJoCo counterpart of ``prefer_visuals``:

    * a body (not the world) that has visual-only geoms (contype = conaffinity = 0) hides its colliding geoms: group 3,
      which MuJoCo's default ``MjvOption`` and the film's ray tracer (``max_group=3``) do not draw. The G1's USD
      colliders (``Cube``, ``right_ankle_roll_link_Cube``, ``torso_link_Cube``: ``UsdPhysics.CollisionAPI`` meshes
      authored ``visibility = invisible``, the model's only three colliders) were drawn as plain grey foot plates and
      torso box. World geoms (ground, terrain) are left as they are.
    * every site is hidden (group 5): the recording poses geoms only, so a site is drawn where the model's ``qpos0``
      puts it. The G1's torso touch-sensor box (0.3 x 0.3 x 0.5 m) stood as a grey box at the world origin."""
    hid = 0
    for b in spec.bodies:
        if b.name == "world":
            continue
        geoms = list(b.geoms)
        if any(g.contype == 0 and g.conaffinity == 0 for g in geoms):
            for g in geoms:
                if (g.contype or g.conaffinity) and g.group < 3:
                    g.group = 3
                    hid += 1
    sites = 0
    for st in spec.sites:
        if st.group < 3:
            st.group = 5
            sites += 1
    return dict(colliders_hidden=hid, sites_hidden=sites)


def checkpoint_provenance(path: str | None) -> dict | None:
    """Repo-relative path (``provenance.relpath``: no machine paths), size and sha256 of a policy checkpoint."""
    if not path:
        return None
    out = dict(path=provenance.relpath(path), in_repo=os.path.abspath(path).startswith(provenance.ROOT + os.sep))
    if not os.path.exists(path):
        return out | dict(sha256=None, missing=True)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return out | dict(sha256=h.hexdigest(), bytes=os.path.getsize(path))


def neutral_unauthored_colors(model, rgb=(0.80, 0.80, 0.80)) -> int:
    """Newton gives a shape without an authored colour a palette colour by shape index (``ModelBuilder.
    _shape_palette_color``: a debug-view rainbow); for a film those shapes get a neutral ``rgb`` instead."""
    from newton import ModelBuilder

    palette = getattr(ModelBuilder, "_SHAPE_COLOR_PALETTE", None)
    if palette is None or getattr(model, "shape_color", None) is None:
        return 0
    # by membership, not by index: shapes copied from prototype builders (add_builder) keep the prototype's index
    pal = np.asarray(palette, np.float32)[:, :3] / 255.0
    col = model.shape_color.numpy().copy()
    hit = [s for s in range(len(col)) if (np.abs(pal - col[s]).max(1) < 1e-5).any()]
    col[hit] = rgb
    model.shape_color.assign(col)
    return len(hit)


def _label_tail(s: str) -> str:
    return str(s).rstrip("/").rsplit("/", 1)[-1]


def newton_twin(adapter, snaps, envs, ends, frames, out_dir, track_body=None):
    """Render twin and per-frame geom poses for a Newton task (CPU; no GPU)."""
    import mujoco
    import warp as wp

    from metalsim.render.newton_bridge import NewtonRenderSource

    t0 = time.perf_counter()
    model = adapter.twin_newton()
    vis = prefer_visuals(model)
    vis["palette_colours_neutralised"] = neutral_unauthored_colors(model)
    specs = []
    src = NewtonRenderSource(model, env_origins=None, headlight=False, deformables=False, spec_hook=specs.append,
                             device="cpu")
    tm = src.m
    names = [mujoco.mj_id2name(tm, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(tm.ngeom)]
    build_s = time.perf_counter() - t0
    E, G = len(envs), tm.ngeom
    xpos = np.zeros((frames, E, G, 3), np.float32)
    xmat = np.zeros((frames, E, G, 9), np.float32)
    track = np.zeros((frames, E, 3), np.float32)
    labels = [_label_tail(x) for x in model.body_label]
    tb = labels.index(track_body) if track_body in labels else None
    state = model.state()
    t1 = time.perf_counter()
    for k in range(frames):
        for j, e in enumerate(envs):
            bq = snaps[k]["body_q"][e]
            state.body_q.assign(wp.array(bq.astype(np.float32), dtype=wp.transform, device="cpu"))
            src.launch(state)
            d = src.host_datas()[0]
            xpos[k, j] = d.geom_xpos.reshape(G, 3)
            xmat[k, j] = d.geom_xmat.reshape(G, 9)
            if tb is not None:
                track[k, j] = bq[tb, :3]
    pose_s = time.perf_counter() - t1
    path = export_twin(specs[0], out_dir)
    ntex = sum(1 for t in specs[0].textures)
    info = dict(kind="newton", geoms=G, meshes=int(tm.nmesh), textures=ntex, materials=int(tm.nmat),
                visibility=vis, build_s=build_s, poses_s=pose_s, shapes=int(model.shape_count))
    return path, names, tm.geom_pos.copy(), tm.geom_quat.copy(), xpos, xmat, track, info


def mujoco_twin(adapter, snaps, envs, ends, frames, out_dir, track_body=None):
    """Render twin and per-frame geom poses for a MuJoCo task: forward kinematics of the render model on the
    recorded ``qpos``."""
    import mujoco

    t0 = time.perf_counter()
    spec = adapter.twin_spec()
    name_geoms(spec)
    vis = mujoco_visibility(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
    build_s = time.perf_counter() - t0
    E, G = len(envs), m.ngeom
    xpos = np.zeros((frames, E, G, 3), np.float32)
    xmat = np.zeros((frames, E, G, 9), np.float32)
    track = np.zeros((frames, E, 3), np.float32)
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, track_body) if track_body else -1
    t1 = time.perf_counter()
    for k in range(frames):
        for j, e in enumerate(envs):
            d.qpos[:] = snaps[k]["qpos"][e]
            mujoco.mj_kinematics(m, d)
            xpos[k, j] = d.geom_xpos
            xmat[k, j] = d.geom_xmat
            if tb >= 0:
                track[k, j] = d.xpos[tb]
    pose_s = time.perf_counter() - t1
    path = export_twin(spec, out_dir)
    info = dict(kind="mujoco", geoms=G, meshes=int(m.nmesh), textures=int(m.ntex), materials=int(m.nmat),
                visibility=vis, build_s=build_s, poses_s=pose_s)
    return path, names, m.geom_pos.copy(), m.geom_quat.copy(), xpos, xmat, track, info


def load_adapter(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


class FilmEarlyEnd(RuntimeError):
    """A filmed env's first episode ended by a termination (e.g. the robot fell) inside the film."""


def record(task, film, viewer, out_dir: str, checkpoint: str | None, n_keep: int = 4) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.perf_counter()
    ck_prov = checkpoint_provenance(checkpoint)
    Adapter = load_adapter(task.adapter) if isinstance(task.adapter, str) else task.adapter
    adapter = Adapter(task, film, checkpoint, viewer)
    fps = film.fps or 1.0 / adapter.frame_dt
    if abs(fps * adapter.frame_dt - 1.0) > 1e-6:
        raise ValueError("the recorder films one frame per policy step: fps must be 1 / step_dt")
    max_frames = film.video_length or int(math.ceil(film.max_seconds / adapter.frame_dt)) + 1
    print(f"[record] {task.name}: {adapter.num_envs} envs, up to {max_frames} frames at {fps:.1f} fps", flush=True)
    adapter.reset()
    snaps, sim_s = [], []
    n = adapter.num_envs
    ends = np.full(n, -1, np.int64)
    first = {}
    t0 = time.perf_counter()

    def terminated():
        """(N,) first episode ended by a termination (only adapters that report ``terminated``; else all False)."""
        return np.asarray(first.get("terminated", np.zeros(n, bool)), bool) & (ends >= 0)

    for k in range(max_frames):
        snap, done, info, timing = adapter.step()
        snaps.append(snap)
        sim_s.append(timing["sim_s"])
        new = np.asarray(done, bool) & (ends < 0)
        for key, v in info.items():
            first.setdefault(key, np.zeros(n, np.asarray(v).dtype))
            first[key][new] = np.asarray(v)[new]
        ends[new] = k
        if k % 30 == 0 or k == max_frames - 1:
            term = f" ({int(terminated().sum())} by termination)" if "terminated" in first else ""
            print(f"[record] frame {k + 1}/{max_frames}: {sim_s[-1] * 1e3:.0f} ms, episodes ended {int((ends >= 0).sum())}"
                  f"/{n}{term}", flush=True)
        if film.video_length is None and (ends >= 0).all():
            break
    rollout_s = time.perf_counter() - t0
    frames = len(snaps)
    term = terminated()
    if hasattr(adapter, "summary"):         # optional: per-env first-episode statistics for the log
        print(adapter.summary(ends), flush=True)
    if "terminated" in first:
        fell = np.flatnonzero(term)
        print(f"[record] first episodes ended by termination: {len(fell)}/{n}"
              + (f" (env:frame {', '.join(f'{e}:{int(ends[e])}' for e in fell)})" if len(fell) else ""), flush=True)
    if isinstance(film.envs, (list, tuple)):
        envs = [int(e) for e in film.envs]
    else:
        ranked = [int(e) for e in adapter.rank({k: v for k, v in first.items()})]
        ranked = [e for e in ranked if not term[e]] + [e for e in ranked if term[e]]    # stable: terminated last
        g2 = film.grid ** 2             # the slots the render stage films by default; the rest are spares
        envs = ranked[:g2] + [e for e in ranked[g2:max(n_keep, g2)] if not term[e]]
    print(f"[record] filmed envs {envs} (ends {[int(ends[e]) for e in envs]})", flush=True)
    early = [e for e in envs if term[e]]
    if early:
        msg = "; ".join(f"env {e} terminated at frame {int(ends[e])} ({ends[e] * adapter.frame_dt:.2f} s) of {frames}"
                        for e in early)
        ok = [e for e in range(n) if not term[e]]
        if not film.allow_early_end:
            raise FilmEarlyEnd(f"[record] {task.name}: {msg}: the film would show the episode ending by a termination "
                               f"(a fall), not by a time-out or success. {len(ok)}/{n} envs did not terminate"
                               f"{' (' + ','.join(map(str, ok[:16])) + ')' if ok else ''}: film one of them (--envs), "
                               f"or pass --allow-early-end to film it anyway.")
        print(f"[record] WARNING (allow_early_end): {msg}", flush=True)
    twin_fn = newton_twin if adapter.backend == "newton" else mujoco_twin
    path, names, gpos, gquat, xpos, xmat, track, twin_info = twin_fn(adapter, snaps, envs, ends, frames, out_dir,
                                                                     viewer.body_name)
    parts = np.zeros((frames, len(envs), 0, 3), np.float32)
    if adapter.backend == "newton" and snaps[0].get("particles") is not None:
        parts = np.stack([np.stack([s["particles"][e] for e in envs]) for s in snaps]).astype(np.float32)
    np.savez(os.path.join(out_dir, "rec.npz"), xpos=xpos, xmat=xmat, particles=parts, track=track,
             ends=np.array([int(ends[e]) for e in envs]), geom_pos=gpos, geom_quat=gquat)
    sim = np.asarray(sim_s)
    out = dict(task=task.name, backend=adapter.backend, venv=task.venv, frames=frames, fps=fps, frame_dt=adapter.frame_dt,
               num_envs=n, envs=envs, checkpoint=ck_prov, allow_early_end=bool(film.allow_early_end),
               early_end_envs=[int(e) for e in early], geom_names=names,
               episodes={str(e): {k: (v[e].item() if hasattr(v[e], "item") else v[e]) for k, v in first.items()}
                         | {"end_frame": int(ends[e])} for e in envs},
               all_first_episodes={k: np.asarray(v).tolist() for k, v in first.items()} | {"end_frame": ends.tolist()},
               particles_per_env=int(parts.shape[2]), particle_radius=float(getattr(adapter, "particle_radius", 0.0)),
               twin=twin_info, adapter=adapter.meta(),
               timing=dict(sim_ms_mean=float(sim.mean() * 1e3), sim_ms_median=float(np.median(sim) * 1e3),
                           sim_ms_first=float(sim[0] * 1e3), sim_s_total=float(sim.sum()), rollout_s=rollout_s,
                           twin_s=twin_info["build_s"] + twin_info["poses_s"],
                           record_wall_s=time.perf_counter() - t_all, env_build_s=adapter.meta().get("env_build_s")),
               software=provenance.software(), hardware=provenance.hardware())
    with open(os.path.join(out_dir, "rec.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[record] done: {frames} frames, sim {out['timing']['sim_ms_mean']:.0f} ms/frame mean, twin {twin_info['geoms']} "
          f"geoms, wall {out['timing']['record_wall_s']:.0f} s -> {provenance.relpath(out_dir)}", flush=True)
    return out


def main(argv=None):
    from metalsim.render.film import tasks
    from metalsim.render.film.cli import apply_overrides, add_override_args

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    add_override_args(ap)
    a = ap.parse_args(argv)
    task = apply_overrides(tasks.get(a.task), a)
    ckpt = a.checkpoint or task.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(provenance.ROOT, ckpt)
    record(task, task.film, task.viewer, a.out, ckpt)


if __name__ == "__main__":
    sys.exit(main())
