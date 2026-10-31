"""Render stage: a recording (``record.py``) -> PNG frames with metalsim's tier-2 path tracer (Tier2Renderer: OmniPBR
materials, importance-sampled HDRI dome, optional analytic sun, OIDN denoiser), particles as render grains
(``Tier2Renderer.set_particles``, metalsim/render/PARTICLES.md). Needs the main venv (.venv: Metal, pyoidn, imageio).

    python -m metalsim.render.film.render --task franka_pour --rec runs/film/franka_pour/rec \
        --out runs/film/franka_pour/hero [--grid 2] [--frames 0:390]

Writes ``frames/frame_%04d.png``, ``render.json`` (settings, per-frame seconds, a timing probe of the first frame:
one pass vs the full frame) and ``render_frames.jsonl`` (one line per frame). The frames are a cache keyed on the
recording and the render settings (``cache_key``, stored in ``render.json``): with the same key a run resumes (existing
frames are kept, unless --no-skip-existing), with another key the old frames, ``render.json`` and
``render_frames.jsonl`` move to ``stale/`` first and nothing of them (exposure, probe, hardware) is reused.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

from metalsim.render.film import provenance

PH = os.environ.get("POLYHAVEN", os.path.join(provenance.ROOT, "assets", "polyhaven"))


def look_at_wxyz(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera quaternion (w, x, y, z): camera -z toward ``target``, +y up."""
    import mujoco

    f = np.asarray(target, float) - np.asarray(eye, float); f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, -f], 1)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel())
    return q


def smooth_track(track: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    """Centred moving average over ``seconds`` (edges: shrinking window), per env. track (F, E, 3)."""
    w = max(int(round(seconds * fps)), 1)
    F = track.shape[0]
    out = np.empty_like(track)
    c = np.concatenate([np.zeros((1,) + track.shape[1:]), np.cumsum(track, 0)])
    for k in range(F):
        a, b = max(0, k - w // 2), min(F, k + w // 2 + 1)
        out[k] = (c[b] - c[a]) / (b - a)
    return out


def camera(viewer, k, j, track_s, fps=30.0, t_mid=0.0):
    """Eye and quaternion of frame ``k``, panel ``j``. ``viewer.orbit_deg_s`` turns the eye about +z through the
    look-at point, symmetric about ``t_mid`` [s] (MuJoCo Warp's README clips, ``mujoco_warp/record.py``: azimuth
    0.05 deg per 5 ms step about mid-clip = 10 deg/s)."""
    eye, lookat = np.asarray(viewer.eye, float), np.asarray(viewer.lookat, float)
    if viewer.origin_type in ("asset_root", "asset_body"):
        base = track_s[k, j]
        lookat = base + lookat
        eye = base + np.asarray(viewer.eye, float)
    orbit = float(getattr(viewer, "orbit_deg_s", 0.0) or 0.0)
    if orbit:
        a = np.deg2rad(orbit * (k / fps - t_mid))
        off = eye - lookat
        off[:2] = (np.cos(a) * off[0] - np.sin(a) * off[1], np.sin(a) * off[0] + np.cos(a) * off[1])
        eye = lookat + off
    return eye, look_at_wxyz(eye, lookat)


def _quat_mat(q):
    import mujoco

    m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(q, float)); return m.reshape(3, 3)


def _cfg_dict(x):
    return dataclasses.asdict(x) if dataclasses.is_dataclass(x) else dict(vars(x))


def cache_key(rec_dir, film, look, viewer, grid, slots) -> str:
    """Key of a shot's frames: the recording (rec.npz, rec.json, twin.xml) and every setting that changes a frame's pixels
    (tier, passes, spp, bounces, denoiser, the look, the viewer with its resolution, grid, filmed slots)."""
    h = hashlib.sha256()
    for name in ("rec.npz", "rec.json", "twin.xml"):
        h.update(name.encode())
        with open(os.path.join(rec_dir, name), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    settings = dict(tier=film.tier, passes=film.passes, spp=film.spp, bounces=film.bounces, denoise=film.denoise,
                    look=_cfg_dict(look), viewer=_cfg_dict(viewer), resolution=list(viewer.resolution), grid=grid,
                    slots=list(slots))
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def _invalidate(out_dir, key):
    """A shot directory rendered under another key (or none): move its frames, render.json and render_frames.jsonl to
    ``stale/`` (replacing an older ``stale/``) and return {} as the metadata to continue from; same key: the metadata."""
    meta_path = os.path.join(out_dir, "render.json")
    try:
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    except ValueError:
        meta = {}
    old = [p for p in (os.path.join(out_dir, "frames"), meta_path, os.path.join(out_dir, "render_frames.jsonl"))
           if os.path.exists(p)]
    if meta.get("cache_key") == key or not old:
        return meta
    n = len(glob.glob(os.path.join(out_dir, "frames", "frame_*.png")))
    stale = os.path.join(out_dir, "stale")
    shutil.rmtree(stale, ignore_errors=True)
    os.makedirs(stale)
    for p in old:
        shutil.move(p, os.path.join(stale, os.path.basename(p)))
    print(f"[render] cache key {key} differs from the shot's ({meta.get('cache_key')}): {n} old frames, render.json and "
          f"render_frames.jsonl moved to {provenance.relpath(stale)}", flush=True)
    return {}


def film_env_margin(model, R, slots, gid, eyes) -> float:
    """``env_margin`` for the ray tracer's compact env grid: its cell (the twin's content box at the initial state plus
    the margin) must hold every recorded geom of the filmed envs (centre +- bounding radius) and every camera position,
    with the growth check's slack of half the margin (env_grid.h ``env_grid_left``). Otherwise the cell grows mid-film
    (a warning and a different image) and a camera outside its cell traces through its neighbours' cells. Measured
    against the box before its x / y symmetrisation (RayTracer), so never too small."""
    from metalsim.sensors.env_grid import content_box

    lo, hi = content_box(model, max_group=3)
    X = R["xpos"][:, slots].astype(np.float64)                     # (F, E, G, 3)
    rb = np.asarray(model.geom_rbound, np.float64)[gid][None, None, :, None]
    plo = np.minimum((X - rb).min((0, 1, 2)), np.min(eyes, 0))
    phi = np.maximum((X + rb).max((0, 1, 2)), np.max(eyes, 0))
    over = max(float(np.max(lo - plo)), float(np.max(phi - hi)), 0.0)
    return max(1.0, 0.1 * float(np.max(hi - lo)), 2.0 * over + 0.5)


def build_scene(rec_dir, rec, look, viewer):
    """The render twin with the look applied: no twin lights or cameras, the film camera, the ground texture,
    material overrides, and the HDRI's sun as a directional light (``look.sun``)."""
    import mujoco

    from metalsim.render.hdr import extract_sun, load_hdr

    spec = mujoco.MjSpec.from_file(os.path.join(rec_dir, "twin.xml"))
    for l in list(spec.lights):
        spec.delete(l)
    for c in list(spec.cameras):
        spec.delete(c)
    spec.visual.headlight.active = 0
    sky, sun, yaw = None, None, np.deg2rad(look.hdri_yaw_deg)
    if look.hdri:
        path = os.path.join(PH, look.hdri)
        cache = path + ".npy"
        hdr = np.load(cache) if os.path.exists(cache) else load_hdr(path)
        sky = hdr
        if look.sun:
            sky, sun = extract_sun(hdr, yaw=yaw)
            spec.worldbody.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 10],
                                     dir=list(-np.asarray(sun["direction"])), diffuse=[1, 1, 1], specular=[0, 0, 0],
                                     ambient=[0, 0, 0], castshadow=True)
    else:
        spec.worldbody.add_light(name="key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 10],
                                 dir=[-0.4, 0.3, -0.85], diffuse=[0.9, 0.9, 0.9], castshadow=True)
    spec.worldbody.add_camera(name="film", pos=[0, 0, 1], fovy=float(viewer.fovy))
    if look.ground_texture:
        tex = spec.add_texture(name="film_ground", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               file=os.path.join(PH, look.ground_texture))
        tex.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
        mat = spec.add_material(name="film_ground", texrepeat=[look.ground_repeat, look.ground_repeat], texuniform=True,
                                rgba=[*look.ground_tint, 1.0], specular=0.3, shininess=0.2)
        mat.roughness = 0.55
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "film_ground"
        for g in spec.geoms:
            if g.name in look.ground_geoms:
                g.material = "film_ground"
    for i, (pat, kw) in enumerate(look.materials.items()):
        mname = f"film_mat_{i}"
        m = spec.add_material(name=mname, rgba=list(kw.get("rgba", (0.8, 0.8, 0.8, 1.0))))
        for key in ("roughness", "metallic", "specular"):
            if key in kw:
                setattr(m, key, float(kw[key]))
        for g in spec.geoms:
            if g.name and re.search(pat, g.name):
                g.material = mname
    model = spec.compile()
    return model, sky, sun, yaw


def render(task, rec_dir, out_dir, frames_arg=None, skip_existing=True, grid=None, envs=None):
    import imageio.v3 as iio
    import mujoco

    from metalsim.render.particles import GrainMaterial
    from metalsim.render.tier2 import Tier2Renderer

    film, viewer, look = task.film, task.viewer, task.look
    rec = json.load(open(os.path.join(rec_dir, "rec.json")))
    R = dict(np.load(os.path.join(rec_dir, "rec.npz")))      # loaded once (NpzFile re-reads per access)
    grid = grid or film.grid
    panels = grid * grid
    slots = list(envs) if envs is not None else list(range(panels))
    if len(slots) < panels or max(slots) >= R["xpos"].shape[1]:
        raise ValueError(f"need {panels} recorded envs, have {R['xpos'].shape[1]}")
    fps = rec["fps"]
    ends = R["ends"]
    F_rec = R["xpos"].shape[0]
    last = [int(ends[s]) if ends[s] >= 0 else F_rec - 1 for s in slots]
    hold = int(round(film.hold_s * fps))
    n_frames = min(max(last) + 1 + hold, film.video_length or 10**9)
    W, H = viewer.resolution
    w, h = W // grid, H // grid
    key = cache_key(rec_dir, film, look, viewer, grid, slots)
    os.makedirs(out_dir, exist_ok=True)
    meta = _invalidate(out_dir, key)
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    t0 = time.perf_counter()
    model, sky, sun, yaw = build_scene(rec_dir, rec, look, viewer)
    names = rec["geom_names"]
    gid = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm) for nm in names])
    if (gid < 0).any():
        raise RuntimeError(f"twin geoms missing after compile: {[n for n, g in zip(names, gid) if g < 0][:5]}")
    # compiled geom frames (mesh recentring) may differ between the MuJoCo versions of the two stages: correct
    C_old_p, C_old_q = R["geom_pos"], R["geom_quat"]
    C_new_p, C_new_q = model.geom_pos[gid], model.geom_quat[gid]
    fix = np.flatnonzero((np.abs(C_old_p - C_new_p).max(1) > 1e-7) | (np.abs(C_old_q - C_new_q).max(1) > 1e-7))
    fixR = [(_quat_mat(C_old_q[i]), C_old_p[i], _quat_mat(C_new_q[i]), C_new_p[i]) for i in fix]
    datas = []
    for _ in range(panels):
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        datas.append(d)
    usd_lights = None
    if sun is not None:
        usd_lights = [{"intensity": 1.0, "color": tuple(np.asarray(sun["irradiance"]) / np.pi),
                       "angle_deg": 2 * np.rad2deg(max(sun["angular_radius"], 0.0047))}]
    track_s = smooth_track(R["track"].astype(np.float64), fps, viewer.smooth_s)
    t_mid = (n_frames - 1) / (2.0 * fps)          # the orbit is symmetric about the film's middle
    eyes = np.array([camera(viewer, min(k, F_rec - 1), s, track_s, fps, t_mid)[0] for k in range(n_frames) for s in slots])
    env_margin = film_env_margin(model, R, slots, gid, eyes)
    # sites are not recorded (only geom poses are): drawn, they would sit at the twin's qpos0 pose all film (the G1's
    # torso_link_touch box at the origin), as Isaac's renderer has no sites
    vis = mujoco.MjvOption()
    vis.sitegroup[:] = 0
    r = Tier2Renderer(model, panels, width=w, height=h, camera="film", spp=film.spp, max_bounces=film.bounces,
                      material_model="omnipbr", headlight=False, tonemap="rtx", exposure=1.0, usd_lights=usd_lights,
                      denoise=film.denoise, aux=bool(film.denoise), center_sample=False, visual=vis,
                      env_margin=env_margin)
    if sky is not None:
        r.set_environment(sky, yaw=yaw)
    has_p = R["particles"].shape[2] > 0
    prad = float(rec.get("particle_radius") or 0.0)
    if has_p:
        pal = [tuple(p) for p in look.particle_palette]
        grain_mat = GrainMaterial(palette=pal, brightness_jitter=0.12)
        g_r = prad / look.grains_per_particle ** (1 / 3) * look.grain_scale
    setup_s = time.perf_counter() - t0
    print(f"[render] {task.name}: {n_frames} frames, {grid}x{grid} panels of {w}x{h}, {film.passes} x {film.spp} spp, "
          f"{len(gid)} twin geoms ({len(fix)} frame corrections), particles/env {R['particles'].shape[2]}, "
          f"env cell {np.round(r.rt.grid.S, 1).tolist()} m (margin {env_margin:.2f} m), setup {setup_s:.1f} s, "
          f"cache key {key}", flush=True)

    rng = range(n_frames)
    if frames_arg:
        p = [int(x) for x in frames_arg.split(":")]
        rng = range(p[0], min(p[1], n_frames), p[2] if len(p) > 2 else 1)
    stats_path = os.path.join(out_dir, "render_frames.jsonl")
    exposure = look.exposure
    meta_path = os.path.join(out_dir, "render.json")
    if exposure <= 0 and "exposure" in meta:        # same cache key: the exposure the shot's frames were made with
        exposure = meta["exposure"]
    frame_png = lambda k: os.path.join(out_dir, "frames", f"frame_{k:04d}.png")
    todo = [k for k in rng if not (skip_existing and os.path.exists(frame_png(k)))]
    if len(todo) < len(rng):
        print(f"[render] resuming: {len(rng) - len(todo)} of {len(rng)} frames already rendered under this key",
              flush=True)
    if os.path.exists(stats_path):                  # one stats line per frame: drop the lines of frames re-rendered now
        redo = set(todo)
        rows = [json.loads(x) for x in open(stats_path) if x.strip()]
        keep = {}
        for row in rows:
            if row.get("frame") not in redo:
                keep[row.get("frame")] = row
        if len(keep) != len(rows):
            _write_atomic(stats_path, "".join(json.dumps(row) + "\n" for row in keep.values()))

    def set_frame(k):
        P = []
        for j, s in enumerate(slots):
            f = min(k, last[j])
            gx = R["xpos"][f, s].astype(np.float64)
            gm = R["xmat"][f, s].astype(np.float64)
            if len(fix):
                for i, (Ro, po, Rn, pn) in zip(fix, fixR):
                    Rr = gm[i].reshape(3, 3)
                    Rb = Rr @ Ro.T
                    pb = gx[i] - Rb @ po
                    gm[i] = (Rb @ Rn).ravel(); gx[i] = pb + Rb @ pn
            datas[j].geom_xpos[gid] = gx
            datas[j].geom_xmat[gid] = gm
            if has_p:
                P.append(R["particles"][f, s])
        if has_p:
            r.set_particles(np.stack(P).astype(np.float32), radius=prad, material=grain_mat,
                            grains_per_particle=look.grains_per_particle, grain_radius=g_r)
        cams = [camera(viewer, k, s, track_s, fps, t_mid) for s in slots]
        return np.stack([c[0] for c in cams]), np.stack([c[1] for c in cams])

    def to_rgb(img, exposure):
        x = np.clip(img * exposure, 0, None)
        x = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
        x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)

    probe = meta.get("probe")
    done = 0
    for k in todo:
        name = frame_png(k)
        tf = time.perf_counter()
        cp, cq = set_frame(k)
        if probe is None:
            # timing probe on the first rendered frame: a warm-up pass (pipeline and structure set-up), then one pass,
            # then the full frame (per pass = (full - one) / (passes - 1))
            ts = time.perf_counter(); r.render_host(datas, passes=1, cam_pos=cp, cam_quat=cq); warm = time.perf_counter() - ts
            ts = time.perf_counter(); r.render_host(datas, passes=1, cam_pos=cp, cam_quat=cq); one = time.perf_counter() - ts
            probe = dict(frame=k, warmup_s=warm, one_pass_s=one)
            tf = time.perf_counter()
            cp, cq = set_frame(k)
        out = r.render_host(datas, passes=film.passes, cam_pos=cp, cam_quat=cq, chunk=4)
        t_render = time.perf_counter() - tf
        hdr = out["hdr_denoised"] if "hdr_denoised" in out else out["hdr"]
        hdr = np.asarray(hdr)[..., :3].reshape(panels, h, w, 3)
        if exposure <= 0:
            lum = hdr @ np.array([0.2126, 0.7152, 0.0722])
            exposure = 0.20 / max(float(np.median(lum)), 1e-6)
            print(f"[render] automatic exposure {exposure:.4f}", flush=True)
        rgb = to_rgb(hdr, exposure)
        tile = rgb.reshape(grid, grid, h, w, 3).transpose(0, 2, 1, 3, 4).reshape(grid * h, grid * w, 3)
        tmp = os.path.join(out_dir, "frames", f".frame_{k:04d}.tmp.png")   # an interrupted write leaves no frame
        iio.imwrite(tmp, tile)
        os.replace(tmp, name)
        el = time.perf_counter() - tf
        if "full_s" not in probe:
            probe.update(full_s=t_render, per_pass_s=(t_render - probe["one_pass_s"]) / max(film.passes - 1, 1))
        st = dict(frame=k, seconds=el, render_s=t_render, grains=int(r.grains.n_grains) if getattr(r, "grains", None) else 0)
        with open(stats_path, "a") as f:
            f.write(json.dumps(st) + "\n")
        done += 1
        meta = dict(task=task.name, grid=grid, slots=slots, envs=[rec["envs"][s] for s in slots], frames=n_frames,
                    fps=fps, width=W, height=H, panel=(w, h), tier=2, passes=film.passes, spp=film.spp,
                    samples_per_pixel=film.passes * film.spp, bounces=film.bounces, denoise=film.denoise,
                    exposure=exposure, viewer=vars(viewer) if not hasattr(viewer, "__dataclass_fields__") else
                    {k_: getattr(viewer, k_) for k_ in viewer.__dataclass_fields__}, look=dict(
                        hdri=look.hdri, sun=look.sun, ground_texture=look.ground_texture,
                        grains_per_particle=look.grains_per_particle, grain_radius=g_r if has_p else None),
                    particles_per_env=int(R["particles"].shape[2]), particle_radius=prad, setup_s=setup_s,
                    probe=probe, twin_geoms=len(gid), frame_corrections=len(fix),
                    env_grid=dict(margin=env_margin, cell=r.rt.grid.S.tolist(), growths=int(r.rt.grid_growths)),
                    cache_key=key, software=provenance.software(("warp", "mujoco", "numpy")),
                    hardware=meta["hardware"] if "hardware" in meta else provenance.hardware())
        _write_atomic(meta_path, json.dumps(meta, indent=1, default=str))
        if done == 1 or k % 15 == 0:
            print(f"[render] frame {k}/{n_frames}: {el:.2f} s (render {t_render:.2f} s)", flush=True)
    print(f"[render] done: {done} frames rendered -> {provenance.relpath(out_dir)}", flush=True)
    return meta


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv=None):
    from metalsim.render.film import tasks
    from metalsim.render.film.cli import add_override_args, apply_overrides

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None, help="start:stop[:step]")
    ap.add_argument("--slots", default=None, help="recorded env slots to film (comma-separated; default 0..grid^2-1)")
    ap.add_argument("--no-skip-existing", action="store_true")
    add_override_args(ap)
    a = ap.parse_args(argv)
    task = apply_overrides(tasks.get(a.task), a)
    slots = [int(x) for x in a.slots.split(",")] if a.slots else None
    render(task, a.rec, a.out, a.frames, not a.no_skip_existing, grid=task.film.grid, envs=slots)


if __name__ == "__main__":
    sys.exit(main())
