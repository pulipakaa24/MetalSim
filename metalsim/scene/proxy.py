"""Collision proxies for seen-only backgrounds (Gaussian splats): the geometry the physics collides with where the
splat is drawn. Research (docs/SCENES.md "Splat backgrounds"): Isaac Lab's NuRec recipe adds a ground plane at z = 0;
NVIDIA's NuRec scenes ship a mesh (nvblox) referenced by the volume's ``proxy``; GaussGym (NKSR), Vid2Sim (TSDF of
rendered depth) and SplatGym (octree of points) build one; InteriorGS ships no mesh but a floor plan
(``structure.json``) and a 2D occupancy map (``occupancy.png`` / ``occupancy.json``).

Every proxy is added to an ``mjSpec`` in geom group ``PROXY_GROUP`` (4): the renderers draw groups 0-3 by default, so a
proxy collides but is never drawn over the splat (``CollisionProxyCfg.visible`` puts it in group 0 for debugging).
"""
from __future__ import annotations

import json
import os

import numpy as np

PROXY_GROUP = 4


def _geom_kw(cfg, name):
    return dict(name=name, group=0 if cfg.visible else PROXY_GROUP, friction=[cfg.friction, 0.005, 0.0001],
                rgba=[0.3, 0.6, 0.9, 0.35])


def add_plane(spec, cfg, z: float | None = None, name="proxy_plane"):
    import mujoco

    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1],
                            pos=[0, 0, cfg.plane_z if z is None else z], **_geom_kw(cfg, name))
    return 1


def read_mesh(path: str):
    """Vertices (V, 3) and triangles (T, 3) from .obj / .stl (trimesh if installed, else a minimal OBJ reader) or a
    USD file's meshes (every UsdGeom.Mesh, world transforms applied, fans triangulated)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".usd", ".usda", ".usdc", ".usdz"):
        from pxr import Usd, UsdGeom

        st = Usd.Stage.Open(path)
        mpu = UsdGeom.GetStageMetersPerUnit(st) or 1.0
        up = UsdGeom.GetStageUpAxis(st)
        V, F, off = [], [], 0
        for p in st.Traverse():
            if not p.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(p)
            pts = np.asarray(m.GetPointsAttr().Get() or [], np.float64)
            if not len(pts):
                continue
            X = np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), np.float64)
            pts = pts @ X[:3, :3] + X[3, :3]
            counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int64)
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64)
            k = 0
            for c in counts:
                for j in range(1, c - 1):
                    F.append((idx[k] + off, idx[k + j] + off, idx[k + j + 1] + off))
                k += c
            V.append(pts); off += len(pts)
        v = np.concatenate(V) * mpu
        if up == "Y":
            v = v @ np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # y-up -> z-up
        return v, np.asarray(F, np.int64)
    try:
        import trimesh

        m = trimesh.load(path, force="mesh")
        return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    except ImportError:
        pass
    if ext != ".obj":
        raise ImportError(f"{path}: install trimesh to read {ext} meshes")
    V, F = [], []
    for line in open(path):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            V.append([float(x) for x in s[1:4]])
        elif s[0] == "f":
            ids = [int(x.split("/")[0]) for x in s[1:]]
            ids = [i - 1 if i > 0 else len(V) + i for i in ids]
            F += [(ids[0], ids[j], ids[j + 1]) for j in range(1, len(ids) - 1)]
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def add_mesh(spec, cfg, A=None, t=None, name="proxy_mesh"):
    """A static mesh proxy; ``A``, ``t``: the splat's file -> sim map, applied unless ``cfg.mesh_in_sim_frame``.
    MuJoCo collides meshes as their convex hulls, so a room mesh is split into convex-friendly pieces only by its
    connected components; for rooms prefer "structure" / "occupancy" (boxes and a height field)."""
    import mujoco

    v, f = read_mesh(cfg.mesh_path)
    if A is not None and not cfg.mesh_in_sim_frame:
        v = v @ np.asarray(A).T + np.asarray(t)
    spec.add_mesh(name=name, uservert=v.astype(np.float32).ravel().tolist(), userface=f.ravel().tolist())
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, **_geom_kw(cfg, name + "_geom"))
    return 1


def _walls_from_polygon(poly):
    poly = np.asarray(poly, float)[:, :2]
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def structure_walls(structure_json: str):
    """Wall segments ((x0, y0), (x1, y1)) of an InteriorGS ``structure.json``: its ``walls`` entries when present
    (keys tried: start/end, points, location), else the edges of every room's floor ``profile`` polygon."""
    s = json.load(open(structure_json))
    segs = []
    for w in s.get("walls", []) or []:
        if isinstance(w, dict):
            if "start" in w and "end" in w:
                segs.append((np.asarray(w["start"], float)[:2], np.asarray(w["end"], float)[:2]))
                continue
            pts = w.get("points") or w.get("location") or w.get("profile")
            if pts is not None and len(pts) >= 2:
                pts = np.asarray(pts, float)[:, :2]
                segs += [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if not segs:
        for r in s.get("rooms", []) or []:
            prof = r.get("profile", [])
            if len(prof) >= 3:
                segs += _walls_from_polygon(prof)
    # merge duplicates (shared walls of adjacent rooms)
    out, seen = [], set()
    for a, b in segs:
        key = tuple(np.round(np.concatenate([np.minimum(a, b), np.maximum(a, b)]), 3))
        if key not in seen and np.linalg.norm(b - a) > 1e-3:
            seen.add(key); out.append((a, b))
    return out


def add_structure(spec, cfg, name="proxy_wall"):
    """Floor plane (z = ``plane_z``) + one box per wall segment, ``wall_height`` x ``wall_thickness``."""
    import mujoco

    n = add_plane(spec, cfg, name="proxy_floor")
    for i, (a, b) in enumerate(structure_walls(cfg.structure_json)):
        d = b - a
        L = float(np.linalg.norm(d))
        yaw = float(np.arctan2(d[1], d[0]))
        c = (a + b) / 2
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[L / 2 + cfg.wall_thickness / 2, cfg.wall_thickness / 2, cfg.wall_height / 2],
                                pos=[c[0], c[1], cfg.plane_z + cfg.wall_height / 2],
                                quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], **_geom_kw(cfg, f"{name}_{i}"))
        n += 1
    return n


def occupancy_grid(png_path: str, json_path: str):
    """InteriorGS 2D occupancy (255 free, 0 occupied, 127 unknown) -> (occupied (H, W) bool, origin xy of pixel (0, 0)'s
    corner [m], resolution [m/px], row axis sign). ``occupancy.json`` keys tried: resolution / scale, origin /
    lower_left / min; image rows run along -y (image convention) unless ``row_up`` is true."""
    import imageio.v3 as iio

    img = np.asarray(iio.imread(png_path))
    if img.ndim == 3:
        img = img[..., 0]
    meta = json.load(open(json_path))
    res = float(meta.get("resolution", meta.get("scale", meta.get("meters_per_pixel", 0.0))))
    org = meta.get("origin", meta.get("lower_left", meta.get("min")))
    if not res or org is None:
        raise ValueError(f"{json_path}: need resolution and origin (got keys {sorted(meta)})")
    return img < 64, np.asarray(org, float)[:2], res, bool(meta.get("row_up", False))


def add_occupancy(spec, cfg, png_path, json_path, name="proxy_occupancy"):
    """Occupied cells as a height field ``wall_height`` tall (free / unknown cells at the floor) + the floor plane."""
    import mujoco

    occ, org, res, row_up = occupancy_grid(png_path, json_path)
    H, W = occ.shape
    hz = occ[::-1] if not row_up else occ          # MuJoCo hfield rows run along +y
    # MuJoCo samples a height field at nrow x ncol grid points spanning 2 x size: put them at the pixel centres
    spec.add_hfield(name=name, size=[(W - 1) * res / 2, (H - 1) * res / 2, cfg.wall_height, 0.01], nrow=H, ncol=W,
                    userdata=hz.astype(np.float32).ravel().tolist())
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                            pos=[org[0] + W * res / 2, org[1] + H * res / 2, cfg.plane_z], **_geom_kw(cfg, name + "_geom"))
    return 1 + add_plane(spec, cfg, name="proxy_floor")


def add_proxy(spec, cfg, A=None, t=None) -> dict:
    """Add the proxy ``cfg`` (``scene.cfg.CollisionProxyCfg``) to ``spec``; returns a summary for logs / docs."""
    kind = cfg.type
    if kind == "none":
        n = 0
    elif kind == "plane":
        n = add_plane(spec, cfg)
    elif kind == "mesh":
        n = add_mesh(spec, cfg, A, t)
    elif kind == "structure":
        n = add_structure(spec, cfg)
    elif kind == "occupancy":
        base = os.path.dirname(cfg.structure_json or cfg.mesh_path or "")
        n = add_occupancy(spec, cfg, os.path.join(base, "occupancy.png"), os.path.join(base, "occupancy.json"))
    else:
        raise ValueError(f"collision proxy type {kind!r}: plane | mesh | structure | occupancy | none")
    return dict(type=kind, geoms=n, group=0 if cfg.visible else PROXY_GROUP)
