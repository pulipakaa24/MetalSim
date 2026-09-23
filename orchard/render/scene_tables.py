"""Compiled scene tables for the batched renderer, built once from a MuJoCo model.

This is the "compile step that flattens the stage into device tables" (WS1): unique meshes
(indexed, smooth normals, UVs), an instance table of (env, slot) pairs per unique mesh, a
material block per drawn geom (base color, PBR parameters, atlas rect), a texture atlas, a
semantic table (geom -> body, model geom id), and camera intrinsics. mjbatch-metal's
unique-mesh instancing is the seed; the material model is extended for PBR shading.

Everything here is numpy; the renderer uploads the arrays into Metal buffers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

_BOX_FACES = np.array([
    [[1, -1, -1], [1, 1, -1], [1, 1, 1]], [[1, -1, -1], [1, 1, 1], [1, -1, 1]],
    [[-1, -1, -1], [-1, 1, 1], [-1, 1, -1]], [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1]],
    [[-1, 1, -1], [-1, 1, 1], [1, 1, 1]], [[-1, 1, -1], [1, 1, 1], [1, 1, -1]],
    [[-1, -1, -1], [1, -1, -1], [1, -1, 1]], [[-1, -1, -1], [1, -1, 1], [-1, -1, 1]],
    [[-1, -1, 1], [1, -1, 1], [1, 1, 1]], [[-1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    [[-1, -1, -1], [-1, 1, -1], [1, 1, -1]], [[-1, -1, -1], [1, 1, -1], [1, -1, -1]],
], dtype=np.float64)


def _uv_sphere(radius, n_lat=12, n_lon=24):
    tris, uvs = [], []
    for i in range(n_lat):
        t0, t1 = np.pi * i / n_lat, np.pi * (i + 1) / n_lat
        for j in range(n_lon):
            p0, p1 = 2 * np.pi * j / n_lon, 2 * np.pi * (j + 1) / n_lon
            pts = {
                "a": ([np.sin(t0) * np.cos(p0), np.sin(t0) * np.sin(p0), np.cos(t0)], [j / n_lon, i / n_lat]),
                "b": ([np.sin(t1) * np.cos(p0), np.sin(t1) * np.sin(p0), np.cos(t1)], [j / n_lon, (i + 1) / n_lat]),
                "c": ([np.sin(t1) * np.cos(p1), np.sin(t1) * np.sin(p1), np.cos(t1)], [(j + 1) / n_lon, (i + 1) / n_lat]),
                "d": ([np.sin(t0) * np.cos(p1), np.sin(t0) * np.sin(p1), np.cos(t0)], [(j + 1) / n_lon, i / n_lat]),
            }
            for tri in (("a", "b", "c"), ("a", "c", "d")):
                tris.append([pts[k][0] for k in tri])
                uvs.append([pts[k][1] for k in tri])
    return np.asarray(tris) * radius, np.asarray(uvs)


def _cylinder(radius, half_len, n=32, caps=True):
    tris = []
    for j in range(n):
        p0, p1 = 2 * np.pi * j / n, 2 * np.pi * (j + 1) / n
        a = [radius * np.cos(p0), radius * np.sin(p0), -half_len]
        b = [radius * np.cos(p1), radius * np.sin(p1), -half_len]
        c = [radius * np.cos(p1), radius * np.sin(p1), half_len]
        d = [radius * np.cos(p0), radius * np.sin(p0), half_len]
        tris += [[a, b, c], [a, c, d]]
        if caps:
            tris += [[[0, 0, half_len], d, c], [[0, 0, -half_len], b, a]]
    return np.asarray(tris)


def _capsule(radius, half_len, n_lon=24, n_lat=12):
    body = _cylinder(radius, half_len, n=n_lon, caps=False)
    sph, _ = _uv_sphere(radius, n_lat=n_lat, n_lon=n_lon)
    top = sph[sph[:, :, 2].mean(axis=1) >= -1e-9] + [0, 0, half_len]
    bot = sph[sph[:, :, 2].mean(axis=1) <= 1e-9] - [0, 0, half_len]
    return np.concatenate([body, top, bot])


def _ellipsoid(size, n_lat=12, n_lon=24):
    sph, uv = _uv_sphere(1.0, n_lat, n_lon)
    return sph * np.asarray(size), uv


def _tris_to_indexed(tri, uv=None, smooth=True):
    """(F,3,3) triangle soup -> indexed vertices with smooth (or flat) normals and UVs."""
    flat = tri.reshape(-1, 3)
    uv_flat = np.zeros((len(flat), 2)) if uv is None else uv.reshape(-1, 2)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    if smooth:
        key = np.round(np.concatenate([flat, uv_flat], axis=1), 7)
    else:
        key = np.round(np.concatenate([flat, uv_flat, np.repeat(fn, 3, axis=0)], axis=1), 7)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = flat[first]
    uvs = uv_flat[first]
    idx = inverse.reshape(-1, 3).astype(np.uint32)
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, idx[:, k], fn)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    return verts, normals, uvs, idx


@dataclass
class TextureAtlas:
    image: np.ndarray                      # (H, W, 4) uint8
    rects: dict = field(default_factory=dict)  # texid -> (u0, v0, du, dv)

    @classmethod
    def build(cls, m: mujoco.MjModel):
        if m.ntex == 0:
            return cls(np.full((4, 4, 4), 255, np.uint8), {})
        pad = 1
        usable = [t for t in range(m.ntex) if int(m.tex_type[t]) != int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)]
        if not usable:
            return cls(np.full((4, 4, 4), 255, np.uint8), {})
        widths = [int(m.tex_width[t]) for t in range(m.ntex)]
        heights = [int(m.tex_height[t]) for t in range(m.ntex)]
        aw = max(widths[t] for t in usable) + 2 * pad
        order = sorted(usable, key=lambda t: -heights[t])
        x = y = shelf_h = 0
        pos = {}
        for t in order:
            w, h = widths[t] + 2 * pad, heights[t] + 2 * pad
            if x + w > aw:
                y += shelf_h
                x = 0
                shelf_h = 0
            pos[t] = (x + pad, y + pad)
            x += w
            shelf_h = max(shelf_h, h)
        ah = y + shelf_h
        img = np.zeros((ah, aw, 4), np.uint8)
        img[..., 3] = 255
        rects = {}
        for t in range(m.ntex):
            if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
                continue  # environment maps are handled separately, not atlased
            w, h = widths[t], heights[t]
            nc = int(m.tex_nchannel[t])
            adr = int(m.tex_adr[t])
            data = m.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)
            px, py = pos[t]
            img[py:py + h, px:px + w, :3] = data[::-1, :, :3]   # MuJoCo rows are bottom-up
            if nc == 4:
                img[py:py + h, px:px + w, 3] = data[::-1, :, 3]
            rects[t] = (px / aw, py / ah, w / aw, h / ah)
        return cls(img, rects)


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in the MuJoCo camera frame (looks along -Z, +Y up)."""
    fx: float   # focal length in pixels (for the given image size)
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.01
    far: float = 10.0

    @classmethod
    def from_model(cls, m: mujoco.MjModel, cam: int, width: int, height: int, near=0.01, far=10.0):
        """Uses `sensorsize`/`focal`/`principal` when authored (Isaac-style physical camera),
        else `fovy` (MuJoCo's default)."""
        sensorsize = m.cam_sensorsize[cam]
        if sensorsize[0] > 0 and sensorsize[1] > 0:
            focal = m.cam_intrinsic[cam][:2]           # focal length (m)
            principal = m.cam_intrinsic[cam][2:4]      # principal point offset (m)
            fx = focal[0] / sensorsize[0] * width
            fy = focal[1] / sensorsize[1] * height
            cx = width / 2 + principal[0] / sensorsize[0] * width
            cy = height / 2 + principal[1] / sensorsize[1] * height
        else:
            fovy = np.deg2rad(m.cam_fovy[cam])
            fy = height / (2 * np.tan(fovy / 2))
            fx = fy
            cx, cy = width / 2, height / 2
        return cls(fx, fy, cx, cy, width, height, near, far)

    def projection(self) -> np.ndarray:
        """OpenGL-style clip matrix (column vectors), depth in [0,1] (Metal convention)."""
        w, h, n, f = self.width, self.height, self.near, self.far
        P = np.zeros((4, 4))
        P[0, 0] = 2 * self.fx / w
        P[1, 1] = 2 * self.fy / h
        P[0, 2] = 1 - 2 * self.cx / w
        P[1, 2] = 2 * self.cy / h - 1
        P[2, 2] = f / (n - f)
        P[2, 3] = n * f / (n - f)
        P[3, 2] = -1.0
        return P


def build_lights(m: mujoco.MjModel) -> np.ndarray:
    """(nlight, 20) float32 per model light: type (0 dir, 1 point, 2 spot), castshadow, cutoff(deg),
    exponent | diffuse rgb, _ | specular rgb, _ | ambient rgb, _ | attenuation xyz, bodyid."""
    L = np.zeros((max(m.nlight, 1), 20), np.float32)
    for i in range(m.nlight):
        lt = int(m.light_type[i]) if hasattr(m, "light_type") else (0 if m.light_directional[i] else 2)
        # mjLIGHT_SPOT=0, DIRECTIONAL=1, POINT=2, IMAGE=3 in MuJoCo >= 3.3
        kind = {1: 0, 2: 1, 0: 2}.get(lt, 0)
        L[i, 0:4] = (kind, float(m.light_castshadow[i]), float(m.light_cutoff[i]), float(m.light_exponent[i]))
        L[i, 4:7] = m.light_diffuse[i]
        L[i, 8:11] = m.light_specular[i]
        L[i, 12:15] = m.light_ambient[i]
        L[i, 16:19] = m.light_attenuation[i]
        L[i, 19] = float(m.light_bodyid[i])
    return L


def scene_params(m: mujoco.MjModel) -> np.ndarray:
    """(5, 4) float32: bounds center+radius | headlight ambient | headlight diffuse | headlight specular + active | sky colour + present."""
    hl = m.vis.headlight
    P = np.zeros((5, 4), np.float32)
    P[0, :3] = m.stat.center; P[0, 3] = max(float(m.stat.extent), 0.1)
    P[1, :3] = hl.ambient; P[2, :3] = hl.diffuse; P[3, :3] = hl.specular; P[3, 3] = float(hl.active)
    # sky: mean colour of the skybox texture (reflection misses and default background), w = present
    for t in range(m.ntex):
        if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
            adr = int(m.tex_adr[t])
            img = m.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)[:, :, :3].astype(np.float32) / 255.0
            P[4, :3] = img.mean(axis=(0, 1)); P[4, 3] = 1.0
            break
    return P


@dataclass
class SceneTables:
    # geometry
    vertices: np.ndarray        # (V, 8) float32: pos, normal, uv
    indices: np.ndarray         # (I,) uint32
    meshes: list                # per unique mesh: dict(i_off, i_count, radius)
    # instancing
    geoms: list                 # slot -> model geom id
    geom_mesh: np.ndarray       # slot -> unique mesh index
    draws: list                 # per unique mesh: (i_off, i_count, first_instance, instance_count)
    inst_table: np.ndarray      # (n_envs * G, 2) uint32 (env, slot) grouped by mesh
    # materials and semantics
    materials: np.ndarray       # (G, 16) float32, see MATERIAL_LAYOUT
    atlas: TextureAtlas
    semantic: np.ndarray        # (G, 4) int32: model geom id, body id, root body id, geom group
    # static (world-frame) flags: geoms whose pose never changes (bodyid 0)
    static_geom: np.ndarray     # (G,) bool
    lights: np.ndarray          # (max(nlight,1), 20) float32, see build_lights
    params: np.ndarray          # (4, 4) float32, see scene_params
    n_envs: int
    G: int

    # material block layout (float4 x 4)
    # 0: rgba (base color); 1: metallic, roughness, specular, emission; 2: atlas u0,v0,du,dv;
    # 3: texrepeat.xy, textured flag, reflectance
    MATERIAL_LAYOUT = ("rgba", "metallic_roughness_specular_emission", "atlas_rect", "texrepeat_flag_reflectance")


def _weld(v, f, decimals=6):
    """Merge coincident vertices (CAD/STL meshes are unwelded triangle soups, which no edge-collapse
    simplifier can reduce)."""
    key = np.round(v, decimals)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    f2 = inverse.reshape(-1)[f]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    return v[first], f2[keep]


def _cluster(v, f, cell):
    """Vertex-clustering decimation: merge vertices per grid cell, drop degenerate/duplicate faces."""
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    nv = inv.max() + 1
    newv = np.zeros((nv, 3)); cnt = np.zeros(nv)
    np.add.at(newv, inv, v); np.add.at(cnt, inv, 1)
    newv /= cnt[:, None]
    f2 = inv[f]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    f2 = f2[keep]
    if len(f2):
        # drop duplicate faces (same vertex set) but keep original winding of the first occurrence
        _, first = np.unique(np.sort(f2, axis=1), axis=0, return_index=True)
        f2 = f2[np.sort(first)]
    return newv, f2


def _decimate(v, f, target_faces):
    """Level-of-detail decimation of a welded mesh; UVs are dropped (returns None).

    Quadric edge collapse (fast_simplification) first; CAD meshes with dense small features often
    stall well above the target, in which case vertex clustering with a bisected cell size is used.
    """
    v, f = _weld(v, f)
    if len(f) <= target_faces:
        return v, f
    try:
        import fast_simplification
        v_d, f_d = fast_simplification.simplify(np.ascontiguousarray(v, np.float64), np.ascontiguousarray(f, np.int64),
                                                target_count=int(target_faces), agg=8)
        v_d, f_d = np.asarray(v_d), np.asarray(f_d)
        if len(f_d) <= 1.5 * target_faces:
            return v_d, f_d
    except ImportError:
        pass
    ext = float((v.max(0) - v.min(0)).max())
    lo, hi = ext / 512, ext / 4          # cell sizes: small -> many faces, large -> few
    best = None
    for _ in range(14):
        cell = np.sqrt(lo * hi)
        v_c, f_c = _cluster(v, f, cell)
        if len(f_c) > target_faces:
            lo = cell
        else:
            hi = cell
            best = (v_c, f_c)
    if best is None:
        best = _cluster(v, f, hi)
    return best


def build_scene_tables(m: mujoco.MjModel, n_envs: int, *, max_group: int = 3, include_planes: bool = True,
                       plane_extent: float = 5.0, smooth: bool = True, decimate_faces: int = 0) -> SceneTables:
    """``decimate_faces``: per-mesh triangle budget for a level of detail (0 = undecimated)."""
    mesh_cache = {}
    meshes, geoms, geom_mesh = [], [], []
    all_v, all_n, all_uv, all_i = [], [], [], []
    v_off = i_off = 0
    atlas = TextureAtlas.build(m)

    def geom_tris(g):
        t = int(m.geom_type[g])
        size = m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            v_full = m.mesh_vert[va:va + vn].astype(np.float64)
            f_full = m.mesh_face[fa:fa + fn].astype(np.int64)
            uv = None
            tca = int(m.mesh_texcoordadr[mid])
            if tca >= 0:
                ftc = m.mesh_facetexcoord[fa:fa + fn].astype(np.int64)
                uv = m.mesh_texcoord[tca + ftc]
            if decimate_faces and len(f_full) > decimate_faces:
                v_d, f_d = _decimate(v_full, f_full, decimate_faces)
                return v_d[f_d], None
            return v_full[f_full], uv
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            tri = _BOX_FACES * size
            uv = np.zeros((len(tri), 3, 2))
            for f in range(len(tri)):
                span = tri[f].max(axis=0) - tri[f].min(axis=0)
                axes = np.argsort(span)[-2:]
                lo = tri[f][:, axes].min(axis=0)
                rng_ = np.maximum(tri[f][:, axes].max(axis=0) - lo, 1e-9)
                uv[f] = (tri[f][:, axes] - lo) / rng_
            return tri, uv
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            return _uv_sphere(size[0])
        if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return _ellipsoid(size[:3])
        if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return _cylinder(size[0], size[1]), None
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return _capsule(size[0], size[1]), None
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            s = (float(size[0]) if size[0] > 0 else plane_extent, float(size[1]) if size[1] > 0 else plane_extent)
            tri = np.array([[[-s[0], -s[1], 0], [s[0], -s[1], 0], [s[0], s[1], 0]],
                            [[-s[0], -s[1], 0], [s[0], s[1], 0], [-s[0], s[1], 0]]])
            uv = (tri[:, :, :2] / (2 * np.array(s)) + 0.5)
            return tri, uv
        return None, None

    for g in range(m.ngeom):
        if m.geom_group[g] >= max_group:
            continue
        t = int(m.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_PLANE and not include_planes:
            continue
        if m.geom_rgba[g][3] == 0.0 and int(m.geom_matid[g]) < 0:
            continue
        key = ("mesh", int(m.geom_dataid[g])) if t == mujoco.mjtGeom.mjGEOM_MESH else (t, tuple(np.round(m.geom_size[g], 6)))
        if key not in mesh_cache:
            tri, uv = geom_tris(g)
            if tri is None:
                continue
            verts, normals, uvs, idx = _tris_to_indexed(tri, uv, smooth=smooth and t == mujoco.mjtGeom.mjGEOM_MESH
                                                        or t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                                 mujoco.mjtGeom.mjGEOM_ELLIPSOID))
            mesh_cache[key] = len(meshes)
            meshes.append({"i_off": i_off, "i_count": int(idx.size),
                           "radius": float(np.linalg.norm(verts, axis=1).max())})
            all_v.append(verts); all_n.append(normals); all_uv.append(uvs)
            all_i.append(idx.reshape(-1) + v_off)
            v_off += len(verts)
            i_off += idx.size
        elif geom_tris(g)[0] is None:
            continue
        geom_mesh.append(mesh_cache[key])
        geoms.append(g)
    if not geoms:
        raise ValueError("no drawable geoms (check max_group / include_planes)")

    G = len(geoms)
    verts = np.concatenate(all_v).astype(np.float32)
    inter = np.zeros((len(verts), 8), np.float32)
    inter[:, 0:3] = verts
    inter[:, 3:6] = np.concatenate(all_n).astype(np.float32)
    inter[:, 6:8] = np.concatenate(all_uv).astype(np.float32)
    indices = np.concatenate(all_i).astype(np.uint32)

    draws, table = [], []
    first = 0
    geom_mesh = np.asarray(geom_mesh)
    for mi in range(len(meshes)):
        slots = np.flatnonzero(geom_mesh == mi)
        pairs = [(e, int(s)) for s in slots for e in range(n_envs)]
        table.extend(pairs)
        draws.append((meshes[mi]["i_off"], meshes[mi]["i_count"], first, len(pairs)))
        first += len(pairs)
    inst_table = np.asarray(table, np.uint32).reshape(-1, 2)

    mats = np.zeros((G, 16), np.float32)
    semantic = np.zeros((G, 4), np.int32)
    static = np.zeros(G, bool)
    for slot, g in enumerate(geoms):
        mat = int(m.geom_matid[g])
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        mats[slot, 0:4] = rgba
        if mat >= 0:
            # MuJoCo leaves metallic/roughness at -1 when unset; derive roughness from shininess then
            metallic = float(m.mat_metallic[mat]) if hasattr(m, "mat_metallic") else -1.0
            roughness = float(m.mat_roughness[mat]) if hasattr(m, "mat_roughness") else -1.0
            if metallic < 0:
                metallic = 0.0
            if roughness < 0:
                roughness = 1.0 - float(m.mat_shininess[mat])
            mats[slot, 4:8] = (metallic, float(np.clip(roughness, 0.02, 1.0)), float(m.mat_specular[mat]),
                               float(m.mat_emission[mat]))
            mats[slot, 15] = float(m.mat_reflectance[mat])
            texid = -1
            if m.ntex:
                roles = np.asarray(m.mat_texid[mat]).reshape(-1)
                texid = int(roles[1]) if len(roles) > 1 else int(roles[0])   # mjTEXROLE_RGB
            if texid >= 0 and texid in atlas.rects:
                mats[slot, 8:12] = atlas.rects[texid]
                mats[slot, 12:14] = m.mat_texrepeat[mat]
                if m.mat_texuniform[mat]:
                    # repeat is per world unit: scale by the extent the geom's UVs span
                    gt = int(m.geom_type[g])
                    if gt == mujoco.mjtGeom.mjGEOM_PLANE:
                        ext = np.array([m.geom_size[g][0] if m.geom_size[g][0] > 0 else plane_extent,
                                        m.geom_size[g][1] if m.geom_size[g][1] > 0 else plane_extent]) * 2
                    else:
                        ext = np.sort(m.geom_size[g][:3])[-2:] * 2
                    mats[slot, 12:14] = m.mat_texrepeat[mat] * ext
                mats[slot, 14] = 1.0
        else:
            mats[slot, 4:8] = (0.0, 0.6, 0.5, 0.0)
        body = int(m.geom_bodyid[g])
        semantic[slot] = (g, body, int(m.body_rootid[body]), int(m.geom_group[g]))
        static[slot] = body == 0
    return SceneTables(vertices=inter, indices=indices, meshes=meshes, geoms=geoms, geom_mesh=geom_mesh,
                       draws=draws, inst_table=inst_table, materials=mats, atlas=atlas, semantic=semantic,
                       static_geom=static, lights=build_lights(m), params=scene_params(m), n_envs=n_envs, G=G)


def quats_to_mats(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    M = np.empty((len(q), 3, 3))
    M[:, 0, 0] = 1 - 2 * (y * y + z * z); M[:, 0, 1] = 2 * (x * y - w * z); M[:, 0, 2] = 2 * (x * z + w * y)
    M[:, 1, 0] = 2 * (x * y + w * z); M[:, 1, 1] = 1 - 2 * (x * x + z * z); M[:, 1, 2] = 2 * (y * z - w * x)
    M[:, 2, 0] = 2 * (x * z - w * y); M[:, 2, 1] = 2 * (y * z + w * x); M[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return M
