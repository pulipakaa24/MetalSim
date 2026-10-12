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

from metalsim.render import builtin_geoms as bg

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


def _octagon_rings(apothem: float, inner: float = 0.25):
    """(F,3,3) triangle soup of a z = 0 disc: concentric regular octagons whose apothems double from ``inner`` to
    ``apothem`` (an infinite plane's drawn extent). Triangles stay small where the camera is (the raster translates
    the disc under the camera), so interpolated positions and texture coordinates keep float32 precision near the
    viewer whatever the far distance; ~16 triangles per doubling (170 for a 1e6 m disc). Counter-clockwise from +z."""
    radii = [min(inner, apothem)]
    while radii[-1] < apothem:
        radii.append(min(2.0 * radii[-1], apothem))
    ang = (np.arange(8) + 0.5) * (np.pi / 4)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(8)], 1) / np.cos(np.pi / 8)     # unit apothem
    tris = []
    p = ring * radii[0]
    for k in range(8):
        tris.append([[0.0, 0.0, 0.0], p[k], p[(k + 1) % 8]])
    for a0, a1 in zip(radii[:-1], radii[1:]):
        q0, q1 = ring * a0, ring * a1
        for k in range(8):
            k1 = (k + 1) % 8
            tris += [[q0[k], q1[k], q1[k1]], [q0[k], q1[k1], q0[k1]]]
    return np.asarray(tris, np.float64)


def plane_uv_affine(size, appearance: str, texrepeat=(1.0, 1.0), texuniform: bool = False):
    """Texture coordinates of a plane disc vertex as an affine map of its plane-local (x, y), plus the material repeat
    the legacy tables need: (u_scale, u_add, v_scale, v_add, rep_u, rep_v). "mujoco" appearance: MuJoCo's makePlane
    coordinates (``builtin_geoms.primitive``: finite u = (x + s) / 2s, v = 1 - (y + s) / 2s; infinite u = x / 2,
    v = -y / 2), the material's scale and offset applied as for any plane (``material_uv_transform``; the repeat
    returned is unused). "legacy": ``plane_texture_map`` for the legacy atlas."""
    if appearance != "mujoco":
        return plane_texture_map(size, texrepeat, texuniform)
    sx, sy = float(size[0]), float(size[1])
    us, ua = (1.0 / (2.0 * sx), 0.5) if sx > 0 else (0.5, 0.0)
    vs, va = (-1.0 / (2.0 * sy), 0.5) if sy > 0 else (-0.5, 0.0)
    return us, ua, vs, va, 1.0, 1.0


def plane_texture_map(size, texrepeat, texuniform: bool):
    """MuJoCo's plane texture coordinates (render_context.c; MuJoCo Warp ``_texture_plane``) as an affine map of the
    plane-local (x, y) for this renderer's convention (fragment: ``fract(uv * par.xy)``, atlas rows top-down):
    returns (u_scale, u_add, v_scale, v_add, rep_u, rep_v) with uv = (x * u_scale + u_add, y * v_scale + v_add) and
    rep = par.xy. Per axis: infinite (size <= 0) u = x / 2 at ``texrepeat``; finite u = (x + s) / 2s at ``texrepeat``,
    times s with ``texuniform``; MuJoCo's -0.5 offset (applied when either axis is infinite) shifts by half a tile.
    The v axis is mirrored (MuJoCo's v = (s - y) / 2s, our atlas is stored bottom-up)."""
    sx, sy = float(size[0]), float(size[1])
    rx, ry = float(texrepeat[0]), float(texrepeat[1])
    off = -0.5 if (sx <= 0 or sy <= 0) else 0.0
    rep_u = rx * sx if (sx > 0 and texuniform) else rx
    rep_v = ry * sy if (sy > 0 and texuniform) else ry
    ru, rv = (r if abs(r) > 1e-9 else 1.0 for r in (rep_u, rep_v))
    # u axis: MuJoCo coordinate u * rep + off; ours uv.x * rep -> uv.x = u + off / rep
    u_scale, u_add = (1.0 / (2.0 * sx), 0.5) if sx > 0 else (0.5, 0.0)
    u_add += off / ru
    # v axis: MuJoCo v = (s - y) / 2s or -y / 2; ours samples 1 - fract(v * rep + off): uv.y = -v - off / rep
    v_scale, v_add = (1.0 / (2.0 * sy), -0.5) if sy > 0 else (0.5, 0.0)
    v_add -= off / rv
    return u_scale, u_add, v_scale, v_add, rep_u, rep_v


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


def geom_rgba(m: mujoco.MjModel, g: int) -> np.ndarray:
    """Colour MuJoCo draws geom ``g`` with (``setMaterial`` in mjv_updateScene): the material's rgba, unless the geom's
    own rgba differs from the default (0.5, 0.5, 0.5, 1) or there is no material."""
    mat = int(m.geom_matid[g])
    rgba = np.asarray(m.geom_rgba[g], np.float32)
    if mat >= 0 and np.array_equal(rgba, np.array([0.5, 0.5, 0.5, 1.0], np.float32)):
        return np.asarray(m.mat_rgba[mat], np.float32).copy()
    return rgba.copy()


def _tris_to_indexed_normals(tri, uv, nrm):
    """(F,3,3) triangles with per-corner uv (F,3,2) and normals (F,3,3) -> indexed vertices; corners merge only
    when position, uv and normal all agree (hard edges stay split)."""
    flat = tri.reshape(-1, 3)
    uv_flat = np.zeros((len(flat), 2)) if uv is None else uv.reshape(-1, 2)
    n_flat = nrm.reshape(-1, 3)
    key = np.round(np.concatenate([flat, uv_flat, n_flat], axis=1), 6)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    idx = inverse.reshape(-1, 3).astype(np.uint32)
    normals = n_flat[first]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return flat[first], normals, uv_flat[first], idx


@dataclass
class TextureAtlas:
    """Every model texture (2D, cube, skybox) packed into one RGBA8 image.

    Convention (MuJoCo's, kept for any texture backend): rows are stored as in ``tex_data``, so texture coordinate
    t = 0 is the first row of ``tex_data`` (the top of the image file; MuJoCo's OBJ loader stores 1 - v), exactly as
    MuJoCo uploads it to OpenGL. Cube and skybox textures keep their face strip (+X, -X, +Y, -Y, +Z, -Z stacked
    vertically, or one square face used for all six). Each texture is padded by one texel that wraps around (the 2D
    textures repeat), so bilinear filtering at a tile edge reads the opposite edge rather than black."""
    image: np.ndarray                      # (H, W, 4) uint8
    rects: dict = field(default_factory=dict)  # texid -> (u0, v0, du, dv)
    info: dict = field(default_factory=dict)   # texid -> (kind 0 2D / 1 cube / 2 skybox, faces 1 or 6, srgb 0 / 1)

    @staticmethod
    def texture_info(m: mujoco.MjModel, t: int):
        kind = {int(mujoco.mjtTexture.mjTEXTURE_2D): 0, int(mujoco.mjtTexture.mjTEXTURE_CUBE): 1,
                int(mujoco.mjtTexture.mjTEXTURE_SKYBOX): 2}.get(int(m.tex_type[t]), 0)
        w, h = int(m.tex_width[t]), int(m.tex_height[t])
        faces = 6 if (kind > 0 and h == 6 * w) else 1
        srgb = 1 if (hasattr(m, "tex_colorspace") and int(m.tex_colorspace[t]) == int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)) else 0
        return kind, faces, srgb

    @classmethod
    def build(cls, m: mujoco.MjModel):
        if m.ntex == 0:
            return cls(np.full((4, 4, 4), 255, np.uint8), {}, {})
        pad = 1
        usable = [t for t in range(m.ntex) if int(m.tex_nchannel[t]) in (1, 3, 4)]
        if not usable:
            return cls(np.full((4, 4, 4), 255, np.uint8), {}, {})
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
        rects, info = {}, {}
        for t in usable:
            w, h = widths[t], heights[t]
            nc = int(m.tex_nchannel[t])
            adr = int(m.tex_adr[t])
            data = m.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)
            rgba = np.empty((h, w, 4), np.uint8)
            rgba[..., :3] = data[..., :3] if nc >= 3 else data[..., :1]
            rgba[..., 3] = data[..., 3] if nc == 4 else 255
            px, py = pos[t]
            # one texel of wrap-around padding on every side (np.pad 'wrap')
            img[py - pad:py + h + pad, px - pad:px + w + pad] = np.pad(rgba, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")
            rects[t] = (px / aw, py / ah, w / aw, h / ah)
            info[t] = cls.texture_info(m, t)
        return cls(img, rects, info)

    @classmethod
    def build_legacy(cls, m: mujoco.MjModel):
        """The atlas before 2026-10 (appearance="legacy"): rows flipped, black padding, no skybox."""
        if m.ntex == 0:
            return cls(np.full((4, 4, 4), 255, np.uint8), {}, {})
        pad = 1
        usable = [t for t in range(m.ntex) if int(m.tex_type[t]) != int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)]
        if not usable:
            return cls(np.full((4, 4, 4), 255, np.uint8), {}, {})
        widths = [int(m.tex_width[t]) for t in range(m.ntex)]
        heights = [int(m.tex_height[t]) for t in range(m.ntex)]
        aw = max(widths[t] for t in usable) + 2 * pad
        order = sorted(usable, key=lambda t: -heights[t])
        while True:   # shelf packing; widened only when the column would exceed Metal's 16384-px texture limit
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
            if ah <= 16384 or aw * 2 > 16384:
                break
            aw *= 2
        img = np.zeros((ah, aw, 4), np.uint8)
        img[..., 3] = 255
        rects = {}
        for t in usable:
            w, h = widths[t], heights[t]
            nc = int(m.tex_nchannel[t])
            adr = int(m.tex_adr[t])
            data = m.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)
            px, py = pos[t]
            img[py:py + h, px:px + w, :3] = data[::-1, :, :3]
            if nc == 4:
                img[py:py + h, px:px + w, 3] = data[::-1, :, 3]
            rects[t] = (px / aw, py / ah, w / aw, h / ah)
        return cls(img, rects, {t: cls.texture_info(m, t) for t in rects})

    def skybox(self, m: mujoco.MjModel):
        """(texid, rect, faces) of the first skybox texture (the one MuJoCo draws), or None."""
        for t in range(m.ntex):
            if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX) and t in self.rects:
                return t, self.rects[t], self.info[t][1]
        return None


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
        if sensorsize[1] > 0:                          # MuJoCo's test (mjv_cameraFrustum)
            focal = m.cam_intrinsic[cam][:2]           # focal length (m)
            principal = m.cam_intrinsic[cam][2:4]      # principal point offset (m)
            fx = focal[0] / sensorsize[0] * width
            fy = focal[1] / sensorsize[1] * height
            # MuJoCo's frustum (engine_vis_visualize.c getFrustum, render_gl3.c glFrustum): left / right extents
            # (s/2 -+ p) / f, top / bottom (s/2 -+ p) / f, so the optical axis sits at W/2 - p_x W/s_x and, rows down,
            # H/2 - p_y H/s_y (it was + before 2026-10-12: offset principal points were mirrored)
            cx = width / 2 - principal[0] / sensorsize[0] * width
            cy = height / 2 - principal[1] / sensorsize[1] * height
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


def scene_params(m: mujoco.MjModel, atlas: TextureAtlas | None = None) -> np.ndarray:
    """(7, 4) float32: bounds center+radius | headlight ambient | headlight diffuse | headlight specular + active |
    sky colour + present | skybox atlas rect (u0, v0, du, dv) | skybox (faces, drawn, 0, 0).

    Rows 0-4 are the original layout. Rows 5-6 (with ``atlas``) let the renderers draw MuJoCo's skybox as its cube
    map (the first skybox texture, as ``mjr_render``): direction d maps to cube coordinates (d.x, d.z, -d.y)."""
    hl = m.vis.headlight
    P = np.zeros((7, 4), np.float32)
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
    sky = atlas.skybox(m) if atlas is not None else None
    if sky is not None:
        P[5] = sky[1]; P[6, 0] = float(sky[2]); P[6, 1] = 1.0
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
    # slots drawn as a unit primitive scaled by the per-world geom_size at render time (see build_scene_tables)
    sized: np.ndarray = None    # (G,) bool
    # per-slot texture mapping beyond the material block (see MATERIAL_EXT_LAYOUT); None = all zero
    material_ext: np.ndarray = None   # (G, 8) float32
    # transparency: slots MuJoCo draws blended (alpha < 0.995) and the draws of the remaining (opaque) instances
    transparent: np.ndarray = None    # (G,) bool
    draws_opaque: list = None         # as ``draws``, each mesh's opaque instances only (they come first in its range)
    # infinite planes drawn as a disc centred under the camera (build_scene_tables(plane_ring=...)): per slot
    # flag, size x, size y, 0 | u_scale, u_add, v_scale, v_add (uv = local xy * scale + add; plane_uv_affine)
    plane: np.ndarray = None    # (G, 8) float32 (extra slots: 0)
    # per-world sizes: SIZE_* per slot (0 = mesh baked at the compiled size) and the compiled scale of the slot's mesh
    size_mode: np.ndarray = None    # (G,) int32
    slot_scale: np.ndarray = None   # (G, 3) float32

    # material extension block (float4 x 2), read with the material by every renderer:
    # 0: uv offset xy (added after the texrepeat scale), texture kind (0 2D, 1 cube), sRGB texture (1: decode to linear)
    # 1: cube object scale xyz (geom-frame position -> MuJoCo's cube-map object coordinates), cube faces (1 or 6)
    MATERIAL_EXT_LAYOUT = ("uvoffset_kind_srgb", "cubescale_faces")
    # non-geom objects MuJoCo's renderer draws (metalsim.render.extras): slots G0 = len(geoms) .. G - 1, sites first.
    # With extras, G counts every slot; ``geoms`` / ``geom_mesh`` / ``draws`` / ``inst_table`` keep their meaning for the
    # geom slots, sites add instanced slots (pose rows ngeom + k), dynamic objects (flex, skin, tendon, user meshes) are
    # drawn from GPU-generated vertices (metalsim.render.dynamic_mesh).
    extras: list = field(default_factory=list)      # per extra slot: kind, objtype, objid, name, seg_id (geom seg mode)
    vis: dict = field(default_factory=dict)         # flex visual flags of the mjvOption used
    n_geom_slots: int = 0
    site_ids: list = field(default_factory=list)    # model site id per site slot
    n_pose: int = 0                                 # pose rows per env (ngeom + drawn sites); 0 = ngeom
    slot_pose: np.ndarray = None                    # (instanced slots,) pose row per slot; None = geoms
    slot_mesh_inst: np.ndarray = None               # (instanced slots,) mesh per slot; None = geom_mesh
    dyn: list = field(default_factory=list)         # dynamic objects (kind, id, slot, ...)

    def sized_geom_ids(self) -> list:
        return [g for g, f in zip(self.geoms, self.sized) if f] if self.sized is not None else []

    def pose_rows(self, m) -> int:
        """Pose rows per env the renderer reads (geom_xpos / geom_xmat, then the drawn sites')."""
        return self.n_pose or m.ngeom

    def instanced_slot_pose(self) -> np.ndarray:
        """(instanced slots,) int32 pose row per instanced slot (geoms, then sites)."""
        return np.asarray(self.geoms if self.slot_pose is None else self.slot_pose, np.int32)

    def slot_seg_ids(self) -> np.ndarray:
        """(G,) the id segmentation writes per slot in geom mode: geom + 1, extras ngeom + 1 + j."""
        return self.semantic[:, 0].astype(np.int64) + 1

    def geom_size_modes(self, ngeom: int) -> np.ndarray:
        """(ngeom,) SIZE_* per model geom (0 for geoms not drawn or baked)."""
        out = np.zeros(max(ngeom, 1), np.int32)
        if self.size_mode is not None:
            k = len(self.geoms)
            out[np.asarray(self.geoms, np.int64)] = self.size_mode[:k]
        return out

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


# per-world size modes (SceneTables.size_mode; metalsim.render.sized folds the scale into a copy of geom_xmat)
SIZE_BAKED, SIZE_DIAG, SIZE_SPHERE, SIZE_CYLINDER, SIZE_CAPSULE, SIZE_PLANE = 0, 1, 2, 3, 4, 5


def size_scale(mode: int, size, size0) -> np.ndarray:
    """Scale of the slot's mesh for MuJoCo ``geom_size`` ``size`` (``size0``: the compiled size the capsule mesh was
    built at). Unit meshes: box / ellipsoid diag(size), sphere r, cylinder (r, r, h), plane (sx, sy, 1). A capsule
    keeps its compiled mesh scaled by (r / r0, r / r0, (h + r) / (h0 + r0)): radius and total length exact, the caps
    squashed when the radius-to-length ratio changes (exact for a uniform scale)."""
    s = np.asarray(size, np.float64)
    if mode == SIZE_DIAG:
        return s[:3].astype(np.float32)
    if mode == SIZE_SPHERE:
        return np.full(3, s[0], np.float32)
    if mode == SIZE_CYLINDER:
        return np.array([s[0], s[0], s[1]], np.float32)
    if mode == SIZE_PLANE:
        return np.array([s[0], s[1], 1.0], np.float32)
    if mode == SIZE_CAPSULE:
        r0, h0 = float(size0[0]), float(size0[1])
        return np.array([s[0] / r0, s[0] / r0, (s[1] + s[0]) / (h0 + r0)], np.float32)
    return np.ones(3, np.float32)


SIZED_GEOM_PREFIX = "tslot"      # metalsim.learn.terrain: BoxWindow's per-world terrain box slots ("boxes_local")


def per_world_size_geoms(m: mujoco.MjModel) -> list:
    """Geoms whose size is per world at run time and must be drawn from ``geom_size`` rather than the compiled
    model: the rough terrain's box slots (``tslot<i>``, BoxWindow writes their per-world size and position)."""
    out = []
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith(SIZED_GEOM_PREFIX) and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            out.append(g)
    return out


def hfield_mesh(m: mujoco.MjModel, hid: int):
    """Indexed top surface of heightfield ``hid`` in its geom frame, MuJoCo's geometry: sample (r, c) at
    x = -sx + 2 sx c / (ncol - 1), y = -sy + 2 sy r / (nrow - 1), z = data * sz, each cell split along
    (c, r)-(c+1, r+1) as MuJoCo's collider and renderer do. Smooth vertex normals from the grid's central
    differences (shading only), UVs over the extent. Returns (verts, normals, uvs, idx)."""
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    sx, sy, sz, _ = (float(v) for v in m.hfield_size[hid])
    adr = int(m.hfield_adr[hid])
    Z = m.hfield_data[adr:adr + nrow * ncol].astype(np.float64).reshape(nrow, ncol) * sz
    xs = np.linspace(-sx, sx, ncol); ys = np.linspace(-sy, sy, nrow)
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    r, c = np.mgrid[0:nrow - 1, 0:ncol - 1]
    i00 = (r * ncol + c).ravel(); i01 = i00 + 1; i10 = i00 + ncol; i11 = i10 + 1
    idx = np.concatenate([np.stack([i00, i01, i11], 1), np.stack([i00, i11, i10], 1)]).astype(np.uint32)
    dzdx = np.gradient(Z, xs, axis=1) if ncol > 1 else np.zeros_like(Z)
    dzdy = np.gradient(Z, ys, axis=0) if nrow > 1 else np.zeros_like(Z)
    normals = np.stack([-dzdx.ravel(), -dzdy.ravel(), np.ones(Z.size)], 1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    uvs = np.stack([(X.ravel() + sx) / max(2 * sx, 1e-9), (Y.ravel() + sy) / max(2 * sy, 1e-9)], 1)
    return verts, normals, uvs, idx


def hfield_sides(m: mujoco.MjModel, hid: int, v_base: int):
    """MuJoCo's heightfield sides and base (the solid MuJoCo draws and collides with: the top surface over a box down
    to z = -size[3]): one vertical quad per boundary cell edge from the top profile down to the base, and the base
    rectangle facing -z. Flat outward normals; UVs over the extent as the top. Vertices are new (indices offset by
    ``v_base``), appended after the top grid so the grid keeps its row-major layout. Returns (verts, normals, uvs, idx)."""
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    sx, sy, sz, sb = (float(v) for v in m.hfield_size[hid])
    adr = int(m.hfield_adr[hid])
    Z = m.hfield_data[adr:adr + nrow * ncol].astype(np.float64).reshape(nrow, ncol) * sz
    xs = np.linspace(-sx, sx, ncol); ys = np.linspace(-sy, sy, nrow)
    edges = [  # (x, y, z) along the boundary, outward normal
        (xs, np.full(ncol, -sy), Z[0, :], (0.0, -1.0, 0.0)),
        (np.full(nrow, sx), ys, Z[:, -1], (1.0, 0.0, 0.0)),
        (xs, np.full(ncol, sy), Z[-1, :], (0.0, 1.0, 0.0)),
        (np.full(nrow, -sx), ys, Z[:, 0], (-1.0, 0.0, 0.0)),
    ]
    V, N, I = [], [], []
    n = 0
    for x, y, z, nrm in edges:
        k = len(x)
        top = np.stack([x, y, z], 1); bot = np.stack([x, y, np.full(k, -sb)], 1)
        V += [top, bot]; N.append(np.tile(nrm, (2 * k, 1)))
        i = np.arange(k - 1)
        t0, t1, b0, b1 = n + i, n + i + 1, n + k + i, n + k + i + 1
        I += [np.stack([t0, b0, b1], 1), np.stack([t0, b1, t1], 1)]
        n += 2 * k
    base = np.array([[-sx, -sy, -sb], [sx, -sy, -sb], [sx, sy, -sb], [-sx, sy, -sb]])
    V.append(base); N.append(np.tile((0.0, 0.0, -1.0), (4, 1)))
    I.append(np.array([[n, n + 2, n + 1], [n, n + 3, n + 2]]))
    verts = np.concatenate(V); normals = np.concatenate(N)
    uvs = np.stack([(verts[:, 0] + sx) / max(2 * sx, 1e-9), (verts[:, 1] + sy) / max(2 * sy, 1e-9)], 1)
    return verts, normals, uvs, (np.concatenate(I) + v_base).astype(np.uint32)


def build_scene_tables(m: mujoco.MjModel, n_envs: int, *, max_group: int = 3, include_planes: bool = True,
                       plane_extent: float = 5.0, smooth: bool = True, decimate_faces: int = 0,
                       sized_geoms="auto", draw_hfields: bool = True, appearance: str = "mujoco",
                       visual=False, deformables=(), plane_ring: float | None = None,
                       hfield_skirts: bool = True, per_world_size: bool = False) -> SceneTables:
    """``decimate_faces``: per-mesh triangle budget for a level of detail (0 = undecimated).

    ``sized_geoms``: box geoms drawn as a unit box scaled at render time by the per-world ``geom_size`` (the renderer
    reads MuJoCo Warp's ``Model.geom_size``); "auto" = ``per_world_size_geoms(m)`` (the rough terrain's box slots),
    () = the previous behaviour (every box baked at its compiled size; the slots then drew as 2 mm cubes). Their
    material is the terrain heightfield's (the first drawn hfield geom), when the model has one.
    ``draw_hfields``: heightfield geoms as their exact top surface (False: skipped, the previous behaviour).
    ``appearance``: "mujoco" (default) draws what MuJoCo's renderer draws (``metalsim.render.builtin_geoms``): its
    primitive tessellations, normals and texture coordinates, mesh normals with hard edges kept, SDF geoms from their
    mesh, cube textures, the skybox, and geom rgba resolved against the material as ``mjv_updateScene`` does;
    "legacy" = the tables before 2026-10 (smooth-averaged mesh normals, own UVs, image rows flipped), for comparison.

    ``visual``: the non-geom objects MuJoCo's renderer draws (sites, flexes, skins, spatial tendons; see
    ``metalsim.render.extras``) under this ``mujoco.MjvOption`` (None: MuJoCo's default option), False: geoms only.
    ``deformables``: user ``extras.DeformableMesh`` objects (XPBD / PhysX-port cloth, any per-env vertex array).
    ``hfield_skirts``: also MuJoCo's sides and base of each heightfield (False: top surface only, the previous behaviour).
    ``plane_ring``: infinite planes (a size <= 0) as a disc of octagon rings of this apothem (m) with MuJoCo's plane
    texture coordinates (``plane_uv_affine``), recorded in ``plane`` for the raster to centre under the camera; None =
    the previous behaviour (a 2 ``plane_extent`` square).
    ``per_world_size``: every primitive (box, sphere, ellipsoid, cylinder, capsule, finite plane) drawn from a unit
    mesh scaled at render time by the per-world ``geom_size`` (``size_mode``; see ``metalsim.render.sized``); False =
    only the terrain slots."""
    if appearance not in ("mujoco", "legacy"):
        raise ValueError(f"appearance must be 'mujoco' or 'legacy', got {appearance!r}")
    if isinstance(sized_geoms, str):
        sized_geoms = per_world_size_geoms(m) if sized_geoms == "auto" else ()
    sized_set = {int(g) for g in sized_geoms}
    G_ = mujoco.mjtGeom
    unit_types = {int(G_.mjGEOM_BOX): SIZE_DIAG, int(G_.mjGEOM_ELLIPSOID): SIZE_DIAG, int(G_.mjGEOM_SPHERE): SIZE_SPHERE,
                  int(G_.mjGEOM_CYLINDER): SIZE_CYLINDER, int(G_.mjGEOM_PLANE): SIZE_PLANE}

    def is_inf_plane(g):
        return int(m.geom_type[g]) == int(G_.mjGEOM_PLANE) and (m.geom_size[g][0] <= 0 or m.geom_size[g][1] <= 0)

    slot_like = set(per_world_size_geoms(m)) - sized_set     # terrain slots left out (terrain_slots=False): baked

    def size_mode_of(g):
        if g in sized_set:
            return SIZE_DIAG
        if not per_world_size or is_inf_plane(g) or g in slot_like:
            return SIZE_BAKED
        t = int(m.geom_type[g])
        if t == int(G_.mjGEOM_CAPSULE):
            return SIZE_CAPSULE
        return unit_types.get(t, SIZE_BAKED)
    plane_rows = {}
    mesh_cache = {}
    meshes, geoms, geom_mesh, modes = [], [], [], []
    all_v, all_n, all_uv, all_i = [], [], [], []
    v_off = i_off = 0
    atlas = TextureAtlas.build(m) if appearance == "mujoco" else TextureAtlas.build_legacy(m)

    def geom_tris_mujoco(g, unit=False):
        """(positions, uv, normals) per corner, as MuJoCo's renderer draws geom g (metalsim.render.builtin_geoms)."""
        t = int(m.geom_type[g])
        size = np.ones(3) if unit else m.geom_size[g]
        if t in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):
            mid = int(m.geom_dataid[g])
            if mid < 0:
                return None, None, None
            P, U, N = bg.mesh_triangles(m, mid)
            if decimate_faces and len(P) > decimate_faces:
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                v_d, f_d = _decimate(m.mesh_vert[va:va + vn].astype(np.float64), m.mesh_face[fa:fa + fn].astype(np.int64),
                                     decimate_faces)
                P, U, N = v_d[f_d], None, bg.mujoco_vertex_normals(v_d, f_d)
            if U is None:
                U = bg.texgen_uv(P)
            return P, U, N
        if t == mujoco.mjtGeom.mjGEOM_BOX and unit:
            return bg.primitive(m, t, np.ones(3), plane_extent)
        out = bg.primitive(m, t, size, plane_extent)
        return (None, None, None) if out is None else out

    def geom_tris(g, unit=False):
        if appearance == "mujoco":
            return geom_tris_mujoco(g, unit)
        tri, uv = geom_tris_legacy(g, unit)
        return tri, uv, None

    def geom_tris_legacy(g, unit=False):
        t = int(m.geom_type[g])
        size = np.ones(3) if unit else m.geom_size[g]
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
        if appearance == "mujoco":
            if geom_rgba(m, g)[3] == 0.0:       # mjv_updateScene skips geoms whose (material-resolved) alpha is 0
                continue
        elif m.geom_rgba[g][3] == 0.0 and int(m.geom_matid[g]) < 0:
            continue
        if t == mujoco.mjtGeom.mjGEOM_HFIELD:
            if not draw_hfields:
                continue
            key = ("hfield", int(m.geom_dataid[g]))
            if key not in mesh_cache:
                hid = int(m.geom_dataid[g])
                verts, normals, uvs, idx = hfield_mesh(m, hid)
                idx = idx.reshape(-1) + v_off
                top_count = int(idx.size)
                if hfield_skirts:
                    sv, sn, suv, sidx = hfield_sides(m, hid, v_off + len(verts))
                    verts = np.concatenate([verts, sv]); normals = np.concatenate([normals, sn]); uvs = np.concatenate([uvs, suv])
                    idx = np.concatenate([idx, sidx.reshape(-1)])
                if appearance == "mujoco":
                    uvs = bg.texgen_uv(verts)         # MuJoCo's object-linear mapping (scale / offset per material)
                mesh_cache[key] = len(meshes)
                # row-major (nrow, ncol) top grid from v_off, then the sides / base (indices from i_off + top_count)
                meshes.append({"i_off": i_off, "i_count": int(idx.size), "radius": float(np.linalg.norm(verts, axis=1).max()),
                               "v_off": v_off, "hfield": hid, "top_count": top_count})
                all_v.append(verts); all_n.append(normals); all_uv.append(uvs)
                all_i.append(idx)
                v_off += len(verts); i_off += idx.size
            geom_mesh.append(mesh_cache[key]); geoms.append(g); modes.append(SIZE_BAKED)
            continue
        is_mesh = t == mujoco.mjtGeom.mjGEOM_MESH or (appearance == "mujoco" and t == mujoco.mjtGeom.mjGEOM_SDF)
        mode = size_mode_of(g)
        unit = mode in (SIZE_DIAG, SIZE_SPHERE, SIZE_CYLINDER, SIZE_PLANE)
        ring = plane_ring is not None and is_inf_plane(g)
        if ring:
            mat = int(m.geom_matid[g])
            rep = m.mat_texrepeat[mat] if mat >= 0 else (1.0, 1.0)
            uni = bool(m.mat_texuniform[mat]) if mat >= 0 else False
            tex = plane_uv_affine(m.geom_size[g], appearance, rep, uni)
            plane_rows[len(geoms)] = (float(m.geom_size[g][0]), float(m.geom_size[g][1])) + tex
            key = ("plane_ring",) + tuple(np.round(plane_rows[len(geoms)], 9))
        elif unit:
            key = ("unit", t)
        else:
            key = ("mesh", int(m.geom_dataid[g])) if is_mesh else (t, tuple(np.round(m.geom_size[g], 6)))
        if key not in mesh_cache:
            if ring:
                sx, sy, us, ua, vs, va = plane_rows[len(geoms)][:6]
                tri = _octagon_rings(plane_ring)
                if sx > 0:                              # a finite axis of a mixed plane: clamp (degenerate triangles)
                    tri[..., 0] = np.clip(tri[..., 0], -sx, sx)
                if sy > 0:
                    tri[..., 1] = np.clip(tri[..., 1], -sy, sy)
                uv = np.stack([tri[..., 0] * us + ua, tri[..., 1] * vs + va], -1)
                nrm = None
            else:
                tri, uv, nrm = geom_tris(g, unit=unit)
            if tri is None:
                continue
            if nrm is not None:
                verts, normals, uvs, idx = _tris_to_indexed_normals(tri, uv, nrm)
            else:
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
        elif not ring and geom_tris(g)[0] is None:
            continue
        geom_mesh.append(mesh_cache[key])
        geoms.append(g); modes.append(mode)
    if not geoms:
        from metalsim.render import extras as _extras
        if not _extras.collect(m, n_envs, atlas, visual, deformables, smooth)["extras"]:
            raise ValueError("no drawable geoms (check max_group / include_planes)")
        all_v, all_n, all_uv, all_i = [np.zeros((0, 3))], [np.zeros((0, 3))], [np.zeros((0, 2))], [np.zeros(0, np.uint32)]

    G = len(geoms)
    verts = np.concatenate(all_v).astype(np.float32)
    inter = np.zeros((len(verts), 8), np.float32)
    inter[:, 0:3] = verts
    inter[:, 3:6] = np.concatenate(all_n).astype(np.float32)
    inter[:, 6:8] = np.concatenate(all_uv).astype(np.float32)
    indices = np.concatenate(all_i).astype(np.uint32)

    geom_mesh = np.asarray(geom_mesh)
    # transparent slots (MuJoCo: rgba alpha < 0.995 after the material, mjr_render's test); decided from the geoms'
    # colours here, the instance table below lists each mesh's opaque slots first
    alpha = np.array([(geom_rgba(m, g) if appearance == "mujoco" else
                       (m.mat_rgba[int(m.geom_matid[g])] if int(m.geom_matid[g]) >= 0 else m.geom_rgba[g]))[3] for g in geoms])
    transparent = (alpha < 0.995) & np.array([g not in sized_set and int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_HFIELD)
                                              for g in geoms], bool)   # heightfields stay opaque (tile path)
    draws, draws_opaque, table = [], [], []
    first = 0
    for mi in range(len(meshes)):
        slots = np.flatnonzero(geom_mesh == mi)
        slots = np.concatenate([slots[~transparent[slots]], slots[transparent[slots]]])
        pairs = [(e, int(s)) for s in slots for e in range(n_envs)]
        table.extend(pairs)
        draws.append((meshes[mi]["i_off"], meshes[mi]["i_count"], first, len(pairs)))
        n_op = int((~transparent[slots]).sum()) * n_envs       # may be 0: entries stay aligned with ``draws``
        draws_opaque.append((meshes[mi]["i_off"], meshes[mi]["i_count"], first, n_op))
        first += len(pairs)
    inst_table = np.asarray(table, np.uint32).reshape(-1, 2)

    mats = np.zeros((G, 16), np.float32)
    semantic = np.zeros((G, 4), np.int32)
    static = np.zeros(G, bool)
    ext = np.zeros((G, 8), np.float32)
    ext[:, 4:7] = 1.0; ext[:, 7] = 1.0
    for slot, g in enumerate(geoms):
        mat = int(m.geom_matid[g])
        rgba = (geom_rgba(m, g) if appearance == "mujoco" else
                m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g])
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
            if texid >= 0 and texid in atlas.rects and appearance == "mujoco":
                kind, faces, srgb = atlas.info[texid]
                mats[slot, 8:12] = atlas.rects[texid]
                mats[slot, 14] = 1.0
                if kind == 0:
                    scl, off = bg.material_uv_transform(m, g, mat)
                    mats[slot, 12:14] = scl
                    ext[slot, 0:2] = off
                else:                                   # cube (a skybox texture on a geom maps as a cube too)
                    mats[slot, 12:14] = 1.0
                    ext[slot, 2] = 1.0
                    ext[slot, 4:7] = bg.cube_scale(m, g, mat)
                    ext[slot, 7] = float(faces)
                ext[slot, 3] = float(srgb) if kind == 0 else 0.0     # MuJoCo uploads cube maps as plain RGB
            elif texid >= 0 and texid in atlas.rects:
                mats[slot, 8:12] = atlas.rects[texid]
                mats[slot, 12:14] = m.mat_texrepeat[mat]
                if m.mat_texuniform[mat]:
                    # repeat is per world unit: scale by the extent the geom's UVs span
                    gt = int(m.geom_type[g])
                    if gt == mujoco.mjtGeom.mjGEOM_PLANE:
                        ext = np.array([m.geom_size[g][0] if m.geom_size[g][0] > 0 else plane_extent,
                                        m.geom_size[g][1] if m.geom_size[g][1] > 0 else plane_extent]) * 2
                    elif gt == mujoco.mjtGeom.mjGEOM_HFIELD:      # UVs span the heightfield's extent
                        ext = 2 * m.hfield_size[int(m.geom_dataid[g])][:2]
                    else:
                        ext = np.sort(m.geom_size[g][:3])[-2:] * 2
                    mats[slot, 12:14] = m.mat_texrepeat[mat] * ext
                mats[slot, 14] = 1.0
        else:
            mats[slot, 4:8] = (0.0, 0.6, 0.5, 0.0)
        if slot in plane_rows and mats[slot, 14] > 0.5 and appearance != "mujoco":   # legacy: plane_texture_map's repeat
            mats[slot, 12:14] = plane_rows[slot][6:8]
        body = int(m.geom_bodyid[g])
        semantic[slot] = (g, body, int(m.body_rootid[body]), int(m.geom_group[g]))
        static[slot] = body == 0
    sized = np.array([g in sized_set for g in geoms], bool)
    terrain = [s_ for s_, g in enumerate(geoms) if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_HFIELD)]
    if sized.any() and terrain:
        tm = mats[terrain[0]].copy()            # the slot boxes are pieces of the terrain: its material, untextured
        if tm[14] > 0.5:                        # (a unit box's UVs would stretch one texture tile per face): the
            u0, v0, du, dv = tm[8:12]           # texture's mean colour folded into the base colour
            H_, W_ = atlas.image.shape[:2]
            px = atlas.image[int(v0 * H_):max(int((v0 + dv) * H_), int(v0 * H_) + 1),
                             int(u0 * W_):max(int((u0 + du) * W_), int(u0 * W_) + 1), :3]
            tm[0:3] *= px.reshape(-1, 3).mean(0) / 255.0
        tm[12:15] = (1.0, 1.0, 0.0)
        mats[sized] = tm
    if sized.any():
        ext[sized] = (0, 0, 0, 0, 1, 1, 1, 1)
    params = scene_params(m, atlas if appearance == "mujoco" else None)
    if sized.any():
        # MuJoCo's stat.center / extent include the slots parked 1 km under the terrain (1115 m instead of ~117 m for
        # Isaac's rough terrain), which scaled tier 0's per-env shadow map to 11.6 m texels. Renderer bounds: the
        # drawn geoms except the slots, at qpos0.
        d = mujoco.MjData(m); mujoco.mj_kinematics(m, d)
        keep = [g for g in geoms if g not in sized_set]
        lo = np.min([d.geom_xpos[g] - m.geom_rbound[g] for g in keep], axis=0)
        hi = np.max([d.geom_xpos[g] + m.geom_rbound[g] for g in keep], axis=0)
        params[0, :3] = 0.5 * (lo + hi); params[0, 3] = max(float(0.5 * (hi - lo).max()), 0.1)
    plane = np.zeros((G, 8), np.float32)
    for slot, row in plane_rows.items():
        plane[slot] = (1.0, row[0], row[1], 0.0) + tuple(row[2:6])
    size_mode = np.asarray(modes, np.int32)
    slot_scale = np.ones((G, 3), np.float32)       # compiled scale of each slot's mesh (unit meshes; see size_scale)
    for slot, g in enumerate(geoms):
        slot_scale[slot] = size_scale(int(size_mode[slot]), m.geom_size[g], m.geom_size[g])
    tables = SceneTables(vertices=inter, indices=indices, meshes=meshes, geoms=geoms, geom_mesh=geom_mesh,
                         draws=draws, inst_table=inst_table, materials=mats, atlas=atlas, semantic=semantic,
                         static_geom=static, lights=build_lights(m), params=params, n_envs=n_envs, G=G,
                         sized=sized, material_ext=ext, transparent=transparent, draws_opaque=draws_opaque, n_geom_slots=G,
                         plane=plane, size_mode=size_mode, slot_scale=slot_scale)
    if visual is not False or deformables:
        from metalsim.render import extras as _extras
        _extras.extend_tables(tables, m, n_envs, visual, deformables, smooth)
        Gx = tables.G                              # extra slots: no plane disc, baked, unit scale
        if Gx > G:
            tables.plane = np.concatenate([plane, np.zeros((Gx - G, 8), np.float32)])
            tables.size_mode = np.concatenate([size_mode, np.zeros(Gx - G, np.int32)])
            tables.slot_scale = np.concatenate([slot_scale, np.ones((Gx - G, 3), np.float32)])
    return tables


def quats_to_mats(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    M = np.empty((len(q), 3, 3))
    M[:, 0, 0] = 1 - 2 * (y * y + z * z); M[:, 0, 1] = 2 * (x * y - w * z); M[:, 0, 2] = 2 * (x * z + w * y)
    M[:, 1, 0] = 2 * (x * y + w * z); M[:, 1, 1] = 1 - 2 * (x * x + z * z); M[:, 1, 2] = 2 * (y * z - w * x)
    M[:, 2, 0] = 2 * (x * z - w * y); M[:, 2, 1] = 2 * (y * z + w * x); M[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return M
