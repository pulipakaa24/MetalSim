"""MuJoCo-exact drawable geometry: builtin primitives, mesh normals and texture coordinates.

The reference is MuJoCo's own renderer (``mujoco.Renderer``; src/render/classic, MuJoCo 3.14.0):

* builtin primitives (``render_context.c``: ``sphere``, ``halfSphere``, ``cylinder``, ``disk``, the box list,
  ``makePlane``) with their vertex normals and their explicit texture coordinates, tessellated with the model's
  ``vis.quality`` (numslices / numstacks). Flat pieces (box faces, cylinder side, disks) use the minimum triangles:
  MuJoCo subdivides them only for its per-vertex (Gouraud) lighting, and the coordinates are linear on them.
* mesh normals (``mjr_uploadMesh``): each corner takes the compiled ``mesh_normal`` through ``mesh_facenormal``,
  and on a mesh without texture coordinates a corner whose normal is more than ~37 degrees (dot < 0.8) away from its
  face normal takes the face normal instead. That keeps hard edges of CAD / STL meshes sharp.
* texture coordinates follow MuJoCo's convention: ``t = 0`` is the first row of ``tex_data`` (the top of the image
  file; MuJoCo's OBJ loader stores ``1 - v``). ``texrepeat`` / ``texuniform`` scaling and the offsets of MuJoCo's
  object-linear mapping are applied per material by the renderer (``material_uv_transform``).

All functions return per-corner arrays: positions (F, 3, 3) in the geom frame, texture coordinates (F, 3, 2) and
normals (F, 3, 3).
"""
from __future__ import annotations

import mujoco
import numpy as np

DOT_SHARP = 0.8          # mjr_uploadMesh: corner normal replaced by the face normal below this dot product


def _quad(out, a, b, c, d):
    """Append quad (a, b, c, d) as two triangles; each vertex is (pos, normal, uv)."""
    out.append((a, b, c)); out.append((a, c, d))


def _pack(tris):
    P = np.array([[v[0] for v in t] for t in tris], np.float64)
    N = np.array([[v[1] for v in t] for t in tris], np.float64)
    U = np.array([[v[2] for v in t] for t in tris], np.float64)
    return P, U, N


def _sv(az, el, sign):
    """setVertexSphere: unit sphere vertex (centre at z = sign) and normal."""
    v = (np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), sign + np.sin(el))
    return np.array(v), np.array((v[0], v[1], v[2] - sign))


def unit_sphere(nslice: int, nstack: int):
    """MuJoCo's ``sphere(nSlice, nStack)``: u = azimuth / 2 pi, v = 0.5 - elevation / pi (north pole v = 0)."""
    tris = []
    h = nstack // 2
    for sign in (-1, 1):
        el1 = 0.5 * np.pi * sign * (h - 1) / h
        v1uv = 0.5 - sign * (h - 1) / nstack
        v3uv = 0.0 if sign > 0 else 1.0
        for j in range(nslice):
            az1, az2 = 2 * np.pi * j / nslice, 2 * np.pi * (j + 1) / nslice
            p1, n1 = _sv(az1, el1, 0); p2, n2 = _sv(az2, el1, 0)
            p3 = np.array((0.0, 0.0, sign)); n3 = p3.copy()
            u1, u2, u3 = az1 / (2 * np.pi), az2 / (2 * np.pi), 0.5 * (az1 + az2) / (2 * np.pi)
            a, b, c = (p1, n1, (u1, v1uv)), (p2, n2, (u2, v1uv)), (p3, n3, (u3, v3uv))
            tris.append((a, b, c) if sign > 0 else (c, b, a))
        for i in range(h - 1):
            el1, el2 = 0.5 * np.pi * sign * i / h, 0.5 * np.pi * sign * (i + 1) / h
            ve1, ve2 = 0.5 - sign * i / nstack, 0.5 - sign * (i + 1) / nstack
            for j in range(nslice):
                az1, az2 = 2 * np.pi * j / nslice, 2 * np.pi * (j + 1) / nslice
                u1, u2 = az1 / (2 * np.pi), az2 / (2 * np.pi)
                q = [(*_sv(az1, el1, 0), (u1, ve1)), (*_sv(az2, el1, 0), (u2, ve1)),
                     (*_sv(az2, el2, 0), (u2, ve2)), (*_sv(az1, el2, 0), (u1, ve2))]
                if sign > 0:
                    _quad(tris, q[0], q[1], q[2], q[3])
                else:
                    _quad(tris, q[3], q[2], q[1], q[0])
    return _pack(tris)


def unit_half_sphere(sign: int, nslice: int, nstack: int):
    """MuJoCo's ``halfSphere(sign, nSlice, nStack)`` (capsule caps): centre at z = sign, top v = 0 at the pole."""
    tris = []
    el1 = 0.5 * np.pi * sign * (nstack - 1) / nstack
    ve1 = (nstack - 1) / nstack; v3uv = 1.0
    if sign > 0:
        ve1, v3uv = 1.0 - ve1, 0.0
    for j in range(nslice):
        az1, az2 = 2 * np.pi * j / nslice, 2 * np.pi * (j + 1) / nslice
        p1, n1 = _sv(az1, el1, sign); p2, n2 = _sv(az2, el1, sign)
        p3 = np.array((0.0, 0.0, 2.0 * sign)); n3 = np.array((0.0, 0.0, float(sign)))
        u1, u2, u3 = az1 / (2 * np.pi), az2 / (2 * np.pi), 0.5 * (az1 + az2) / (2 * np.pi)
        a, b, c = (p1, n1, (u1, ve1)), (p2, n2, (u2, ve1)), (p3, n3, (u3, v3uv))
        tris.append((a, b, c) if sign > 0 else (c, b, a))
    for i in range(nstack - 1):
        el1, el2 = 0.5 * np.pi * sign * i / nstack, 0.5 * np.pi * sign * (i + 1) / nstack
        ve1, ve2 = i / nstack, (i + 1) / nstack
        if sign > 0:
            ve1, ve2 = 1.0 - ve1, 1.0 - ve2
        for j in range(nslice):
            az1, az2 = 2 * np.pi * j / nslice, 2 * np.pi * (j + 1) / nslice
            u1, u2 = az1 / (2 * np.pi), az2 / (2 * np.pi)
            q = [(*_sv(az1, el1, sign), (u1, ve1)), (*_sv(az2, el1, sign), (u2, ve1)),
                 (*_sv(az2, el2, sign), (u2, ve2)), (*_sv(az1, el2, sign), (u1, ve2))]
            if sign > 0:
                _quad(tris, q[0], q[1], q[2], q[3])
            else:
                _quad(tris, q[3], q[2], q[1], q[0])
    return _pack(tris)


def unit_cylinder_side(nslice: int):
    """MuJoCo's open ``cylinder``: radius 1, z in [-1, 1], radial normals, u = azimuth / 2 pi, v = (1 - z) / 2."""
    tris = []
    for j in range(nslice):
        az1, az2 = 2 * np.pi * j / nslice, 2 * np.pi * (j + 1) / nslice
        u1, u2 = j / nslice, (j + 1) / nslice
        c1, s1, c2, s2 = np.cos(az1), np.sin(az1), np.cos(az2), np.sin(az2)
        _quad(tris, (np.array((c1, s1, -1.0)), np.array((c1, s1, 0.0)), (u1, 1.0)),
              (np.array((c2, s2, -1.0)), np.array((c2, s2, 0.0)), (u2, 1.0)),
              (np.array((c2, s2, 1.0)), np.array((c2, s2, 0.0)), (u2, 0.0)),
              (np.array((c1, s1, 1.0)), np.array((c1, s1, 0.0)), (u1, 0.0)))
    return _pack(tris)


def unit_disk(sign: int, nslice: int):
    """MuJoCo's ``disk(sign)`` at z = sign: u = 0.5 + 0.5 x, v = 0.5 + 0.5 y, normal (0, 0, sign)."""
    tris = []
    n = np.array((0.0, 0.0, float(sign)))
    c = (np.array((0.0, 0.0, float(sign))), n, (0.5, 0.5))
    for j in range(nslice):
        az1, az2 = 2 * np.pi * j / nslice, 2 * np.pi * (j + 1) / nslice
        p1 = np.array((np.cos(az1), np.sin(az1), float(sign))); p2 = np.array((np.cos(az2), np.sin(az2), float(sign)))
        a, b = (p1, n, (0.5 + 0.5 * p1[0], 0.5 + 0.5 * p1[1])), (p2, n, (0.5 + 0.5 * p2[0], 0.5 + 0.5 * p2[1]))
        tris.append((a, b, c) if sign > 0 else (c, b, a))
    return _pack(tris)


def unit_box():
    """MuJoCo's box list: per face u from the first in-plane axis, v = (1 - second) / 2; faces +-X use (y, z),
    +-Y use (x, z), +-Z use (x, y)."""
    tris = []
    for axis in range(3):
        a1, a2 = [k for k in range(3) if k != axis]
        for s in (1.0, -1.0):
            n = np.zeros(3); n[axis] = s

            def v(p, q):
                x = np.zeros(3); x[axis] = s; x[a1] = p; x[a2] = q
                return (x, n, (0.5 * (p + 1.0), 0.5 * (1.0 - q)))
            # counter-clockwise seen from outside
            if (s > 0) == ((a1, a2) in ((1, 2), (0, 1))):
                _quad(tris, v(-1, -1), v(1, -1), v(1, 1), v(-1, 1))
            else:
                _quad(tris, v(-1, -1), v(-1, 1), v(1, 1), v(1, -1))
    return _pack(tris)


def _scale(P, U, N, size):
    """Scale unit geometry by ``size``; normals by the inverse scale (OpenGL normal matrix)."""
    size = np.asarray(size, np.float64)
    N2 = N / size
    N2 /= np.maximum(np.linalg.norm(N2, axis=-1, keepdims=True), 1e-12)
    return P * size, U, N2


def primitive(m: mujoco.MjModel, gtype: int, size, plane_extent: float = 5.0):
    """Drawable triangles of a builtin geom as MuJoCo's renderer draws it. Returns (P, U, N) or None.

    ``size`` is the model's ``geom_size`` row. Planes with size 0 (infinite) are drawn ``plane_extent`` wide."""
    q = m.vis.quality
    ns, nk = int(q.numslices), int(q.numstacks)
    s = np.asarray(size, np.float64)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _scale(*unit_sphere(ns, nk), (s[0], s[0], s[0]))
    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return _scale(*unit_sphere(ns, nk), s[:3])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return _scale(*unit_box(), s[:3])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        parts = [unit_disk(-1, ns), unit_cylinder_side(ns), unit_disk(1, ns)]
        P, U, N = (np.concatenate([p[k] for p in parts]) for k in range(3))
        return _scale(P, U, N, (s[0], s[0], s[1]))
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, hl = s[0], s[1]
        P, U, N = _scale(*unit_cylinder_side(ns), (r, r, hl))
        parts = [(P, U, N)]
        for sign in (1, -1):
            Ph, Uh, Nh = unit_half_sphere(sign, ns, nk // 2)
            Ph = Ph * r
            Ph[..., 2] += sign * (hl - r)          # centre at z = sign * half-length (renderGeom's capsule transform)
            parts.append((Ph, Uh, Nh))
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))
    if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        sx = float(s[0]) if s[0] > 0 else plane_extent
        sy = float(s[1]) if s[1] > 0 else plane_extent
        P = np.array([[[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0]], [[-sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]]], np.float64)
        N = np.zeros_like(P); N[..., 2] = 1.0
        # makePlane: finite axis u = (x + sx) / 2sx, v = 1 - (y + sy) / 2sy; infinite axis u = x / 2, v = -y / 2
        u = (P[..., 0] + sx) / (2 * sx) if s[0] > 0 else 0.5 * P[..., 0]
        v = 1.0 - (P[..., 1] + sy) / (2 * sy) if s[1] > 0 else -0.5 * P[..., 1]
        return P, np.stack([u, v], -1), N
    return None


def face_normals(P):
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    return fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)


def mesh_triangles(m: mujoco.MjModel, mid: int):
    """Mesh ``mid`` as MuJoCo's renderer draws it (``mjr_uploadMesh``). Returns (P, U or None, N)."""
    va = int(m.mesh_vertadr[mid]); fa, fnum = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    faces = m.mesh_face[fa:fa + fnum].astype(np.int64)
    P = m.mesh_vert[va + faces].astype(np.float64)
    na = int(m.mesh_normaladr[mid])
    N = m.mesh_normal[na + m.mesh_facenormal[fa:fa + fnum].astype(np.int64)].astype(np.float64)
    tca = int(m.mesh_texcoordadr[mid])
    U = None
    if tca >= 0:
        U = m.mesh_texcoord[tca + m.mesh_facetexcoord[fa:fa + fnum].astype(np.int64)].astype(np.float64)
    else:
        fnrm = face_normals(P)
        sharp = np.einsum("fkc,fc->fk", N, fnrm) < DOT_SHARP
        N = np.where(sharp[..., None], fnrm[:, None, :], N)
    return P, U, N


def mujoco_vertex_normals(v, f, smooth=False):
    """Per-corner normals of an indexed mesh by MuJoCo's rule (``mjCMesh::MakeNormal`` without ``smoothnormal``,
    then ``mjr_uploadMesh``'s face-normal fallback): area-weighted vertex normals with the faces more than ~37 deg
    off removed, and corners still more than ~37 deg off their face take the face normal. For meshes the renderer
    rebuilds (level-of-detail decimation). Returns (F, 3, 3)."""
    P = v[f]
    cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(cr, axis=1, keepdims=True)
    fn = cr / np.maximum(area, 1e-300)
    vn = np.zeros_like(v, dtype=np.float64)
    for k in range(3):
        np.add.at(vn, f[:, k], fn * area)
    if not smooth:
        unit = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        rem = np.zeros_like(vn)
        for k in range(3):
            off = np.einsum("fc,fc->f", fn, unit[f[:, k]]) < DOT_SHARP
            np.add.at(rem, f[off, k], fn[off] * area[off])
        vn = vn - rem
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(ln > 1e-15, vn / np.maximum(ln, 1e-300), np.array([0.0, 0.0, 1.0]))
    N = vn[f]
    sharp = np.einsum("fkc,fc->fk", N, fn) < DOT_SHARP
    return np.where(sharp[..., None], fn[:, None, :], N)


def texgen_uv(P):
    """MuJoCo's object-linear 2D mapping for geoms without explicit coordinates (meshes without texcoord,
    heightfields): s = 0.5 x scl0 - 0.5, t = -0.5 y scl1 - 0.5. The stored coordinates are (0.5 x, -0.5 y);
    the scale and the -0.5 offset are applied per material (``material_uv_transform``)."""
    return np.stack([0.5 * P[..., 0], -0.5 * P[..., 1]], -1)


def mjv_size(gtype: int, size):
    """``mjv_initGeom``'s size: sphere (r, r, r), capsule / cylinder (r, r, half-length), others as given."""
    s = np.asarray(size, np.float64)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.array([s[0], s[0], s[0]])
    if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.array([s[0], s[0], s[1]])
    return s[:3].copy()


def explicit_uv(m: mujoco.MjModel, g: int) -> bool:
    """True when MuJoCo maps a 2D texture on geom ``g`` through explicit coordinates (builtin primitives, meshes
    with texcoord); False for the object-linear mapping (meshes without texcoord, heightfields)."""
    t = int(m.geom_type[g])
    if t in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):
        mid = int(m.geom_dataid[g])
        return mid >= 0 and int(m.mesh_texcoordadr[mid]) >= 0
    return t != mujoco.mjtGeom.mjGEOM_HFIELD


def material_uv_transform(m: mujoco.MjModel, g: int, mat: int):
    """(scale xy, offset xy) taking the stored coordinates to MuJoCo's texture coordinates (``settexture``)."""
    t = int(m.geom_type[g])
    size = mjv_size(t, m.geom_size[g])
    rep = np.asarray(m.mat_texrepeat[mat], np.float64)
    uni = bool(m.mat_texuniform[mat])
    off = np.zeros(2)
    if explicit_uv(m, g):
        scl = np.where(rep > 0, rep, 1.0)
        if uni:
            scl = scl * np.where(size[:2] > 0, size[:2], 1.0)
        if t == mujoco.mjtGeom.mjGEOM_PLANE and (size[0] <= 0 or size[1] <= 0):
            off[:] = -0.5
    else:
        scl = rep.copy()
        if int(m.geom_dataid[g]) >= 0:
            scl = np.where(size[:2] > 0, scl / np.maximum(size[:2], mujoco.mjMINVAL), scl)
        if uni:
            scl = np.where(size[:2] > 0, scl * size[:2], scl)
        off[:] = -0.5
    return scl, off


def cube_scale(m: mujoco.MjModel, g: int, mat: int):
    """Per-axis factor taking geom-frame positions to the object coordinates MuJoCo's cube mapping reads
    (``settexture``, cube branch): the unit shape of a scaled builtin (texuniform: times the geom size), mesh and
    heightfield coordinates as they are."""
    t = int(m.geom_type[g])
    size = mjv_size(t, m.geom_size[g])
    uni = bool(m.mat_texuniform[mat])
    if t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_BOX,
             mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        unit = 1.0 / np.maximum(size, 1e-12)
        return unit * size if uni else unit
    if uni:
        return np.where(size > 0, size, 1.0)
    return np.ones(3)
