"""Dynamic meshes: GPU-generated per-env vertices of the dynamic slots of the scene tables (``metalsim.render.extras``).

Flexes, skins, spatial tendons and user deformable meshes change shape every frame. ``DynamicMeshes`` turns the
simulation state into vertex records (position, normal, uv; the layout of the static vertices) for every env in a
few compute dispatches (``shaders/dynamic_mesh.metal``), as MuJoCo's renderer builds them on the CPU:

* 2D flex (cloth): the smoothed surface offset by +radius and -radius along the vertex normals (each vertex normal
  the normalized sum of its triangles' unit normals, MuJoCo's ``addNormal``), the boundary closed by side faces
  (``makeSide``), or with ``flatskin`` per-face normals; 3D flex (soft volume): its shell offset by +radius;
  ``mjVIS_FLEXFACE`` without the skin flag: flat element faces offset along their normals (``makeFace``);
* 1D flex (cable), or a flex with the skin flag off: one open cylinder per edge (``mjVIS_FLEXEDGE``), spheres at the
  vertices with ``mjVIS_FLEXVERT``, MuJoCo's builtin tessellation (``vis.quality.numslices`` / ``numstacks``);
* skin: linear blend skinning by the bones (``xpos`` / ``xquat``), area-weighted vertex normals, ``inflate``;
* spatial tendon: per segment between consecutive wrap points, a capsule of the tendon width (half inside a wrapping
  geom, none across a pulley); the slots for the model's maximum number of points are always generated and unused ones
  are degenerate. (MuJoCo draws a catenary instead for a two-site slack spring tendon under gravity; not reproduced.)
* user ``DeformableMesh``: the surface (radius 0) or a 2D-flex-style skin (radius > 0) of an (N, V, 3) array.

Records are written at ``rbase`` (in records) of a vertex buffer: per object k, env e, local vertex l at
``obj_base[k] + e * nvert[k] + l``. With a ray tracer the positions include the env's grid offset (the acceleration
structure separates envs spatially) and the raster subtracts it again.

Sources per frame: ``flexvert_xpos`` (N, nflexvert, 3), ``xpos`` / ``xquat`` (skins), ``ten_wrapadr``,
``ten_wrapnum``, ``wrap_obj``, ``wrap_xpos`` (tendons) as (Metal buffer, byte offset) pairs from MuJoCo Warp's ``Data``
(``sources_from_sim``) or host buffers (``write_host``), and one (N, V, 3) array per user mesh, by name.
"""
from __future__ import annotations

import mujoco
import numpy as np
import Metal

S_SMOOTH, S_FLATSKIN, S_SIDE, S_FACE = 0, 1, 2, 3
T_EDGE, T_VERT, T_TENDON, T_CATENARY = 0, 1, 2, 3

_DYNC = 16      # DynConsts words
_MJMINVAL = 1e-15


def catenary_mode(m, i: int):
    """MuJoCo's ``mjv_isCatenary`` static conditions for tendon ``i``: (-1, 0) never a catenary; (0, L): always (a
    deadband spring [0, L]); (1, L): while its length is below L (a length limit [0, L] without stiffness)."""
    if m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY) or np.linalg.norm(m.opt.gravity) <= _MJMINVAL:
        return -1, 0.0
    if int(m.tendon_num[i]) != 2:
        return -1, 0.0
    W_ = mujoco.mjtWrap
    adr = int(m.tendon_adr[i])
    if any(int(m.wrap_type[adr + k]) != W_.mjWRAP_SITE for k in range(2)):
        return -1, 0.0
    poly = lambda name: np.asarray(getattr(m, name, np.zeros((m.ntendon, 1))))[i]     # noqa: E731
    has_stiff = float(m.tendon_stiffness[i]) != 0 or bool(np.any(poly("tendon_stiffnesspoly") != 0))
    ls = np.asarray(m.tendon_lengthspring[i], np.float64)
    limitedspring = has_stiff and ls[0] == 0 and ls[1] > 0
    rng = np.asarray(m.tendon_range[i], np.float64)
    limitedconstraint = (not has_stiff) and int(m.tendon_limited[i]) == 1 and rng[0] == 0
    damping = float(m.tendon_damping[i]) != 0 or bool(np.any(poly("tendon_dampingpoly") != 0))
    if not (limitedspring or limitedconstraint) or damping or float(m.tendon_frictionloss[i]) != 0:
        return -1, 0.0
    for j in range(m.nu):
        if int(m.actuator_trntype[j]) == int(mujoco.mjtTrn.mjTRN_TENDON) and int(m.actuator_trnid[j][0]) == i:
            return -1, 0.0
    return (0, float(ls[1])) if limitedspring else (1, float(rng[1]))


def catenary_points(x0, x1, gravity, length, npts):
    """NumPy (float64) port of MuJoCo's ``mjv_catenary``: points from x0 to x1 (2: straight, 3: hanging bead, npts)."""
    x0, x1 = np.asarray(x0, np.float64), np.asarray(x1, np.float64)
    if np.linalg.norm(x1 - x0) > length:
        return np.stack([x0, x1])
    up = -np.asarray(gravity, np.float64)
    up = up / np.linalg.norm(up)
    x01 = x1 - x0
    across = x01 - up * np.dot(up, x01)
    nrm = np.linalg.norm(across)
    across = np.zeros(3) if nrm < _MJMINVAL else across / nrm
    h, v = float(np.dot(x01, across)), float(np.dot(x01, up))
    if length > 100 * h:
        d_up = -0.5 * (np.sqrt(length * length - h * h) - v)
        d_across = h * d_up / (2 * d_up - v)
        return np.stack([x0, x0 + up * d_up + across * d_across, x1])
    intercept = 1 / np.sqrt(np.sqrt(length * length - v * v) / h - 1)

    def resid(b, grad=False):
        a = 0.5 / b
        r = 1 / np.sqrt(2 * b * np.sinh(a) - 1) - intercept
        if not grad:
            return r
        return r, (a * np.cosh(a) - np.sinh(a)) * (2 * b * np.sinh(a) - 1) ** -1.5
    b = intercept / np.sqrt(24)
    for _ in range(50):
        res, g = resid(b, True)
        if abs(res) < 1e-9:
            break
        step = -res / g
        for _ in range(10):
            if abs(resid(b + step)) < abs(res):
                break
            step *= 0.5
        b += step
    bh = b * h
    h_off = -0.5 * (np.log((length + v) / (length - v)) * bh - h)
    v_off = -np.cosh(h_off / bh) * bh
    pts = [x0]
    for i in range(1, npts - 1):
        hz = i * h / npts
        pts.append(x0 + across * hz + up * (bh * np.cosh((hz - h_off) / bh) + v_off))
    pts.append(x1)
    return np.stack(pts)


def impedance(solimp, pos, margin):
    """MuJoCo's ``getimpedance`` (engine_core_constraint.c): the constraint impedance at distance ``pos``."""
    s = np.asarray(solimp, np.float64)
    if s[0] == s[1] or s[2] <= _MJMINVAL:
        return 0.5 * (s[0] + s[1])
    x = abs((pos - margin) / s[2])
    if x >= 1 or x <= 0:
        return s[1] if x >= 1 else s[0]
    if s[4] == 1:
        y = x
    elif x <= s[3]:
        y = (1 / s[3] ** (s[4] - 1)) * x ** s[4]
    else:
        y = 1 - (1 / (1 - s[3]) ** (s[4] - 1)) * (1 - x) ** s[4]
    return s[0] + y * (s[1] - s[0])


def tendon_limit_imp(m, i: int, ten_length: float) -> float:
    """Impedance MuJoCo mixes a default-grey tendon's colour with (its active limit's; the upper one's when both)."""
    if not int(m.tendon_limited[i]):
        return 0.0
    imp = 0.0
    for dist in (ten_length - m.tendon_range[i][0], m.tendon_range[i][1] - ten_length):
        if dist < m.tendon_margin[i]:
            imp = impedance(m.tendon_solimp_lim[i], dist, m.tendon_margin[i])
    return float(imp)


def _capsule_template(nslice: int, nstack: int):
    """MuJoCo's capsule (``mjrCYLINDEROPEN`` + ``halfSphere`` x 2, ``nstack`` rings per hemisphere): (V, 4) unit
    direction + end (-1 / +1) and (F, 3) triangles. Rings run from the bottom pole to the top pole."""
    az = 2 * np.pi * np.arange(nslice) / nslice
    rings = []                                   # (elevation, end)
    for i in range(nstack - 1, -1, -1):
        rings.append((-np.pi / 2 * i / nstack, -1.0))
    for i in range(0, nstack):
        rings.append((np.pi / 2 * i / nstack, 1.0))
    V = [(0.0, 0.0, -1.0, -1.0)]
    for el, end in rings:
        for a in az:
            V.append((np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el), end))
    V.append((0.0, 0.0, 1.0, 1.0))
    top = len(V) - 1
    F = []
    for j in range(nslice):
        j1 = (j + 1) % nslice
        F.append((0, 1 + j1, 1 + j))                     # bottom pole fan
        rr = len(rings) - 1
        F.append((1 + rr * nslice + j, 1 + rr * nslice + j1, top))
    for r in range(len(rings) - 1):
        for j in range(nslice):
            j1 = (j + 1) % nslice
            a, b = 1 + r * nslice + j, 1 + r * nslice + j1
            c, d = a + nslice, b + nslice
            F += [(a, b, d), (a, d, c)]
    return np.asarray(V, np.float32), np.asarray(F, np.int64)


def _cylinder_template(nslice: int):
    """MuJoCo's open cylinder (``mjrCYLINDEROPEN``): two rings at z = -1, +1."""
    az = 2 * np.pi * np.arange(nslice) / nslice
    V = [(np.cos(a), np.sin(a), 0.0, end) for end in (-1.0, 1.0) for a in az]
    F = []
    for j in range(nslice):
        j1 = (j + 1) % nslice
        F += [(j, j1, nslice + j1), (j, nslice + j1, nslice + j)]
    return np.asarray(V, np.float32), np.asarray(F, np.int64)


def _boundary_edges(tris):
    """Oriented boundary edges (as they run in their triangle) of a triangle list."""
    cnt = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(a, b), max(a, b))
            cnt[k] = cnt.get(k, 0) + 1
    out = []
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if cnt[(min(a, b), max(a, b))] == 1:
                out.append((int(a), int(b)))
    return out


class _Plan:
    """Host tables of the generation (built once)."""

    def __init__(self, m, tables, tube_slices=None, cap_stacks=None, tube_lod=None, tendons="mujoco"):
        """``tube_slices`` / ``cap_stacks``: one tessellation for every tube; else per tube object ``tube_lod``
        ({slot: (slices, stacks)}, ``tube_tessellation``); else MuJoCo's ``vis.quality``."""
        self.m = m
        vis = tables.vis or {}
        self.full = (int(m.vis.quality.numslices), max(int(m.vis.quality.numstacks) // 2, 1))
        nslice = int(tube_slices or self.full[0])
        nstack_half = int(cap_stacks or self.full[1])
        self.fixed = (nslice, nstack_half) if (tube_slices or cap_stacks) else None
        self.tube_lod = dict(tube_lod or {})
        self.cyl_t = _cylinder_template(nslice)
        self.cap_t = _capsule_template(nslice, nstack_half)
        self._templates = {}
        self.blocks = []          # (kind, key, p_off, n): position sources
        self.ptot = 0
        self.nent, self.flist, self.faces = [], [], []
        self.srec, self.trec = [], []
        self.skin_v, self.skin_b, self.bones = [], [], []
        self.objects = []         # per dynamic slot: dict(slot, nvert, tris (F,3) local, obj_base)
        self.rtot = 0
        self.tmpl = []            # template vertices (float4): cylinder then capsule
        self.tmpl_off = {}
        # tendons="mujoco": slack two-site tendons as MuJoCo's catenary, default-grey tendons tinted by their limit
        # impedance; "straight": capsules between the wrap points only (before 2026-10-13)
        if tendons not in ("mujoco", "straight"):
            raise ValueError(f"tendons must be 'mujoco' or 'straight', got {tendons!r}")
        self.tendons = tendons
        self.cat = []             # (tendon, mode, length): catenary-capable tendons (catenary_mode)
        self.tint = []            # (tendon, slot, mode): default-grey tendons recoloured at their limits
        self.ncat_pts = min(int(m.vis.quality.numslices) + 1, 100)
        flex_dyn = [d for d in tables.dyn if d["kind"] == "flex"]
        if flex_dyn:
            self.blocks.append(("flex", None, 0, int(m.nflexvert)))
            self.ptot = int(m.nflexvert)
        for d in tables.dyn:
            if d["kind"] == "mesh":
                self.blocks.append(("mesh", d["mesh"].name, self.ptot, int(d["mesh"].nvert)))
                d["p_off"] = self.ptot
                self.ptot += int(d["mesh"].nvert)
            elif d["kind"] == "skin":
                d["p_off"] = self.ptot
                self.ptot += int(m.skin_vertnum[d["id"]])
        self._normals_done = set()
        for d in tables.dyn:
            getattr(self, "_" + d["kind"])(d, vis)
        self.nwrap2 = 2 * int(m.nwrap)

    # -- helpers ---------------------------------------------------------------------------------------------------

    def _template(self, kind, slot):
        """(V, F, offset in the template table) of the tube template of this object's tessellation."""
        ns, nk = self.fixed or self.tube_lod.get(slot, self.full)
        key = (kind, int(ns), int(nk))
        if key not in self._templates:
            V, F = _cylinder_template(int(ns)) if kind == "cyl" else _capsule_template(int(ns), int(nk))
            self._templates[key] = (V, F, len(self.tmpl))
            self.tmpl.extend(V.tolist())
        return self._templates[key]

    def _object(self, slot):
        ob = {"slot": slot, "nvert": 0, "tris": []}
        self.objects.append(ob)
        return ob

    def _vert_s(self, ob, mode, a, b=0, c=0, d=0, pr=0.0, ns=1.0, uv=(0.0, 0.0)):
        l = ob["nvert"]; ob["nvert"] += 1
        self.srec.append([mode, a, b, c, d, l, 0, len(self.objects) - 1, pr, ns, float(uv[0]), float(uv[1])])
        return l

    def _normals(self, key, pverts, faces_p, mode):
        """Vertex-normal entries for P vertices ``pverts`` from triangles ``faces_p`` (P indices)."""
        if key in self._normals_done:
            return
        self._normals_done.add(key)
        f0 = len(self.faces)
        adj = {int(p): [] for p in pverts}
        for k, f in enumerate(faces_p):
            self.faces.append((int(f[0]), int(f[1]), int(f[2]), 0))
            for v in f:
                if int(v) in adj:
                    adj[int(v)].append(f0 + k)
        for p in pverts:
            lst = adj[int(p)]
            self.nent.append((int(p), len(self.flist), len(lst), mode))
            self.flist.extend(lst)

    def _skin_surface(self, ob, P0, nv, tris, radius, flat, shells, uv=None):
        """MuJoCo's 2D flex skin (``makeSmooth`` +-radius, ``makeSide`` on the shell edges) on P vertices P0 + i."""
        uvv = (lambda i: uv[i]) if uv is not None else (lambda i: (0.0, 0.0))
        if not flat:
            top = [self._vert_s(ob, S_SMOOTH, P0 + i, pr=radius, ns=1.0, uv=uvv(i)) for i in range(nv)]
            for t in tris:
                ob["tris"].append((top[t[0]], top[t[1]], top[t[2]]))
            if radius != 0:
                bot = [self._vert_s(ob, S_SMOOTH, P0 + i, pr=-radius, ns=-1.0, uv=uvv(i)) for i in range(nv)]
                for t in tris:
                    ob["tris"].append((bot[t[0]], bot[t[2]], bot[t[1]]))
        else:
            for t in tris:
                q = [P0 + int(x) for x in t]
                ob["tris"].append(tuple(self._vert_s(ob, S_FLATSKIN, q[k], q[0], q[1], q[2], radius, 1.0, uvv(t[k])) for k in range(3)))
            if radius != 0:
                for t in tris:
                    q = [P0 + int(x) for x in (t[0], t[2], t[1])]
                    ob["tris"].append(tuple(self._vert_s(ob, S_FLATSKIN, q[k], q[0], q[1], q[2], -radius, -1.0, uvv((t[0], t[2], t[1])[k])) for k in range(3)))
        if radius != 0:
            for (i0, i1) in shells:
                a, b = P0 + int(i0), P0 + int(i1)
                # makeSide(i0, i1, +r): (i0 +r, i1 -r, i1 +r), normal cross(x1 - x0, n1)
                ob["tris"].append(tuple(self._vert_s(ob, S_SIDE, p, a, b, 0, s * radius, 1.0, uvv(ii))
                                        for p, s, ii in ((a, 1, i0), (b, -1, i1), (b, 1, i1))))
                # makeSide(i1, i0, -r): (i1 -r, i0 +r, i0 -r), normal -cross(x0 - x1, n0)
                ob["tris"].append(tuple(self._vert_s(ob, S_SIDE, p, b, a, 0, s * radius, -1.0, uvv(ii))
                                        for p, s, ii in ((b, -1, i1), (a, 1, i0), (a, -1, i0))))

    def _tube(self, ob, kind, mode, a, b, r):
        V, F, t0 = self._template(kind, ob["slot"])
        base = ob["nvert"]
        for tv in range(len(V)):
            self.trec.append([mode, a, b, t0 + tv, base + tv, 0, len(self.objects) - 1, 0, r])
        ob["nvert"] += len(V)
        ob["tris"].extend((F + base).tolist())

    # -- object kinds ------------------------------------------------------------------------------------------------

    def _flex(self, d, vis):
        m, f = self.m, d["id"]
        ob = self._object(d["slot"])
        dim, va, nv = int(m.flex_dim[f]), int(m.flex_vertadr[f]), int(m.flex_vertnum[f])
        r = float(m.flex_radius[f])
        elems = m.flex_elem[m.flex_elemdataadr[f]: m.flex_elemdataadr[f] + (dim + 1) * m.flex_elemnum[f]].reshape(-1, dim + 1)
        shells = m.flex_shell[m.flex_shelldataadr[f]: m.flex_shelldataadr[f] + dim * m.flex_shellnum[f]].reshape(-1, dim) if dim > 1 else np.zeros((0, dim), int)
        uv = None
        if int(m.flex_texcoordadr[f]) >= 0:
            uv = np.zeros((nv, 2), np.float32)
            tc = m.flex_texcoord[int(m.flex_texcoordadr[f]):]
            et = m.flex_elemtexcoord[m.flex_elemdataadr[f]: m.flex_elemdataadr[f] + (dim + 1) * m.flex_elemnum[f]].reshape(-1, dim + 1)
            for e in range(len(elems))[::-1]:
                for k in range(dim + 1):
                    uv[elems[e, k]] = tc[et[e, k]]
        if d["faces"]:
            if vis.get("flexskin"):
                flat = bool(m.flex_flatskin[f])
                tri = elems[:, :3] if dim == 2 else shells
                self._normals(("flex", f), range(va, va + nv), tri + va, 0)
                if dim == 2:
                    self._skin_surface(ob, va, nv, elems, r, flat, shells, uv)
                else:
                    self._skin_surface(ob, va, nv, shells, 0.0, flat, (), uv) if r == 0 else None
                    if r != 0:   # 3D shell: +radius only, no bottom layer or sides
                        if not flat:
                            top = [self._vert_s(ob, S_SMOOTH, va + i, pr=r, ns=1.0, uv=(uv[i] if uv is not None else (0, 0))) for i in range(nv)]
                            for t in shells:
                                ob["tris"].append((top[t[0]], top[t[1]], top[t[2]]))
                        else:
                            for t in shells:
                                q = [va + int(x) for x in t]
                                ob["tris"].append(tuple(self._vert_s(ob, S_FLATSKIN, q[k], q[0], q[1], q[2], r, 1.0) for k in range(3)))
            else:   # mjVIS_FLEXFACE: flat element faces, offset along their own normal (makeFace)
                layer = int(vis.get("flex_layer", 0))
                if dim == 2:
                    fl = [(e[0], e[1], e[2]) for e in elems] + [(e[0], e[2], e[1]) for e in elems]
                else:
                    lay = m.flex_elemlayer[m.flex_elemadr[f]: m.flex_elemadr[f] + m.flex_elemnum[f]]
                    fl = []
                    for e, L in zip(elems, lay):
                        if L == layer:
                            fl += [(e[0], e[1], e[2]), (e[0], e[2], e[3]), (e[0], e[3], e[1]), (e[1], e[3], e[2])]
                for t in fl:
                    q = [va + int(x) for x in t]
                    ob["tris"].append(tuple(self._vert_s(ob, S_FACE, q[k], q[0], q[1], q[2], r, 1.0) for k in range(3)))
        if d["lines"] and r > 0:
            if vis.get("flexedge"):
                edges = m.flex_edge[m.flex_edgeadr[f]: m.flex_edgeadr[f] + m.flex_edgenum[f]]
                for a, b in edges:
                    self._tube(ob, "cyl", T_EDGE, va + int(a), va + int(b), r)
            if vis.get("flexvert"):
                for i in range(nv):
                    self._tube(ob, "cap", T_VERT, va + i, va + i, r)

    def _skin(self, d, vis):
        m, i = self.m, d["id"]
        ob = self._object(d["slot"])
        P0, va, nv = d["p_off"], int(m.skin_vertadr[i]), int(m.skin_vertnum[i])
        faces = m.skin_face[m.skin_faceadr[i]: m.skin_faceadr[i] + m.skin_facenum[i]].astype(np.int64)
        per_vert = [[] for _ in range(nv)]
        b0 = len(self.bones)
        for k, j in enumerate(range(m.skin_boneadr[i], m.skin_boneadr[i] + m.skin_bonenum[i])):
            self.bones.append(list(m.skin_bonebindpos[j]) + [float(m.skin_bonebodyid[j])] + list(m.skin_bonebindquat[j]))
            for q in range(m.skin_bonevertadr[j], m.skin_bonevertadr[j] + m.skin_bonevertnum[j]):
                per_vert[int(m.skin_bonevertid[q])].append((b0 + k, float(m.skin_bonevertweight[q])))
        for v in range(nv):
            self.skin_v.append((P0 + v, va + v, len(self.skin_b), len(per_vert[v])))
            self.skin_b.extend(per_vert[v])
        self._normals(("skin", i), range(P0, P0 + nv), faces + P0, 1)
        uv = None
        if int(m.skin_texcoordadr[i]) >= 0:
            uv = m.skin_texcoord[int(m.skin_texcoordadr[i]): int(m.skin_texcoordadr[i]) + nv]
        inflate = float(m.skin_inflate[i])
        idx = [self._vert_s(ob, S_SMOOTH, P0 + v, pr=inflate, ns=1.0, uv=(uv[v] if uv is not None else (0, 0))) for v in range(nv)]
        for t in faces:
            ob["tris"].append((idx[t[0]], idx[t[1]], idx[t[2]]))

    def _tendon(self, d, vis):
        m, i = self.m, d["id"]
        ob = self._object(d["slot"])
        w = float(m.tendon_width[i])
        mode, L = catenary_mode(m, i) if self.tendons == "mujoco" else (-1, 0.0)
        if mode >= 0:          # up to ncat_pts - 1 segments of its catenary (or one straight one), from emit_tube mode 3
            k = len(self.cat)
            self.cat.append((i, mode, L))
            for j in range(self.ncat_pts - 1):
                self._tube(ob, "cap", T_CATENARY, k, j, w)
        else:
            for j in range(max(d["max_points"] - 1, 0)):
                self._tube(ob, "cap", T_TENDON, i, j, w)
        # MuJoCo's limit colour: no material and exactly the default grey (not when always drawn as a catenary)
        if (self.tendons == "mujoco" and int(m.tendon_matid[i]) == -1 and mode != 0 and int(m.tendon_limited[i])
                and np.array_equal(np.asarray(m.tendon_rgba[i], np.float32), np.array([0.5, 0.5, 0.5, 1.0], np.float32))):
            self.tint.append((i, int(d["slot"]), mode))

    def _mesh(self, d, vis):
        mesh = d["mesh"]
        ob = self._object(d["slot"])
        P0, nv = d["p_off"], int(mesh.nvert)
        tris = mesh.tris
        self._normals(("mesh", mesh.name), range(P0, P0 + nv), tris + P0, 0)
        shells = _boundary_edges(tris) if mesh.radius > 0 else ()
        self._skin_surface(ob, P0, nv, tris, float(mesh.radius), False, shells, mesh.uv)

    # -- final layout ------------------------------------------------------------------------------------------------

    def finalize(self, n_envs):
        base = 0
        for ob in self.objects:
            ob["obj_base"] = base
            ob["tris"] = np.asarray(ob["tris"], np.int64).reshape(-1, 3)
            base += n_envs * ob["nvert"]
        self.rtot = base
        for r in self.srec:
            ob = self.objects[r[7]]
            r[5] = ob["obj_base"] + r[5]; r[6] = ob["nvert"]; r[7] = 0
        for r in self.trec:
            ob = self.objects[r[6]]
            r[4] = ob["obj_base"] + r[4]; r[5] = ob["nvert"]; r[6] = 0

    def arrays(self):
        srec = np.zeros((max(len(self.srec), 1), 12), np.int32)
        for k, r in enumerate(self.srec):
            srec[k, :8] = r[:8]
            srec[k, 8:12].view(np.float32)[:] = r[8:12]
        trec = np.zeros((max(len(self.trec), 1), 12), np.int32)
        for k, r in enumerate(self.trec):
            trec[k, :8] = r[:8]
            trec[k, 8:9].view(np.float32)[:] = r[8]
        nent = np.asarray(self.nent or [(0, 0, 0, 0)], np.int32).reshape(-1, 4)
        flist = np.asarray(self.flist or [0], np.int32)
        faces = np.asarray(self.faces or [(0, 0, 0, 0)], np.int32).reshape(-1, 4)
        skin_v = np.asarray(self.skin_v or [(0, 0, 0, 0)], np.int32).reshape(-1, 4)
        skin_b = np.zeros((max(len(self.skin_b), 1), 2), np.int32)
        for k, (b, w) in enumerate(self.skin_b):
            skin_b[k, 0] = b; skin_b[k, 1:2].view(np.float32)[:] = w
        bones = np.asarray(self.bones or [[0.0] * 8], np.float32).reshape(-1, 8)
        tmpl = np.asarray(self.tmpl, np.float32).reshape(-1, 4)
        cat = np.zeros((max(len(self.cat), 1), 4), np.int32)          # shaders/dynamic_mesh.metal CatRec
        for k, (i, mode, L) in enumerate(self.cat):
            cat[k, 0], cat[k, 1] = i, mode
            cat[k, 2:3].view(np.float32)[:] = L
        tint = np.zeros((max(len(self.tint), 1), 20), np.float32)     # TintRec
        ti = tint.view(np.int32)
        m = self.m
        for k, (i, slot, mode) in enumerate(self.tint):
            ti[k, 0:4] = (i, slot, mode, int(m.tendon_limited[i]))
            tint[k, 4:7] = (m.tendon_range[i][0], m.tendon_range[i][1], m.tendon_margin[i])
            tint[k, 8:13] = m.tendon_solimp_lim[i]
            tint[k, 16:20] = m.tendon_rgba[i]
        return dict(srec=srec, trec=trec, nent=nent, flist=flist, faces=faces, skin_v=skin_v, skin_b=skin_b, bones=bones, tmpl=tmpl,
                    cat=cat, tint=tint)

    def validate(self, n_envs):
        """Index bounds of every table (a kernel never reads or writes outside its buffers)."""
        P = self.ptot
        for r in self.srec:
            assert 0 <= r[1] < P and 0 <= r[2] < P and 0 <= r[3] < P and 0 <= r[4] < P, r
            assert 0 <= r[5] and r[5] + (n_envs - 1) * r[6] < self.rtot
        for r in self.trec:
            if r[0] == T_CATENARY:
                assert 0 <= r[1] < len(self.cat) and 0 <= r[2] < self.ncat_pts - 1, r
            elif r[0] != T_TENDON:
                assert 0 <= r[1] < P and 0 <= r[2] < P, r
            assert 0 <= r[3] < len(self.tmpl) and 0 <= r[4] + (n_envs - 1) * r[5] < self.rtot
        for (p, s, c, _) in self.nent:
            assert 0 <= p < P and 0 <= s and s + c <= len(self.flist)
        for f in self.faces:
            assert all(0 <= x < P for x in f[:3])
        assert all(0 <= x < len(self.faces) for x in self.flist)
        for (p, src, s, c) in self.skin_v:
            assert 0 <= p < P and 0 <= src < self.m.nskinvert and s + c <= len(self.skin_b)
        assert all(0 <= b < len(self.bones) for b, _ in self.skin_b)
        assert all(0 <= int(b[3]) < self.m.nbody for b in self.bones)
        for ob in self.objects:
            if len(ob["tris"]):
                assert ob["tris"].min() >= 0 and ob["tris"].max() < ob["nvert"]


def _quat_mat_np(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _unit_or_x(v):
    n = np.linalg.norm(v)
    return np.array([1.0, 0.0, 0.0]) if n < 1e-15 else v / n


def emulate(plan: _Plan, n_envs: int, src: dict) -> np.ndarray:
    """NumPy reference of the generation kernels (float64): ``src`` = {"flexvert_xpos": (N, nflexvert, 3), "xpos",
    "xquat", "ten_wrapadr", "ten_wrapnum", "wrap_obj" (N, 2 nwrap), "wrap_xpos" (N, 2 nwrap, 3), mesh name: (N, V, 3)}.
    Returns (n_records, 8) records without env offsets (record of object k, env e, vertex l at obj_base + e nvert + l)."""
    m = plan.m
    A = plan.arrays()
    out = np.zeros((max(plan.rtot, 1), 8))
    tmpl = A["tmpl"].astype(np.float64)
    for e in range(n_envs):
        P = np.zeros((max(plan.ptot, 1), 3))
        for kind, key, p_off, nv in plan.blocks:
            P[p_off:p_off + nv] = np.asarray(src["flexvert_xpos" if kind == "flex" else key], np.float64)[e][:nv]
        for (p, s_, start, cnt) in plan.skin_v:
            acc = np.zeros(3)
            v0 = m.skin_vert[s_].astype(np.float64)
            for b, w in plan.skin_b[start:start + cnt]:
                bp = plan.bones[b]
                body = int(bp[3])
                bq = np.asarray(bp[4:8], np.float64)
                q = np.asarray(src["xquat"][e][body], np.float64)
                qq = np.zeros(4)
                mujoco.mju_mulQuat(qq, q, np.array([bq[0], -bq[1], -bq[2], -bq[3]]))
                R = _quat_mat_np(qq)
                acc += w * (R @ v0 + np.asarray(src["xpos"][e][body], np.float64) - R @ np.asarray(bp[0:3], np.float64))
            P[p] = acc
        Nn = np.zeros_like(P)
        for (p, start, cnt, mode) in plan.nent:
            acc = np.zeros(3)
            for fi in plan.flist[start:start + cnt]:
                a, b, c, _ = plan.faces[fi]
                fn = np.cross(P[b] - P[a], P[c] - P[a])
                acc += _unit_or_x(fn) if mode == 0 else fn
            Nn[p] = _unit_or_x(acc) if mode == 0 else acc / max(np.linalg.norm(acc), 1e-15)
        for r in plan.srec:
            mode, a, b, c, d_, dst, vstride, _, pr, ns, u, v = r
            if mode == S_FACE:
                fn = _unit_or_x(np.cross(P[c] - P[b], P[d_] - P[b]))
                pos, nrm = P[a] + pr * fn, ns * fn
            else:
                pos = P[a] + pr * Nn[a]
                if mode == S_SMOOTH:
                    nrm = ns * Nn[a]
                elif mode == S_FLATSKIN:
                    nrm = ns * _unit_or_x(np.cross(P[c] - P[b], P[d_] - P[b]))
                else:
                    nrm = ns * _unit_or_x(np.cross(P[c] - P[b], Nn[c]))
            out[dst + e * vstride] = (*pos, *nrm, u, v)
        cps = []
        for (ti, cmode, L) in plan.cat:           # catenary points per catenary tendon (emit_tube mode 3)
            adr = int(src["ten_wrapadr"][e][ti])
            x0, x1 = np.asarray(src["wrap_xpos"][e][adr], np.float64), np.asarray(src["wrap_xpos"][e][adr + 1], np.float64)
            slack = cmode == 0 or float(src["ten_length"][e][ti]) < L
            cps.append(catenary_points(x0, x1, m.opt.gravity, L, plan.ncat_pts) if slack else np.stack([x0, x1]))
        for r in plan.trec:
            mode, a, b, tv, dst, vstride = r[:6]
            rad = r[8]
            t = tmpl[tv]
            if mode == T_CATENARY:
                pts = cps[a]
                if b + 1 >= len(pts):
                    out[dst + e * vstride] = (*pts[0], 0, 0, 1, 0, 0)
                    continue
                p0, p1 = pts[b], pts[b + 1]
            elif mode == T_TENDON:
                adr, num = int(src["ten_wrapadr"][e][a]), int(src["ten_wrapnum"][e][a])
                j = adr + b
                ok = b < num - 1 and j + 1 < plan.nwrap2
                if ok:
                    o0, o1 = int(src["wrap_obj"][e][j]), int(src["wrap_obj"][e][j + 1])
                    ok = o0 != -2 and o1 != -2
                    if o0 >= 0 and o1 >= 0:
                        rad *= 0.5
                    p0, p1 = np.asarray(src["wrap_xpos"][e][j], np.float64), np.asarray(src["wrap_xpos"][e][j + 1], np.float64)
                if not ok:
                    out[dst + e * vstride] = (*np.asarray(src["wrap_xpos"][e][max(adr, 0)], np.float64), 0, 0, 1, 0, 0)
                    continue
            else:
                p0 = P[a]
                p1 = P[b] if mode == T_EDGE else p0
            dvec = p1 - p0
            q = np.zeros(4)
            mujoco.mju_quatZ2Vec(q, dvec)
            R = _quat_mat_np(q)
            h = 0.5 * np.linalg.norm(dvec)
            pos = 0.5 * (p0 + p1) + R @ np.array([rad * t[0], rad * t[1], rad * t[2] + h * t[3]])
            out[dst + e * vstride] = (*pos, *(R @ t[:3]), 0, 0)
    return out[:plan.rtot]


def model_viewpoints(m, data=None):
    """(K, 3) positions sensors may look from at the initial state: every camera and site."""
    import mujoco
    d = data
    if d is None:
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    return np.concatenate([np.asarray(d.cam_xpos).reshape(-1, 3), np.asarray(d.site_xpos).reshape(-1, 3)])


def camera_view(m, cam, fy):
    """(viewpoints, resolution) of a model camera rendered with focal length ``fy`` pixels."""
    import mujoco
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return np.asarray(d.cam_xpos[cam]).reshape(1, 3), 1.0 / float(fy)


def beam_resolution(azimuths, elevations):
    """Smallest angular spacing between a lidar's distinct beam azimuths / elevations (rad)."""
    out = np.inf
    for a in (azimuths, elevations):
        u = np.unique(np.round(np.asarray(a, np.float64).ravel(), 9))
        if len(u) > 1:
            out = min(out, float(np.diff(u).min()))
    return out


def tube_tessellation(m, tables, viewpoints, resolution, *, full_px=3.0, subpixel_px=0.5, margin=2.0, data=None):
    """Tessellation of each tube object (tendons, flex edge cylinders, flex vertex spheres) for a camera or sensor
    (rendering only; nothing here touches physics): ``viewpoints`` (K, 3) world positions it looks from, ``resolution``
    its angular size of one pixel or beam (rad). A tube's projected width is 2 r / (d resolution) with d its closest
    distance to any viewpoint at the model's initial state (key frame 0, else qpos0) divided by ``margin`` (it may come
    that much closer). At ``full_px`` pixels or more: MuJoCo's tessellation (``vis.quality``); thinner: 20 slices and
    4 rings per cap; under ``subpixel_px``: 8 slices and 2 rings. Pixel centres sample a tube's width uniformly, so the
    coverage an n-gon loses is relative, not absolute: its mean silhouette width is (n / pi) sin(pi / n) of the circle's
    (n = 20: 0.996, worst orientation cos(pi / n) = 0.988) whatever the pixel size; under half a pixel the tube's own
    silhouette is sampling noise. Returns {slot: (slices, rings per cap)}."""
    import mujoco
    full = (int(m.vis.quality.numslices), max(int(m.vis.quality.numstacks) // 2, 1))
    d = data
    if d is None:
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    vp = np.asarray(viewpoints, np.float64).reshape(-1, 3)
    out = {}
    for dd in tables.dyn:
        if dd["kind"] == "tendon":
            i = dd["id"]
            a, n = int(d.ten_wrapadr[i]), int(d.ten_wrapnum[i])
            pts = np.asarray(d.wrap_xpos).reshape(-1, 3)[a:a + n]
            r = float(m.tendon_width[i])
        elif dd["kind"] == "flex" and dd.get("lines"):
            f = dd["id"]
            pts = d.flexvert_xpos[m.flex_vertadr[f]: m.flex_vertadr[f] + m.flex_vertnum[f]]
            r = float(m.flex_radius[f])
        else:
            continue
        if len(pts) == 0 or not len(vp) or resolution is None:
            out[dd["slot"]] = full
            continue
        dist = max(float(np.min(np.linalg.norm(pts[:, None, :] - vp[None, :, :], axis=-1))) / margin, 1e-6)
        r_px = r / (dist * float(resolution))
        if 2.0 * r_px >= full_px:
            out[dd["slot"]] = full
            continue
        lod = (20, 4) if 2.0 * r_px >= subpixel_px else (8, 2)
        out[dd["slot"]] = (min(lod[0], full[0]), min(lod[1], full[1]))
    return out


class DynamicMeshes:
    """Per-env vertex records of the dynamic slots, generated on the GPU each frame.

    ``vbuf`` / ``rbase``: write into an existing vertex buffer from record ``rbase`` on (the ray tracer's, after its
    static vertices); None allocates one. ``tpr`` / ``stride``: the ray tracer's env grid (offsets baked into the
    positions; 0 for a raster-only renderer). ``tube_slices`` / ``cap_stacks``: tessellation of the cylinders, capsules
    and spheres (default MuJoCo's ``vis.quality.numslices`` and ``numstacks / 2``)."""

    def __init__(self, ctx, model, tables, n_envs, *, tpr=1, stride=0.0, vbuf=None, rbase=0, tube_slices=None, cap_stacks=None,
                 tube_lod=None, tendons="mujoco"):
        self.ctx, self.m, self.t, self.n = ctx, model, tables, n_envs
        self.tpr, self.stride = int(tpr), float(stride)
        plan = _Plan(model, tables, tube_slices, cap_stacks, tube_lod, tendons)
        plan.finalize(n_envs)
        plan.validate(n_envs)
        self.plan = plan
        self.objects = plan.objects
        self.n_records = plan.rtot
        self.ptot = max(plan.ptot, 1)
        c = ctx
        self.rbase = int(rbase)
        self.vbuf = vbuf if vbuf is not None else c.buffer(32 * max(self.n_records, 1), label="dyn_vertices")
        A = plan.arrays()
        self.bufs = {k: c.buffer(v.nbytes, v, "dyn_" + k) for k, v in A.items()}
        self.n_srec, self.n_trec, self.n_nent, self.n_skin = len(plan.srec), len(plan.trec), len(plan.nent), len(plan.skin_v)
        self.P = c.buffer(4 * 3 * n_envs * self.ptot, label="dyn_positions")
        self.Nn = c.buffer(4 * 3 * n_envs * self.ptot, label="dyn_normals")
        # raster: local triangles of every object, one draw (instance = env) per object
        loc, self.draws = [], []
        off = 0
        for ob in self.objects:
            loc.append(ob["tris"].reshape(-1).astype(np.uint32))
            self.draws.append((off, int(ob["tris"].size), self.rbase + ob["obj_base"], ob["nvert"], ob["slot"]))
            off += int(ob["tris"].size)
        li = np.concatenate(loc) if loc else np.zeros(3, np.uint32)
        self.local_ibuf = c.buffer(max(li.nbytes, 16), li if li.size else None, "dyn_local_indices")
        lib = c.library("dynamic_mesh")
        self.p = {k: c.compute_pipeline(lib, k) for k in ("copy_block", "skin_lbs", "vnormal", "emit_surface", "emit_tube",
                                                         "catenary", "tendon_tint")}
        # catenary points (emit_tube mode 3) and the limit-colour records (tendon_tint)
        self.n_cat, self.n_tint, self.ncat_pts = len(plan.cat), len(plan.tint), plan.ncat_pts
        self.CP = c.buffer(16 * n_envs * max(self.n_cat, 1) * self.ncat_pts, label="dyn_catenary_points")
        self.CPn = c.buffer(4 * n_envs * max(self.n_cat, 1), label="dyn_catenary_count")
        g = np.asarray(model.opt.gravity, np.float64)
        up = -g / max(np.linalg.norm(g), 1e-30)
        self.cat_consts = np.zeros(8, np.uint32)
        self.cat_consts.view(np.float32)[0:3] = up
        self.cat_consts[4:6] = (self.ncat_pts, max(self.n_cat, 1))
        self.tint_consts = np.zeros(8, np.uint32)
        self.tint_consts.view(np.float32)[0:4] = model.vis.rgba.constraint
        self.tint_consts[4:8] = (self.n_tint, tables.G, max(int(model.ntendon), 1), n_envs)
        self.skin_vert = c.buffer(4 * 3 * max(model.nskinvert, 1), model.skin_vert.astype(np.float32) if model.nskinvert else None, "dyn_skin_vert")
        # host-path source buffers
        nw = max(2 * int(model.nwrap), 1)
        nt = max(int(model.ntendon), 1)
        self._host = {"flexvert_xpos": c.buffer(4 * 3 * n_envs * max(model.nflexvert, 1), label="dyn_h_flexvert"),
                      "xpos": c.buffer(4 * 3 * n_envs * model.nbody, label="dyn_h_xpos"),
                      "xquat": c.buffer(4 * 4 * n_envs * model.nbody, label="dyn_h_xquat"),
                      "ten_wrapadr": c.buffer(4 * n_envs * nt, label="dyn_h_wrapadr"),
                      "ten_wrapnum": c.buffer(4 * n_envs * nt, label="dyn_h_wrapnum"),
                      "wrap_obj": c.buffer(4 * n_envs * nw, label="dyn_h_wrapobj"),
                      "wrap_xpos": c.buffer(4 * 3 * n_envs * nw, label="dyn_h_wrapxpos"),
                      "ten_length": c.buffer(4 * n_envs * nt, label="dyn_h_tenlength")}
        self._host_mesh = {}
        for d in tables.dyn:
            if d["kind"] == "mesh":
                self._host_mesh[d["mesh"].name] = c.buffer(4 * 3 * n_envs * d["mesh"].nvert, label="dyn_h_" + d["mesh"].name)
        self.kinds = {d["kind"] for d in tables.dyn}
        self.mesh_names = [d["mesh"].name for d in tables.dyn if d["kind"] == "mesh"]

    # -- RT support -------------------------------------------------------------------------------------------------

    def expanded_indices(self):
        """Per object, (N * F * 3,) uint32 indices into the vertex buffer of every env's triangles (ray tracing)."""
        out = []
        for ob in self.objects:
            e = np.arange(self.n, dtype=np.int64)[:, None] * ob["nvert"]
            out.append((ob["tris"].reshape(1, -1) + e + self.rbase + ob["obj_base"]).reshape(-1).astype(np.uint32))
        return out

    # -- sources ------------------------------------------------------------------------------------------------------

    def sources_from_sim(self, d, overrides=None):
        """(buffer, offset) per source name from a MuJoCo Warp ``Data`` (``sim.d``) and ``overrides`` ({name: Warp
        array}: ``flexvert_xpos`` (e.g. ``XPBDSim.x``) and one (N, V, 3) array per user mesh)."""
        from metalsim.interop import warp_metal as wm
        src = {}
        overrides = dict(overrides or {})
        names = []
        if "flex" in self.kinds:
            names.append("flexvert_xpos")
        if "skin" in self.kinds:
            names += ["xpos", "xquat"]
        if "tendon" in self.kinds:
            names += ["ten_wrapadr", "ten_wrapnum", "wrap_obj", "wrap_xpos"]
            if self.n_cat or self.n_tint:
                names.append("ten_length")
        for k in names:
            a = overrides.pop(k, None)
            if a is None:
                a = getattr(d, k)
            v = wm.buffer_of(a)
            src[k] = (v.buffer, v.offset)
        for k in self.mesh_names:
            a = overrides.pop(k, None)
            if a is None:
                raise ValueError(f"deformable mesh {k!r}: pass its (N, V, 3) vertex array in sources")
            v = wm.buffer_of(a)
            src[k] = (v.buffer, v.offset)
        if overrides:
            raise ValueError(f"unknown dynamic sources {sorted(overrides)}")
        return src

    def write_host(self, datas, overrides=None):
        """Host path: write the sources from N ``mujoco.MjData`` (and ``overrides``: {name: (N, V, 3) array}) into
        host buffers; returns the source dict."""
        c, n, m = self.ctx, self.n, self.m
        overrides = dict(overrides or {})
        src = {}
        if "flex" in self.kinds:
            fx = overrides.pop("flexvert_xpos", None)
            fx = np.stack([dd.flexvert_xpos for dd in datas]) if fx is None else np.asarray(fx)
            c.write_buffer(self._host["flexvert_xpos"], fx.astype(np.float32).reshape(n, -1, 3))
            src["flexvert_xpos"] = (self._host["flexvert_xpos"], 0)
        if "skin" in self.kinds:
            c.write_buffer(self._host["xpos"], np.stack([dd.xpos for dd in datas]).astype(np.float32))
            c.write_buffer(self._host["xquat"], np.stack([dd.xquat for dd in datas]).astype(np.float32))
            src["xpos"] = (self._host["xpos"], 0); src["xquat"] = (self._host["xquat"], 0)
        if "tendon" in self.kinds:
            nw = 2 * int(m.nwrap)
            c.write_buffer(self._host["ten_wrapadr"], np.stack([dd.ten_wrapadr for dd in datas]).astype(np.int32))
            c.write_buffer(self._host["ten_wrapnum"], np.stack([dd.ten_wrapnum for dd in datas]).astype(np.int32))
            c.write_buffer(self._host["wrap_obj"], np.stack([np.asarray(dd.wrap_obj).reshape(-1)[:nw] for dd in datas]).astype(np.int32))
            c.write_buffer(self._host["wrap_xpos"], np.stack([np.asarray(dd.wrap_xpos).reshape(-1)[:3 * nw] for dd in datas]).astype(np.float32))
            for k in ("ten_wrapadr", "ten_wrapnum", "wrap_obj", "wrap_xpos"):
                src[k] = (self._host[k], 0)
            if self.n_cat or self.n_tint:
                c.write_buffer(self._host["ten_length"], np.stack([dd.ten_length for dd in datas]).astype(np.float32))
                src["ten_length"] = (self._host["ten_length"], 0)
        for k in self.mesh_names:
            a = overrides.pop(k, None)
            if a is None:
                raise ValueError(f"deformable mesh {k!r}: pass its (N, V, 3) vertex array in sources")
            c.write_buffer(self._host_mesh[k], np.asarray(a, np.float32).reshape(n, -1, 3))
            src[k] = (self._host_mesh[k], 0)
        if overrides:
            raise ValueError(f"unknown dynamic sources {sorted(overrides)}")
        return src

    # -- encode -------------------------------------------------------------------------------------------------------

    def _consts(self, **kw):
        a = np.zeros(_DYNC, np.uint32)
        a[0:4] = (self.n, self.ptot, self.tpr, self.rbase)
        a.view(np.float32)[4] = self.stride
        a[8:12] = (kw.get("n_src", 0), kw.get("src_stride", 0), kw.get("src_start", 0), kw.get("dst", 0))
        a[12:16] = (self.m.nbody, max(self.m.ntendon, 1), 2 * int(self.m.nwrap), kw.get("n_rec", 0))
        return a

    def _dispatch(self, ce, pipe, bufs, consts, nx, ci):
        ce.setComputePipelineState_(pipe)
        for i, (b, o) in enumerate(bufs):
            ce.setBuffer_offset_atIndex_(b, o, i)
        ce.setBytes_length_atIndex_(consts.tobytes(), consts.nbytes, ci)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(int(nx), self.n, 1), Metal.MTLSize(64, 1, 1))

    def encode(self, cb, src):
        """Encode the generation of every dynamic slot's records from ``src`` into ``cb`` (one compute encoder, serial
        dispatches)."""
        if not self.objects:
            return
        B = self.bufs
        ce = cb.computeCommandEncoder()
        P, Nn = (self.P, 0), (self.Nn, 0)
        for kind, key, p_off, nv in self.plan.blocks:
            s = src["flexvert_xpos"] if kind == "flex" else src[key]
            self._dispatch(ce, self.p["copy_block"], [s, P], self._consts(n_src=nv, src_stride=nv, dst=p_off), nv, 2)
        if self.n_skin:
            self._dispatch(ce, self.p["skin_lbs"], [(B["skin_v"], 0), (B["skin_b"], 0), (B["bones"], 0), (self.skin_vert, 0),
                                                     src["xpos"], src["xquat"], P], self._consts(n_rec=self.n_skin), self.n_skin, 7)
        if self.plan.nent:
            self._dispatch(ce, self.p["vnormal"], [(B["nent"], 0), (B["flist"], 0), (B["faces"], 0), P, Nn],
                           self._consts(n_rec=self.n_nent), self.n_nent, 5)
        if self.plan.srec:
            self._dispatch(ce, self.p["emit_surface"], [(B["srec"], 0), P, Nn, (self.vbuf, 0)],
                           self._consts(n_rec=self.n_srec), self.n_srec, 4)
        if self.plan.trec:
            dummy = (B["flist"], 0)
            if self.n_cat:          # catenary points of the slack two-site tendons (before the tubes read them)
                ce.setBytes_length_atIndex_(self.cat_consts.tobytes(), self.cat_consts.nbytes, 7)
                self._dispatch(ce, self.p["catenary"], [(B["cat"], 0), src["ten_wrapadr"], src["wrap_xpos"], src["ten_length"],
                                                        (self.CP, 0), (self.CPn, 0)], self._consts(), self.n_cat, 6)
            ce.setBytes_length_atIndex_(self.cat_consts.tobytes(), self.cat_consts.nbytes, 11)
            ce.setBuffer_offset_atIndex_(self.CP, 0, 9)
            ce.setBuffer_offset_atIndex_(self.CPn, 0, 10)
            self._dispatch(ce, self.p["emit_tube"], [(B["trec"], 0), (B["tmpl"], 0), P,
                                                     src.get("ten_wrapadr", dummy), src.get("ten_wrapnum", dummy),
                                                     src.get("wrap_obj", dummy), src.get("wrap_xpos", dummy), (self.vbuf, 0)],
                           self._consts(n_rec=self.n_trec), self.n_trec, 8)
        ce.endEncoding()

    def encode_tint(self, cb, src, t_colors, inst_col):
        """MuJoCo's limit colour of default-grey tendons: after the model sync has written ``inst_col`` (n_envs, G, 4)
        from ``t_colors``, overwrite the tinted tendons' slots (shaders/dynamic_mesh.metal tendon_tint)."""
        if not self.n_tint or src is None:
            return
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.p["tendon_tint"])
        ce.setBuffer_offset_atIndex_(self.bufs["tint"], 0, 0)
        b, o = src["ten_length"]
        ce.setBuffer_offset_atIndex_(b, o, 1)
        ce.setBuffer_offset_atIndex_(t_colors, 0, 2)
        ce.setBuffer_offset_atIndex_(inst_col, 0, 3)
        ce.setBytes_length_atIndex_(self.tint_consts.tobytes(), self.tint_consts.nbytes, 4)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n_tint, self.n, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()

    def records(self):
        """(n_records, 8) float32 view of the generated records (synchronize first)."""
        return self.ctx.buffer_array(self.vbuf, np.float32, (self.rbase + self.n_records, 8))[self.rbase:]

    def object_vertices(self, k, env):
        """(V, 8) records of dynamic object k in env (synchronize first)."""
        ob = self.objects[k]
        r = self.records()
        a = ob["obj_base"] + env * ob["nvert"]
        return r[a:a + ob["nvert"]]


class PoseGather:
    """Per frame, the pose rows the renderer reads: the geom poses followed by the drawn sites' (``SceneTables.n_pose``
    rows per env)."""

    def __init__(self, ctx, model, tables, n_envs):
        self.ctx, self.m, self.n = ctx, model, n_envs
        self.site_ids = np.asarray(tables.site_ids, np.int32)
        self.rows = model.ngeom + len(self.site_ids)
        c = ctx
        self.ids = c.buffer(max(self.site_ids.nbytes, 16), self.site_ids, "gather_site_ids")
        self.xpos = c.buffer(4 * 3 * n_envs * self.rows, label="pose_xpos")
        self.xmat = c.buffer(4 * 9 * n_envs * self.rows, label="pose_xmat")
        self.pipe = c.compute_pipeline(c.library("dynamic_mesh"), "gather_poses")
        self.consts = np.array([n_envs, model.ngeom, max(model.nsite, 1), len(self.site_ids)], np.uint32)

    def encode(self, cb, gx, gm, sx, sm):
        """(buffer, offset) of geom_xpos, geom_xmat, site_xpos, site_xmat -> (xpos, xmat) buffers (offset 0)."""
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        for i, (b, o) in enumerate((gx, gm, sx, sm, (self.ids, 0), (self.xpos, 0), (self.xmat, 0))):
            ce.setBuffer_offset_atIndex_(b, o, i)
        ce.setBytes_length_atIndex_(self.consts.tobytes(), self.consts.nbytes, 7)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.rows, self.n, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()
        return (self.xpos, 0), (self.xmat, 0)

    def host(self, gx, gm, datas):
        """Host path: (N, ngeom, 3), (N, ngeom, 9) geom poses + the sites' from MjData -> (N, rows, 3), (N, rows, 9)."""
        if not len(self.site_ids):
            return gx, gm
        sx = np.stack([d.site_xpos[self.site_ids] for d in datas]).astype(np.float32)
        sm = np.stack([d.site_xmat[self.site_ids] for d in datas]).astype(np.float32).reshape(self.n, -1, 9)
        return np.concatenate([gx, sx], 1), np.concatenate([gm, sm], 1)
