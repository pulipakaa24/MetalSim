"""Newton -> MetalSim renderer bridge: a Newton ``Model``/``State`` as a render source for tiers 0/1/2, the ray-traced
sensors and the Replicator annotators.

Every MetalSim renderer compiles its scene from a ``mujoco.MjModel`` (``scene_tables.build_scene_tables``) and reads the
per-frame poses from a "sim" object: ``sim.d.geom_xpos`` / ``geom_xmat`` (N, ngeom), ``cam_xpos`` / ``cam_xmat``,
``light_xpos`` / ``light_xdir``, ``site_xpos`` / ``site_xmat``, the body poses ``xpos`` / ``xmat`` (annotators), torch
views ``sim.t.*`` and the completion event ``sim.event``. ``NewtonRenderSource`` is that object for a Newton model:

* **Render twin.** A ``mujoco.MjModel`` with one geom per rendered Newton shape of one world (plus the global, world -1,
  shapes), compiled once with ``mujoco.MjSpec`` (``twin="newton"``): primitives keep their type and size, meshes /
  convex meshes / cones become mesh assets (the scale baked in), heightfields become hfields, colours come from
  ``Model.shape_color`` (opacity from ``Model.shape_opacity``, Newton >= 1.6). Or the MJCF the Newton model was imported
  from (``twin=<MjModel>``, the MuJoCo-via-Newton path): its geoms, materials, textures, lights and cameras are kept and
  each MuJoCo body is matched to the Newton body of the same name in every world. The twin is only a render
  description; nothing simulates it, and the renderers are used unchanged.
* **Per-frame poses.** Warp kernels write the twin's pose buffers from ``State.body_q`` (geom = body_q[body] x local
  shape transform x the twin's compiled geom frame) for every world in one launch per frame kind: no host work, the
  same arrays every frame, so ``update`` can replay a captured graph and the launches can also be recorded into the
  caller's own Newton step graph (``launch``).
* **Worlds.** Env ``e`` draws world ``e``'s shapes plus the global shapes, as Isaac Lab 3.0's Newton renderer does
  (each camera sees its own world and world -1). Poses are written relative to an env origin (``env_origins``), so the
  renderers' per-env tiling and the ray tracer's per-env spatial offset see each world around its own origin; cameras
  and lights given without a body are in that env frame. When the worlds do not share one shape layout, the twin holds
  every world's shapes and env ``e`` hides the others (warned; the cost grows with the number of worlds).
* **Cameras, lights, sites.** Newton has no cameras or lights; they are given here (``Camera``, ``Light``), either fixed
  in the env frame or mounted on a Newton body. Newton ``SITE`` shapes become MuJoCo sites (lidar / radar mounts).

Deformables (``particle_q``, cloth and soft-body surface triangles) are not drawn by this module: they go through the
renderers' dynamic-mesh and particle paths (workstream A and the MPM particle renderer) once those land; see
docs/RENDER_COVERAGE.md. The shape-layer differences from Isaac Lab 3.0 are listed in docs/NEWTON_RENDER.md.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import mujoco
import numpy as np
import warp as wp

HIDDEN_BODY = -2          # slot table entry: shape not present in this env (drawn far below the env)
HIDDEN_Z = -1.0e6

# -- small numpy transform helpers (Newton transforms: p xyz, q xyzw; MuJoCo quats: wxyz) -----------------------------


def _qmul(a, b):
    """xyzw quaternion product."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _qrot(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _tf(p=(0, 0, 0), q=(0, 0, 0, 1)) -> np.ndarray:
    return np.concatenate([np.asarray(p, float), np.asarray(q, float)])


def _tf_mul(a, b) -> np.ndarray:
    return np.concatenate([a[:3] + _qrot(a[3:], b[:3]), _qmul(a[3:], b[3:])])


def _tf_inv(a) -> np.ndarray:
    qi = np.array([-a[3], -a[4], -a[5], a[6]])
    return np.concatenate([-_qrot(qi, a[:3]), qi])


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], float)


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], float)


def lookat_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera orientation (wxyz; the camera looks along its -Z, +Y up) at ``pos`` looking at ``target``."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(f, (0.0, 1.0, 0.0))
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    R = np.stack([x, y, -f], 1)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def _label_tail(label: str) -> str:
    return str(label).split("/")[-1]


# -- configuration ----------------------------------------------------------------------------------------------------


@dataclass
class Camera:
    """A pinhole camera in MuJoCo's convention (looks along -Z, +Y up). ``body``: a Newton body label (or its last
    path component) or a body index within a world; the pose is then relative to that body. Otherwise the pose is in
    the env frame. ``quat`` (wxyz) or ``lookat`` (a point in the same frame) sets the orientation."""
    name: str = "cam"
    pos: tuple = (1.5, -1.5, 1.2)
    quat: tuple | None = None
    lookat: tuple | None = (0.0, 0.0, 0.3)
    up: tuple = (0.0, 0.0, 1.0)
    fovy: float = 45.0
    body: str | int | None = None
    # physical pinhole (MuJoCo ``sensorsize`` / ``focal`` / ``principal`` in metres; Isaac's aperture / focal length):
    # overrides ``fovy`` when ``sensorsize`` is set
    sensorsize: tuple | None = None
    focal: tuple | None = None
    principal: tuple = (0.0, 0.0)
    resolution: tuple | None = None


@dataclass
class Light:
    """A MuJoCo light (``directional`` or spot). ``body`` as for ``Camera``."""
    pos: tuple = (0.0, 0.0, 3.0)
    dir: tuple = (0.3, 0.2, -1.0)
    directional: bool = True
    diffuse: tuple = (0.7, 0.7, 0.7)
    specular: tuple = (0.3, 0.3, 0.3)
    ambient: tuple = (0.0, 0.0, 0.0)
    castshadow: bool = True
    body: str | int | None = None


@dataclass
class _Slot:
    """One twin frame (geom, site, camera, light or body) per env: Newton body index or -1 (static) / HIDDEN_BODY,
    and the local transform (static: world frame)."""
    body: np.ndarray      # (n,) int32
    xf: np.ndarray        # (n, 7) float64


class _Data:
    """MuJoCo-layout pose arrays the renderers read (subset of mujoco_warp.Data)."""


# -- kernels ----------------------------------------------------------------------------------------------------------


@wp.kernel
def _frames(body_q: wp.array[wp.transform], fb: wp.array[wp.int32], fx: wp.array[wp.transform],
            origin: wp.array[wp.vec3], J: int, out_pos: wp.array2d[wp.vec3], out_mat: wp.array2d[wp.mat33]):
    e, j = wp.tid()
    i = e * J + j
    b = fb[i]
    X = fx[i]
    if b >= 0:
        X = wp.transform_multiply(body_q[b], X)
    p = wp.transform_get_translation(X) - origin[e]
    if b == -2:
        p = wp.vec3(0.0, 0.0, -1.0e6)
    out_pos[e, j] = p
    out_mat[e, j] = wp.quat_to_matrix(wp.transform_get_rotation(X))


@wp.kernel
def _lights(body_q: wp.array[wp.transform], lb: wp.array[wp.int32], lx: wp.array[wp.transform],
            lpos: wp.array[wp.vec3], ldir: wp.array[wp.vec3], origin: wp.array[wp.vec3], L: int,
            out_pos: wp.array2d[wp.vec3], out_dir: wp.array2d[wp.vec3]):
    e, j = wp.tid()
    i = e * L + j
    b = lb[i]
    X = lx[i]
    if b >= 0:
        X = wp.transform_multiply(body_q[b], X)
    out_pos[e, j] = wp.transform_point(X, lpos[j]) - origin[e]
    out_dir[e, j] = wp.transform_vector(X, ldir[j])


@wp.kernel
def _gather_points(q: wp.array[wp.vec3], flags: wp.array[wp.int32], idx: wp.array[wp.int32], origin: wp.array[wp.vec3],
                   V: int, out: wp.array2d[wp.vec3]):
    """Per env, its particles (cloth / soft-body surface vertices or free particles) in the env frame; absent or
    inactive ones far below the env."""
    e, v = wp.tid()
    i = idx[e * V + v]
    p = wp.vec3(0.0, 0.0, -1.0e6)
    if i >= 0:
        if (flags[i] & 1) != 0:
            p = q[i] - origin[e]
    out[e, v] = p


# -- shape description ------------------------------------------------------------------------------------------------


@dataclass
class _ShapeInfo:
    shape: int
    world: int
    body: int
    gtype: int
    scale: tuple
    src: object
    flags: int
    rgba: tuple
    label: str
    key: tuple = field(default=())


class NewtonRenderSource:
    """A Newton model as the ``sim`` argument of ``Tier0Renderer`` / ``Tier2Renderer`` / ``RayTracer`` /
    ``IsaacAnnotators``: build the renderer from ``self.m``, then each frame ``v = src.update(state)`` and
    ``renderer.render(src, v)``.

    ``twin``: "newton" (render twin compiled from the Newton shapes) or the ``mujoco.MjModel`` of the MJCF the Newton
    model was imported from (MuJoCo-via-Newton; ``SolverMuJoCo`` or any other solver).
    ``shapes`` (twin "newton"): "visible" (``ShapeFlags.VISIBLE``, Newton's viewer and Isaac Lab's default), "collision"
    (``COLLIDE_SHAPES``) or "all". ``collider_fallback``: bodies with no rendered shape draw their primitive colliders
    (Isaac Lab 3.0's ``newton_clone_utils`` rule; imported collision meshes stay hidden).
    ``cameras`` / ``lights``: ``Camera`` / ``Light`` lists (twin "newton": a default camera and directional light when
    None; an MJCF twin keeps its own and appends these). ``env_origins``: (N, 3) env frame origins in the Newton world,
    "auto" (each world's mean initial body position minus the mean over worlds: exact for ``ModelBuilder.replicate``
    grids and Isaac Lab's env grid, both centred on the origin) or None (all zero).
    ``spec_hook``: called with the ``mujoco.MjSpec`` before compiling (twin "newton"; materials, extra cameras, ...)."""

    def __init__(self, model, *, twin="newton", shapes: str = "visible", collider_fallback: bool = True,
                 cameras=None, lights=None, env_origins="auto", headlight: bool = True, spec_hook=None,
                 deformables: bool = True, device=None):
        self.newton = model
        self.device = wp.get_device(device or model.device)
        self.shape_mode = shapes
        nw = int(getattr(model, "world_count", 0) or 0)
        self.n = max(nw, 1)
        self._nw = nw
        self._body_q0 = model.body_q.numpy().astype(np.float64) if model.body_count else np.zeros((0, 7))
        self._body_world = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
        self._labels = list(getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])
        self.origins = self._env_origins(env_origins)
        if isinstance(twin, mujoco.MjModel):
            self.kind = "mjcf"
            self._build_from_mjcf(twin, cameras or [], lights or [])
        elif twin == "newton":
            self.kind = "newton"
            self._build_from_newton(collider_fallback, cameras, lights, headlight, spec_hook)
        else:
            raise ValueError("twin must be 'newton' or a mujoco.MjModel")
        self._build_particles(deformables)
        self._alloc()
        # CPU models, or no pyobjc (the .venv-newton152 environment): no event; the host path (``host_datas``) works
        self.event, self._wm = None, None
        if self.device.is_metal:
            try:
                from metalsim.interop import warp_metal as wm
                self.event, self._wm = wm.SharedEvent(str(self.device), "metalsim.render.newton"), wm
            except ImportError:
                pass
        self._graphs = {}

    # -- env frames ---------------------------------------------------------------------------------------------------

    def _env_origins(self, env_origins) -> np.ndarray:
        n = self.n
        if env_origins is None:
            return np.zeros((n, 3))
        if isinstance(env_origins, str):
            if env_origins != "auto":
                raise ValueError(env_origins)
            # each world's mean initial body position, relative to the centre of the worlds (``replicate`` and Isaac
            # Lab's env grid are centred on the origin)
            o = np.zeros((n, 3))
            if self._nw and len(self._body_q0):
                has = np.zeros(n, bool)
                for e in range(n):
                    sel = self._body_world == e
                    if sel.any():
                        o[e] = self._body_q0[sel, :3].mean(0); has[e] = True
                if has.any():
                    o[has] -= o[has].mean(0)
            return o
        o = np.asarray(env_origins, float).reshape(n, 3)
        return o

    def _world_bodies(self, e: int) -> np.ndarray:
        if not self._nw:
            return np.arange(len(self._body_world))
        return np.flatnonzero(self._body_world == e)

    def _resolve_body(self, spec, e: int) -> int:
        """Newton body index in env ``e`` for a camera/light ``body`` spec (label, label tail or in-world index)."""
        wb = self._world_bodies(e)
        if isinstance(spec, (int, np.integer)):
            return int(wb[int(spec)])
        glob = np.flatnonzero(self._body_world < 0) if self._nw else []        # global bodies: in every env
        hits = [b for b in list(wb) + list(glob) if self._labels[b] == spec or _label_tail(self._labels[b]) == spec]
        if not hits:
            raise ValueError(f"no Newton body {spec!r} in world {e}")
        return int(hits[0])

    # -- twin from Newton shapes --------------------------------------------------------------------------------------

    def _shape_infos(self, collider_fallback: bool):
        m = self.newton
        G = newton_geo()
        F = newton_flags()
        S = m.shape_count
        types = m.shape_type.numpy(); scales = m.shape_scale.numpy(); flags = m.shape_flags.numpy()
        bodies = m.shape_body.numpy(); worlds = m.shape_world.numpy() if S else np.zeros(0, int)
        colors = m.shape_color.numpy() if getattr(m, "shape_color", None) is not None else None
        opac = m.shape_opacity.numpy() if getattr(m, "shape_opacity", None) is not None else None
        labels = list(getattr(m, "shape_label", None) or getattr(m, "shape_key", None) or [f"shape_{i}" for i in range(S)])
        want = {"visible": F.VISIBLE, "collision": F.COLLIDE_SHAPES, "all": F.VISIBLE | F.COLLIDE_SHAPES}[self.shape_mode]
        drawable = np.array([int(flags[s]) & int(want) != 0 and not (int(flags[s]) & int(F.SITE))
                             and int(types[s]) not in (int(G.GAUSSIAN), int(G.NONE)) for s in range(S)], bool)
        if collider_fallback and self.shape_mode == "visible":
            # a body (or a world's static shapes, taken as one group) with no visible shape draws its primitive
            # colliders; static groups also their planes and heightfields (an MJCF floor is a collider only)
            grp = lambda s: int(bodies[s]) if bodies[s] >= 0 else ("static", int(worlds[s]))
            covered = set(grp(s) for s in range(S) if drawable[s])
            prim = {int(G.SPHERE), int(G.BOX), int(G.CAPSULE), int(G.CYLINDER), int(G.ELLIPSOID), int(G.CONE)}
            for s in range(S):
                ok = prim | ({int(G.PLANE), int(G.HFIELD)} if bodies[s] < 0 else set())
                if (not drawable[s] and grp(s) not in covered and int(flags[s]) & int(F.COLLIDE_SHAPES)
                        and int(types[s]) in ok):
                    drawable[s] = True
        if (types == int(G.GAUSSIAN)).any():
            warnings.warn("Newton Gaussian shapes are not drawn by the MetalSim bridge")
        infos, sites = [], []
        src_keys = {}
        for s in range(S):
            is_site = bool(int(flags[s]) & int(F.SITE))
            if not drawable[s] and not is_site:
                continue
            rgba = (0.5, 0.5, 0.5, 1.0)
            if colors is not None:
                rgba = (*[float(c) for c in colors[s][:3]], float(opac[s]) if opac is not None else 1.0)
            src = m.shape_source[s] if hasattr(m, "shape_source") else None
            if colors is None and src is not None and getattr(src, "color", None) is not None:
                rgba = (*[float(c) for c in src.color[:3]], 1.0)
            if src is not None and int(types[s]) in (int(G.MESH), int(G.CONVEX_MESH), int(G.HFIELD)):
                sk = src_keys.get(id(src))
                if sk is None:
                    sk = _geometry_hash(src)
                    src_keys[id(src)] = sk
            else:
                sk = None
            info = _ShapeInfo(s, int(worlds[s]), int(bodies[s]), int(types[s]), tuple(np.round(scales[s], 7)), src,
                              int(flags[s]), rgba, labels[s])
            info.key = (info.gtype, info.scale, sk, info.rgba)
            (sites if is_site else infos).append(info)
        return infos, sites

    def _build_from_newton(self, collider_fallback, cameras, lights, headlight, spec_hook):
        m = self.newton
        n = self.n
        infos, site_infos = self._shape_infos(collider_fallback)
        tf_np = m.shape_transform.numpy().astype(np.float64) if m.shape_count else np.zeros((0, 7))
        bws = None
        if self._nw:
            bws = m.body_world_start.numpy() if hasattr(m, "body_world_start") else None

        def local_body(b, w):
            if b < 0 or not self._nw or w < 0:
                return b
            return b - int(bws[w]) if bws is not None else b

        glob = [i for i in infos if i.world < 0 or not self._nw]
        per = [[i for i in infos if i.world == e] for e in range(n)] if self._nw else [[]]
        sig = lambda lst: [(i.key, local_body(i.body, i.world) if i.body >= 0 else -1) for i in lst]
        homog = all(sig(per[e]) == sig(per[0]) for e in range(n))
        if not homog:
            warnings.warn("Newton worlds have different shape layouts: the render twin holds every world's shapes and "
                          "each env hides the others (cost grows with the number of worlds)")
        # template: (info for the twin's geometry, per-env shape index or -1 hidden)
        template = [(i, np.full(n, i.shape)) for i in glob]
        if homog:
            for k, i in enumerate(per[0]):
                template.append((i, np.array([per[e][k].shape for e in range(n)])))
        else:
            for e in range(n):
                for i in per[e]:
                    col = np.full(n, -1); col[e] = i.shape
                    template.append((i, col))
        # sites: world 0 (+ global) layout, mapped per world by order (homogeneous worlds)
        gsites = [i for i in site_infos if i.world < 0 or not self._nw]
        wsites = [[i for i in site_infos if i.world == e] for e in range(n)] if self._nw else [[]]
        site_tmpl = [(i, np.full(n, i.shape)) for i in gsites]
        if all(len(wsites[e]) == len(wsites[0]) for e in range(n)):
            for k, i in enumerate(wsites[0]):
                site_tmpl.append((i, np.array([wsites[e][k].shape for e in range(n)])))
        # twin bodies: the template bodies (env 0's, or each world's in the heterogeneous layout)
        body_ids = []
        for i, col in template + site_tmpl:
            if i.body >= 0 and i.body not in body_ids:
                body_ids.append(i.body)
        cams = list(cameras) if cameras is not None else [Camera()]
        lts = list(lights) if lights is not None else [Light()]
        for c in cams + lts:
            if c.body is not None:
                b0 = self._resolve_body(c.body, 0)
                if b0 not in body_ids:
                    body_ids.append(b0)

        spec = mujoco.MjSpec()
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
        spec.visual.headlight.active = int(bool(headlight))
        twin_body = {}
        used = set()
        for b in body_ids:
            nm = _label_tail(self._labels[b]) if b < len(self._labels) else f"body_{b}"
            if nm in used or not nm:
                nm = f"{nm}_{b}"
            used.add(nm)
            sb = spec.worldbody.add_body(name=nm)
            sb.mass = 1.0; sb.inertia = [1e-3, 1e-3, 1e-3]; sb.explicitinertial = True
            twin_body[b] = sb
        mesh_cache = {}
        extra = []                     # per geom: local transform applied before the twin's compiled geom frame
        names, gnames, snames = set(), [], []
        G = newton_geo()
        for k, (i, col) in enumerate(template):
            parent = twin_body[i.body] if i.body >= 0 else spec.worldbody
            gname = _label_tail(i.label) or f"shape_{i.shape}"
            if gname in names:
                gname = f"{gname}_{i.shape}"
            names.add(gname)
            gnames.append(gname)
            kw = dict(name=gname, contype=0, conaffinity=0, rgba=list(i.rgba), group=0)
            sc = i.scale
            pre = _tf()
            t = i.gtype
            if t == int(G.SPHERE):
                kw.update(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[sc[0], 0, 0])
            elif t == int(G.BOX):
                kw.update(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(sc[:3]))
            elif t == int(G.CAPSULE):
                kw.update(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[sc[0], sc[1], 0])
            elif t == int(G.CYLINDER):
                kw.update(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[sc[0], sc[1], 0])
            elif t == int(G.ELLIPSOID):
                kw.update(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=list(sc[:3]))
            elif t == int(G.PLANE):
                # Newton: full width / length (0 = infinite); MuJoCo: half sizes (0 = infinite)
                kw.update(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5 * sc[0], 0.5 * sc[1], 0.1])
            elif t in (int(G.MESH), int(G.CONVEX_MESH), int(G.CONE)):
                mk = (i.key[2], sc) if t != int(G.CONE) else ("cone", sc)
                if mk not in mesh_cache:
                    v, f = _mesh_arrays(i, G)
                    mname = f"mesh_{len(mesh_cache)}"
                    me = spec.add_mesh(name=mname)
                    me.uservert = v.reshape(-1).tolist(); me.userface = f.reshape(-1).tolist()
                    try:
                        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                    except AttributeError:
                        pass
                    mesh_cache[mk] = mname
                kw.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_cache[mk])
            elif t == int(G.HFIELD):
                hf = i.src
                data = np.asarray(hf.data, np.float64).reshape(int(hf.nrow), int(hf.ncol))
                z = float(hf.min_z) + data * (float(hf.max_z) - float(hf.min_z))
                zlo, zhi = float(z.min()), float(z.max())
                norm = (z - zlo) / (zhi - zlo) if zhi > zlo else np.zeros_like(z)
                hname = f"hfield_{k}"
                spec.add_hfield(name=hname, size=[float(hf.hx), float(hf.hy), max(zhi - zlo, 1e-6), 0.01],
                                nrow=int(hf.nrow), ncol=int(hf.ncol), userdata=norm.reshape(-1).tolist())
                kw.update(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hname)
                pre = _tf((0.0, 0.0, zlo))
            else:
                raise AssertionError(t)
            parent.add_geom(**kw)
            extra.append(pre)
        for k, (i, col) in enumerate(site_tmpl):
            parent = twin_body[i.body] if i.body >= 0 else spec.worldbody
            sname = _label_tail(i.label) or f"site_{i.shape}"
            if sname in snames:
                sname = f"{sname}_{i.shape}"
            snames.append(sname)
            parent.add_site(name=sname, size=[0.01, 0, 0], group=4)
        cam_q = []
        for c in cams:
            cam_q.append(np.asarray(c.quat, float) if c.quat is not None else lookat_quat(c.pos, c.lookat, c.up))
            sc = spec.worldbody.add_camera(name=c.name, pos=[0, 0, 0], quat=[1, 0, 0, 0], fovy=float(c.fovy))
            if c.sensorsize is not None:
                sc.sensor_size = list(c.sensorsize); sc.focal_length = list(c.focal); sc.principal_length = list(c.principal)
                sc.resolution = list(c.resolution or (1, 1))
        for lt in lts:
            spec.worldbody.add_light(pos=[0, 0, 0], dir=list(np.asarray(lt.dir, float) / np.linalg.norm(lt.dir)),
                                     type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if lt.directional else mujoco.mjtLightType.mjLIGHT_SPOT,
                                     castshadow=bool(lt.castshadow), diffuse=list(lt.diffuse), specular=list(lt.specular),
                                     ambient=list(lt.ambient))
        if spec_hook is not None:
            spec_hook(spec)
        tm = spec.compile()
        self.m = tm
        oid = lambda kind, nm: mujoco.mj_name2id(tm, kind, nm)
        # per-geom frames: Newton shape frame x pre x twin's compiled geom frame (mesh recentring). MuJoCo orders the
        # compiled geoms by body, so the template is matched by name.
        sbody = _shape_body(self.newton)
        geom_slots = [None] * tm.ngeom
        self.geom_shape = np.full((n, tm.ngeom), -1, np.int32)
        for (i, col), pre, gname in zip(template, extra, gnames):
            g = oid(mujoco.mjtObj.mjOBJ_GEOM, gname)
            C = _tf(tm.geom_pos[g], _wxyz_to_xyzw(tm.geom_quat[g]))
            body = np.empty(n, np.int32); xf = np.empty((n, 7))
            for e in range(n):
                s_ = int(col[e])
                if s_ < 0:
                    body[e] = HIDDEN_BODY; xf[e] = _tf()
                    continue
                self.geom_shape[e, g] = s_
                body[e] = max(int(sbody[s_]), -1); xf[e] = _tf_mul(_tf_mul(tf_np[s_], pre), C)
            geom_slots[g] = _Slot(body, xf)
        for g in range(tm.ngeom):          # geoms added by spec_hook: fixed in the env frame (or on their twin body)
            if geom_slots[g] is None:
                geom_slots[g] = self._extra_slot(tm, int(tm.geom_bodyid[g]), _tf(tm.geom_pos[g], _wxyz_to_xyzw(tm.geom_quat[g])), body_ids)
        self._geoms = geom_slots
        # bodies (annotators): world + twin bodies (added in body_ids order, children of the world)
        self._bodies = [self._static_slot(_tf())] + [self._body_slot(b, _tf()) for b in body_ids]
        self._bodies += [self._static_slot(_tf())] * (tm.nbody - len(self._bodies))
        # sites
        site_slots = [None] * tm.nsite
        for (i, col), sname in zip(site_tmpl, snames):
            k = oid(mujoco.mjtObj.mjOBJ_SITE, sname)
            body = np.array([int(sbody[int(col[e])]) for e in range(n)], np.int32)
            xf = np.stack([tf_np[int(col[e])] for e in range(n)])
            site_slots[k] = _Slot(body, xf)
        for k in range(tm.nsite):
            if site_slots[k] is None:
                site_slots[k] = self._extra_slot(tm, int(tm.site_bodyid[k]), _tf(tm.site_pos[k], _wxyz_to_xyzw(tm.site_quat[k])), body_ids)
        self._sites = site_slots
        # cameras / lights (the bridge's own come first in the compiled order: they are added before spec_hook)
        self._cams = [self._mount_slot(c.body, _tf(c.pos, _wxyz_to_xyzw(q))) for c, q in zip(cams, cam_q)]
        self._cams += [self._extra_slot(tm, int(tm.cam_bodyid[c]), _tf(tm.cam_pos[c], _wxyz_to_xyzw(tm.cam_quat[c])), body_ids)
                       for c in range(len(cams), tm.ncam)]
        self._lights = [self._mount_slot(lt.body, _tf(lt.pos)) for lt in lts]
        self._light_local = [(np.zeros(3), np.asarray(lt.dir, float) / np.linalg.norm(lt.dir)) for lt in lts]
        for i in range(len(lts), tm.nlight):
            self._lights.append(self._extra_slot(tm, int(tm.light_bodyid[i]), _tf(), body_ids))
            self._light_local.append((np.asarray(tm.light_pos[i], float), np.asarray(tm.light_dir[i], float)))
        self.twin_body_newton = [-1] + list(body_ids)       # twin body id -> Newton body index (world 0 / template)
        self._set_stat(tm)

    # -- twin from the source MJCF ------------------------------------------------------------------------------------

    def _build_from_mjcf(self, mj: mujoco.MjModel, cameras, lights):
        if cameras or lights:
            raise NotImplementedError("extra cameras/lights on an MJCF twin: add them to the MJCF")
        n = self.n
        self.m = mj
        names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(mj.nbody)]
        # Newton body per (env, MuJoCo body), by name, then by order for unnamed bodies
        match = np.full((n, mj.nbody), -1, np.int64)
        for e in range(n):
            wb = list(self._world_bodies(e)) + (list(np.flatnonzero(self._body_world < 0)) if self._nw else [])
            tails = {}
            for b in wb:
                tails.setdefault(_label_tail(self._labels[b]), b)
            for b in range(1, mj.nbody):
                if names[b] and names[b] in tails:
                    match[e, b] = tails[names[b]]
            unnamed = [b for b in range(1, mj.nbody) if match[e, b] < 0]
            if unnamed and len(wb) == mj.nbody - 1:
                for b in unnamed:
                    match[e, b] = wb[b - 1]
        # chain: MuJoCo body -> (nearest matched ancestor, transform ancestor -> body)
        chain = {}
        for b in range(mj.nbody):
            X = _tf(); a = b
            while a > 0 and match[0, a] < 0:
                if mj.body_jntnum[a] > 0:
                    raise ValueError(f"MuJoCo body {names[a]!r} has joints but no Newton body of that name")
                X = _tf_mul(_tf(mj.body_pos[a], _wxyz_to_xyzw(mj.body_quat[a])), X)
                a = int(mj.body_parentid[a])
            chain[b] = (a, X)
        d0 = mujoco.MjData(mj)
        mujoco.mj_kinematics(mj, d0)

        def slot(bodyid, local):
            a, X = chain[int(bodyid)]
            loc = _tf_mul(X, local)
            if a == 0:
                return self._static_slot(loc)
            body = match[:, a].astype(np.int32)
            if (body < 0).any():
                raise ValueError(f"MuJoCo body {names[a]!r} is missing in some Newton worlds")
            return _Slot(body, np.repeat(loc[None], n, 0))

        self._geoms = [slot(mj.geom_bodyid[g], _tf(mj.geom_pos[g], _wxyz_to_xyzw(mj.geom_quat[g]))) for g in range(mj.ngeom)]
        self._bodies = [slot(b, _tf()) for b in range(mj.nbody)]
        self._sites = [slot(mj.site_bodyid[s], _tf(mj.site_pos[s], _wxyz_to_xyzw(mj.site_quat[s]))) for s in range(mj.nsite)]
        if mj.ncam and (np.asarray(mj.cam_mode) != 0).any():
            warnings.warn("MJCF camera modes other than 'fixed' are drawn as fixed in their body frame")
        self._cams = [slot(mj.cam_bodyid[c], _tf(mj.cam_pos[c], _wxyz_to_xyzw(mj.cam_quat[c]))) for c in range(mj.ncam)]
        self._lights = [slot(mj.light_bodyid[i], _tf()) for i in range(mj.nlight)]
        self._light_local = [(np.asarray(mj.light_pos[i], float), np.asarray(mj.light_dir[i], float)) for i in range(mj.nlight)]
        self.geom_shape = np.full((n, mj.ngeom), -1, np.int32)
        self.twin_body_newton = [int(match[0, b]) if b else -1 for b in range(mj.nbody)]

    # -- particles: cloth / soft-body surfaces and free particles -----------------------------------------------------

    def _build_particles(self, enabled: bool):
        """Per-env index tables into ``particle_q``: the vertices of the surface triangles (``tri_indices``: cloth and
        the boundary of soft bodies) as one deforming mesh for the renderers' dynamic-mesh path
        (``extras.DeformableMesh``), and the particles in no triangle as a particle set for the particle renderer
        (``set_particles``); Isaac Lab 3.0's Newton renderer splits them the same way. World ``e`` and the global
        (world -1) particles go to env ``e``; worlds with different topologies are concatenated and hidden per env."""
        m, n = self.newton, self.n
        self.deformables, self.sources = [], {}
        self.particle_radius, self.particle_color = 0.0, (0.7, 0.6, 0.4)
        self._cloth_idx = self._free_idx = None
        P = int(getattr(m, "particle_count", 0) or 0)
        if not enabled or P == 0:
            return
        pw = m.particle_world.numpy() if getattr(m, "particle_world", None) is not None else np.full(P, -1)
        if not self._nw:
            pw = np.full(P, -1)
        T = int(getattr(m, "tri_count", 0) or 0)
        tris = m.tri_indices.numpy().reshape(-1, 3).astype(np.int64) if T else np.zeros((0, 3), np.int64)
        in_tri = np.zeros(P, bool); in_tri[tris.reshape(-1)] = True
        tw = pw[tris[:, 0]] if T else np.zeros(0, int)

        def layout(groups):
            """groups[e]: (global particle ids, local triangles or None) -> (n, V) table, template triangles."""
            sig = [(len(g[0]), None if g[1] is None else g[1].tobytes()) for g in groups]
            if all(x == sig[0] for x in sig):
                V = len(groups[0][0])
                idx = np.stack([g[0] for g in groups]).astype(np.int32) if V else np.zeros((n, 0), np.int32)
                return idx, groups[0][1]
            warnings.warn("Newton worlds have different particle / cloth layouts: each env hides the other worlds' ones")
            offs = np.cumsum([0] + [len(g[0]) for g in groups])
            idx = np.full((n, offs[-1]), -1, np.int32)
            ftris = []
            for e, g in enumerate(groups):
                idx[e, offs[e]:offs[e + 1]] = g[0]
                if g[1] is not None:
                    ftris.append(g[1] + offs[e])
            return idx, (np.concatenate(ftris) if ftris else None)

        if T:
            groups = []
            for e in range(n):
                sel = (tw == e) | (tw < 0)
                tt = tris[sel]
                verts, local = np.unique(tt.reshape(-1), return_inverse=True)
                groups.append((verts, local.reshape(-1, 3)))
            idx, ltris = layout(groups)
            col = (0.7, 0.5, 0.3)                       # Newton viewer's default triangle colour
            tc = getattr(m, "tri_color", None)
            if tc is not None:
                col = tuple(float(c) for c in np.asarray(tc.numpy()).reshape(-1, 3)[0])
            self._cloth_idx = idx
            self.deformables.append(_deformable_mesh("newton_triangles", ltris, idx.shape[1], (*col, 1.0)))
        free = np.flatnonzero(~in_tri)
        if len(free):
            groups = [(free[(pw[free] == e) | (pw[free] < 0)], None) for e in range(n)]
            idx, _ = layout(groups)
            self._free_idx = idx
            r = m.particle_radius.numpy()[free] if hasattr(m.particle_radius, "numpy") else np.full(len(free), float(m.particle_radius))
            self.particle_radius = float(np.median(r))
            if np.ptp(r) > 1e-6 * max(abs(self.particle_radius), 1e-9):
                warnings.warn("Newton particles have different radii: the particle renderer draws them at the median")

    # -- slot helpers -------------------------------------------------------------------------------------------------

    def _static_slot(self, local) -> _Slot:
        """A frame fixed in each env's frame: stored in the Newton world (origin + local) like static shapes."""
        xf = np.repeat(np.asarray(local, float)[None], self.n, 0).copy()
        xf[:, :3] += self.origins
        return _Slot(np.full(self.n, -1, np.int32), xf)

    def _extra_slot(self, tm, twin_body: int, local, body_ids) -> _Slot:
        """A frame added through ``spec_hook``: on its twin body (mapped to the Newton body per env) or fixed in the env."""
        if twin_body > 0 and twin_body - 1 < len(body_ids):
            return self._body_slot(body_ids[twin_body - 1], local)
        return self._static_slot(local)

    def _body_slot(self, b0: int, local) -> _Slot:
        """A frame on template body ``b0`` (world 0's index), mapped to the same in-world body in every env."""
        n = self.n
        body = np.empty(n, np.int32)
        if not self._nw or self._body_world[b0] < 0:
            body[:] = b0
        else:
            k = int(np.searchsorted(self._world_bodies(int(self._body_world[b0])), b0))
            for e in range(n):
                wb = self._world_bodies(e)
                body[e] = int(wb[k]) if k < len(wb) else HIDDEN_BODY
        return _Slot(body, np.repeat(np.asarray(local, float)[None], n, 0))

    def _mount_slot(self, spec, local) -> _Slot:
        if spec is None:
            return self._static_slot(local)
        n = self.n
        body = np.array([self._resolve_body(spec, e) for e in range(n)], np.int32)
        return _Slot(body, np.repeat(np.asarray(local, float)[None], n, 0))

    def _set_stat(self, tm):
        """MuJoCo's compiled ``stat`` of the twin sees every geom at its body origin; the renderers use it for the
        shadow-map bounds and the ray tracer's env stride: recompute it from the initial state of env 0."""
        bq = self._body_q0
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g, sl in enumerate(self._geoms):
            b = int(sl.body[0])
            if b == HIDDEN_BODY or int(tm.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            X = _tf_mul(bq[b], sl.xf[0]) if b >= 0 else sl.xf[0]
            p = X[:3] - self.origins[0]
            r = float(tm.geom_rbound[g]) if tm.geom_rbound[g] > 0 else 0.1
            if int(tm.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_HFIELD):
                hs = tm.hfield_size[int(tm.geom_dataid[g])]
                r = float(np.linalg.norm(hs[:3]))
            lo = np.minimum(lo, p - r); hi = np.maximum(hi, p + r)
        if np.isfinite(lo).all():
            tm.stat.center[:] = 0.5 * (lo + hi)
            tm.stat.extent = max(float(0.5 * (hi - lo).max()) * 2.0, 0.1)

    # -- device buffers -----------------------------------------------------------------------------------------------

    def _alloc(self):
        n, m, dev = self.n, self.m, self.device
        d = self.d = _Data()

        def pose(J):
            return wp.zeros((n, max(J, 1)), dtype=wp.vec3, device=dev), wp.zeros((n, max(J, 1)), dtype=wp.mat33, device=dev)

        d.geom_xpos, d.geom_xmat = pose(m.ngeom)
        d.xpos, d.xmat = pose(m.nbody)
        d.site_xpos, d.site_xmat = pose(m.nsite)
        d.cam_xpos, d.cam_xmat = pose(m.ncam)
        d.light_xpos, _ = pose(m.nlight)
        d.light_xdir = wp.zeros((n, max(m.nlight, 1)), dtype=wp.vec3, device=dev)

        def table(slots):
            if not slots:
                return None
            J = len(slots)
            body = np.stack([s.body for s in slots], 1).reshape(-1).astype(np.int32)     # (n, J)
            xf = np.stack([s.xf for s in slots], 1).reshape(-1, 7).astype(np.float32)
            return (J, wp.array(body, dtype=wp.int32, device=dev), wp.array(xf, dtype=wp.transform, device=dev))

        self._tab = {k: table(getattr(self, "_" + k)) for k in ("geoms", "bodies", "sites", "cams")}
        self._out = {"geoms": (d.geom_xpos, d.geom_xmat), "bodies": (d.xpos, d.xmat), "sites": (d.site_xpos, d.site_xmat),
                     "cams": (d.cam_xpos, d.cam_xmat)}
        self._ltab = None
        if self._lights:
            L = len(self._lights)
            lb = np.stack([s.body for s in self._lights], 1).reshape(-1).astype(np.int32)
            lx = np.stack([s.xf for s in self._lights], 1).reshape(-1, 7).astype(np.float32)
            lp = np.array([p for p, _ in self._light_local], np.float32); ld = np.array([q for _, q in self._light_local], np.float32)
            self._ltab = (L, wp.array(lb, dtype=wp.int32, device=dev), wp.array(lx, dtype=wp.transform, device=dev),
                          wp.array(lp, dtype=wp.vec3, device=dev), wp.array(ld, dtype=wp.vec3, device=dev))
        self._origin = wp.array(self.origins.astype(np.float32), dtype=wp.vec3, device=dev)
        self._ptab = []
        for name, idx in (("newton_triangles", self._cloth_idx), ("particles", self._free_idx)):
            if idx is None or idx.shape[1] == 0:
                continue
            out = wp.zeros((n, idx.shape[1]), dtype=wp.vec3, device=dev)
            self._ptab.append((idx.shape[1], wp.array(idx.reshape(-1), dtype=wp.int32, device=dev), out))
            if name == "particles":
                d.particle_pos = out
            else:
                d.cloth_q = out
                self.sources[name] = out
        self._dummy_bq = wp.zeros(1, dtype=wp.transform, device=dev)
        self._t = None
        # the twin's model fields some renderer paths read (per-world geom_size for sized slots: none in a twin)
        self.geom_shape_wp = wp.array(self.geom_shape, dtype=wp.int32, device=dev)
        wp.synchronize_device(dev)

    @property
    def t(self):
        """Zero-copy MPS tensor views of the pose arrays (``BatchSim.t``; the annotators read them)."""
        if self._t is None:
            from metalsim.physics.batch import _TensorViews
            self._t = _TensorViews(self.d, self.n)
        return self._t

    # -- per frame ----------------------------------------------------------------------------------------------------

    def launch(self, state) -> None:
        """Enqueue the pose kernels for ``state`` on the current Warp stream (no signal); can be recorded into a graph."""
        bq = state.body_q if (state is not None and state.body_q is not None and state.body_q.shape[0]) else self._dummy_bq
        with wp.ScopedDevice(self.device):
            for k, tab in self._tab.items():
                if tab is None:
                    continue
                J, fb, fx = tab
                pos, mat = self._out[k]
                wp.launch(_frames, dim=(self.n, J), inputs=[bq, fb, fx, self._origin, J], outputs=[pos, mat])
            if self._ptab and state is not None and getattr(state, "particle_q", None) is not None:
                for V, idx, out in self._ptab:
                    wp.launch(_gather_points, dim=(self.n, V), inputs=[state.particle_q, self.newton.particle_flags, idx,
                                                                       self._origin, V], outputs=[out])
            if self._ltab is not None:
                L, lb, lx, lp, ld = self._ltab
                wp.launch(_lights, dim=(self.n, L), inputs=[bq, lb, lx, lp, ld, self._origin, L],
                          outputs=[self.d.light_xpos, self.d.light_xdir])

    def update(self, state, graph: bool = True) -> int:
        """Write the twin poses from ``state`` and signal ``self.event``; returns the value the renderer waits for.
        ``graph``: replay a captured graph per ``State`` object (Newton's ping-pong states each get one)."""
        if graph:
            key = (id(state), state.body_q.ptr if state is not None and state.body_q is not None else 0,
                   state.particle_q.ptr if state is not None and getattr(state, "particle_q", None) is not None else 0)
            g = self._graphs.get(key)
            if g is None:
                with wp.ScopedDevice(self.device):
                    with wp.ScopedCapture() as cap:
                        self.launch(state)
                g = self._graphs[key] = cap.graph
            with wp.ScopedDevice(self.device):
                wp.capture_launch(g)
        else:
            self.launch(state)
        return self.signal()

    def signal(self) -> int:
        if self.event is None:
            return 0
        v = self.event.next_value()
        self._wm.signal(self.event, v, str(self.device))
        return v

    # BatchSim-compatible ordering
    def wait(self, event, value: int) -> None:
        self._wm.wait(event, value, str(self.device))

    def after(self, value: int) -> None:
        from metalsim.interop import torch_bridge as tb
        tb.wait_event(self.event, value)

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    # -- deformables / particles in a renderer -------------------------------------------------------------------------

    def renderer_kwargs(self) -> dict:
        """Constructor arguments a renderer needs for this source beyond the model: the cloth / soft-body surfaces as
        ``deformables`` (the dynamic-mesh path); pass ``sources=src.sources`` to ``render`` / ``trace``."""
        return {"deformables": list(self.deformables)} if self.deformables else {}

    def attach_particles(self, renderer, **kw):
        """Draw the free particles (``particle_q`` not in any triangle) with the renderer's particle path
        (``set_particles``), bound to this source's per-env buffer (refreshed by ``update``; no host copy). Defaults:
        one round grain per particle at the particle radius, the Newton viewer's particle colour. ``kw`` overrides
        ``set_particles`` arguments (e.g. ``material``, ``grains_per_particle``)."""
        if getattr(self.d, "particle_pos", None) is None:
            return None
        if not hasattr(renderer, "set_particles"):
            raise NotImplementedError("this renderer has no particle path (metalsim.render.particles)")
        args = dict(radius=self.particle_radius, grains_per_particle=1, grain_radius=self.particle_radius,
                    radius_jitter=0.0, grain_shape={"irregularity": 0.0, "elongation": 0.0})
        if "material" not in kw:
            from metalsim.render.particles import GrainMaterial
            args["material"] = GrainMaterial.uniform(self.particle_color)
        args.update(kw)
        return renderer.set_particles(self.d.particle_pos, **args)

    # -- host path (tests, tools) -------------------------------------------------------------------------------------

    def host_datas(self):
        """The current pose buffers as a list of N MjData-like objects for ``render_host`` / ``trace_host``
        (synchronizes)."""
        wp.synchronize_device(self.device)
        f = {k: getattr(self.d, k).numpy() for k in ("geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "light_xpos",
                                                      "light_xdir", "site_xpos", "site_xmat", "xpos", "xmat")}
        out = []
        for e in range(self.n):
            o = _Data()
            for k, a in f.items():
                setattr(o, k, a[e].reshape(a.shape[1], -1).astype(np.float64))
            out.append(o)
        return out

    # -- segmentation / colours ---------------------------------------------------------------------------------------

    def shape_of_geom(self) -> np.ndarray:
        """(N, ngeom) Newton shape index of each twin geom per env (-1: none; MJCF twins: -1)."""
        return self.geom_shape

    def per_env_colors(self, renderer) -> np.ndarray:
        """(N, G, 4) colour multipliers for ``renderer.set_colors`` so each env shows its own worlds' ``shape_color``
        where worlds differ (the twin's material is world 0's)."""
        if self.kind != "newton":
            return np.ones((self.n, renderer.G, 4), np.float32)
        col = self.newton.shape_color.numpy() if getattr(self.newton, "shape_color", None) is not None else None
        out = np.ones((self.n, renderer.G, 4), np.float32)
        if col is None:
            return out
        for slot, g in enumerate(renderer.tables.geoms):
            base = self.m.geom_rgba[g][:3]
            for e in range(self.n):
                s = self.geom_shape[e, g]
                if s >= 0:
                    out[e, slot, :3] = np.where(base > 1e-6, col[s][:3] / np.maximum(base, 1e-6), 1.0)
        return out


# -- helpers ----------------------------------------------------------------------------------------------------------


@dataclass
class _DeformableSpec:
    """Stand-in for ``metalsim.render.extras.DeformableMesh`` before the dynamic-mesh path is installed (same fields)."""
    name: str
    tris: np.ndarray
    nvert: int
    rgba: tuple = (0.5, 0.5, 0.5, 1.0)
    radius: float = 0.0
    uv: np.ndarray | None = None
    body: int = 0
    material: dict | None = None


def _deformable_mesh(name, tris, nvert, rgba):
    try:
        from metalsim.render.extras import DeformableMesh
    except ImportError:
        DeformableMesh = _DeformableSpec
    return DeformableMesh(name=name, tris=np.asarray(tris, np.int64).reshape(-1, 3), nvert=int(nvert), rgba=tuple(rgba))


def mujoco_qpos_to_joint_q(mj: mujoco.MjModel, qpos, model, joint_q: np.ndarray | None = None,
                           offsets=None) -> np.ndarray:
    """Newton ``joint_q`` (all worlds) from MuJoCo ``qpos`` per world ((N, nq) or (nq,) for world 0), for a Newton
    model imported from the same MJCF: joints matched by name per world; free / ball quaternions wxyz -> xyzw. Joints
    absent from ``mj`` keep ``joint_q`` (default ``model.joint_q``). ``offsets`` (N, 3): world origins added to the
    free joints' positions (worlds replicated with a spacing)."""
    qpos = np.atleast_2d(np.asarray(qpos, float))
    jq = (model.joint_q.numpy() if joint_q is None else joint_q).astype(np.float64).copy()
    labels = [_label_tail(l) for l in (getattr(model, "joint_label", None) or getattr(model, "joint_key", None))]
    jw = model.joint_world.numpy(); qs = model.joint_q_start.numpy()
    nw = int(getattr(model, "world_count", 0) or 0)
    for e in range(len(qpos)):
        for j in range(mj.njnt):
            nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
            hits = [k for k in range(len(labels)) if labels[k] == nm and (jw[k] == e or not nw or jw[k] < 0)]
            if not hits:
                raise ValueError(f"no Newton joint {nm!r} in world {e}")
            q0, a, t = int(qs[hits[0]]), int(mj.jnt_qposadr[j]), int(mj.jnt_type[j])
            if t == mujoco.mjtJoint.mjJNT_FREE:
                jq[q0:q0 + 3] = qpos[e, a:a + 3] + (0.0 if offsets is None else np.asarray(offsets, float)[e])
                jq[q0 + 3:q0 + 7] = _wxyz_to_xyzw(qpos[e, a + 3:a + 7])
            elif t == mujoco.mjtJoint.mjJNT_BALL:
                jq[q0:q0 + 4] = _wxyz_to_xyzw(qpos[e, a:a + 4])
            else:
                jq[q0] = qpos[e, a]
    return jq.astype(np.float32)


def joint_q_to_mujoco_qpos(mj: mujoco.MjModel, model, joint_q=None) -> np.ndarray:
    """Inverse of ``mujoco_qpos_to_joint_q``: (N, nq) MuJoCo ``qpos`` per world from Newton ``joint_q``."""
    jq = (model.joint_q if joint_q is None else joint_q)
    jq = jq.numpy() if hasattr(jq, "numpy") else np.asarray(jq)
    labels = [_label_tail(l) for l in (getattr(model, "joint_label", None) or getattr(model, "joint_key", None))]
    jw = model.joint_world.numpy(); qs = model.joint_q_start.numpy()
    nw = int(getattr(model, "world_count", 0) or 0)
    out = np.tile(np.asarray(mj.qpos0, float), (max(nw, 1), 1))
    for e in range(len(out)):
        for j in range(mj.njnt):
            nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
            hits = [k for k in range(len(labels)) if labels[k] == nm and (jw[k] == e or not nw or jw[k] < 0)]
            if not hits:
                raise ValueError(f"no Newton joint {nm!r} in world {e}")
            q0, a, t = int(qs[hits[0]]), int(mj.jnt_qposadr[j]), int(mj.jnt_type[j])
            if t == mujoco.mjtJoint.mjJNT_FREE:
                out[e, a:a + 3] = jq[q0:q0 + 3]; out[e, a + 3:a + 7] = _xyzw_to_wxyz(jq[q0 + 3:q0 + 7])
            elif t == mujoco.mjtJoint.mjJNT_BALL:
                out[e, a:a + 4] = _xyzw_to_wxyz(jq[q0:q0 + 4])
            else:
                out[e, a] = jq[q0]
    return out


def newton_geo():
    import newton
    return newton.GeoType


def newton_flags():
    import newton
    return newton.ShapeFlags


def _shape_body(model):
    sb = getattr(model, "_bridge_shape_body", None)
    if sb is None:
        sb = model.shape_body.numpy()
        try:
            model._bridge_shape_body = sb
        except AttributeError:
            pass
    return sb


def _geometry_hash(src) -> tuple:
    import hashlib
    h = hashlib.blake2s(digest_size=8)
    for name in ("vertices", "indices", "data"):
        a = getattr(src, name, None)
        if a is not None:
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    for name in ("hx", "hy", "min_z", "max_z", "nrow", "ncol"):
        a = getattr(src, name, None)
        if a is not None:
            h.update(str(float(a)).encode())
    return (type(src).__name__, h.hexdigest())


def _mesh_arrays(info: _ShapeInfo, G):
    """Vertices (scale baked in) and faces of a mesh / convex mesh / cone shape; mirrored scales flip the winding."""
    sc = np.asarray(info.scale, float)
    if info.gtype == int(G.CONE):
        import newton
        mesh = newton.Mesh.create_cone(float(sc[0]), float(sc[1]), up_axis=newton.Axis.Z, compute_inertia=False)
        v = np.asarray(mesh.vertices, float); f = np.asarray(mesh.indices, np.int64).reshape(-1, 3)
        return v, f
    src = info.src
    v = np.asarray(src.vertices, float) * sc[:3]
    f = np.asarray(src.indices, np.int64).reshape(-1, 3)
    if np.prod(sc[:3]) < 0:
        f = f[:, [0, 2, 1]]
    return v, f
