"""MuJoCo bodies and geoms as MPM colliders: Newton proxy bodies and shapes mirroring an ``MjModel``.

The MPM solver (Newton 1.5.2's ``SolverImplicitMPM``) rasterizes colliders from its Newton ``Model``: one triangle
mesh per body (the body's shapes merged in the body frame with Newton's own ``Mesh.create_*`` tessellations), moved
every step by ``state.body_q``, plus the static shapes. Isaac Lab's coupler puts the robot bodies it couples into the
MPM model as *proxy bodies* (``BodyFlags.PROXY``). MetalSim's robot is an ``MjModel`` simulated by ``BatchSim``, so
this module adds, to a Newton ``ModelBuilder`` world, one proxy body per selected MuJoCo body (body frame = MuJoCo's
``xpos`` / ``xquat`` frame, centre of mass ``body_ipos``, MuJoCo's mass and inertia) with one shape per selected geom
(the geom's pose in its body, type and size), and the geoms of bodies welded to the world as static shapes.

Geom types: plane (``size`` 0 = Newton's 1000 m plane), sphere, capsule, cylinder, box, ellipsoid (as a mesh: Newton
1.5.2's collider meshing has no ellipsoid) and mesh. Height fields and SDF geoms raise.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

_G = mujoco.mjtGeom
_PLANE, _SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER, _BOX, _MESH = (
    int(_G.mjGEOM_PLANE), int(_G.mjGEOM_SPHERE), int(_G.mjGEOM_CAPSULE), int(_G.mjGEOM_ELLIPSOID),
    int(_G.mjGEOM_CYLINDER), int(_G.mjGEOM_BOX), int(_G.mjGEOM_MESH))


def _xform(pos, quat_wxyz) -> wp.transform:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return wp.transform(wp.vec3(*(float(v) for v in pos)), wp.quat(x, y, z, w))


def _mat2quat(xmat) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, np.float64).ravel())
    return q


def default_geoms(model: mujoco.MjModel) -> np.ndarray:
    """Geoms that collide in MuJoCo (``contype`` or ``conaffinity`` non-zero)."""
    return np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))


def is_static_body(model: mujoco.MjModel, b: int) -> bool:
    """A body that never moves: welded to the world and not a mocap body (nor welded to one)."""
    w = int(model.body_weldid[b])
    return w == 0 and int(model.body_mocapid[b]) < 0


@dataclass
class ColliderLayout:
    """How the proxy bodies of one world map onto the MuJoCo model (identical in every world)."""

    body_map: np.ndarray      # (nb,) MuJoCo body id of proxy body k (Newton body w * nb + k)
    geoms: np.ndarray         # MuJoCo geom ids on proxy bodies
    static_geoms: np.ndarray  # MuJoCo geom ids added as static shapes

    @property
    def nb(self) -> int:
        return len(self.body_map)


def layout(model: mujoco.MjModel, geoms=None, bodies=None) -> ColliderLayout:
    """Select the colliders: ``geoms`` (default :func:`default_geoms`); ``bodies`` restricts the moving ones to those
    MuJoCo body ids (default: every non-static body with a selected geom). Static geoms are always kept."""
    geoms = default_geoms(model) if geoms is None else np.asarray(sorted({int(g) for g in geoms}), dtype=int)
    static = np.array([g for g in geoms if is_static_body(model, int(model.geom_bodyid[g]))], dtype=int)
    moving = np.array([g for g in geoms if not is_static_body(model, int(model.geom_bodyid[g]))], dtype=int)
    if bodies is not None:
        keep = {int(b) for b in bodies}
        bad = [b for b in keep if is_static_body(model, b)]
        if bad:
            raise ValueError(f"bodies {bad} are welded to the world; their geoms are static colliders already")
        moving = np.array([g for g in moving if int(model.geom_bodyid[g]) in keep], dtype=int)
    body_map = np.array(sorted({int(model.geom_bodyid[g]) for g in moving}), dtype=int)
    return ColliderLayout(body_map=body_map, geoms=moving, static_geoms=static)


class GeomMeshes:
    """Newton meshes of mesh / ellipsoid geoms, built once and shared by every world."""

    def __init__(self, newton, model: mujoco.MjModel):
        self.newton, self.model, self._cache = newton, model, {}

    def get(self, g: int):
        if g not in self._cache:
            m, nt = self.model, self.newton
            if int(m.geom_type[g]) == _MESH:
                d = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[d]), int(m.mesh_vertnum[d])
                fa, fn = int(m.mesh_faceadr[d]), int(m.mesh_facenum[d])
                v = np.asarray(m.mesh_vert[va:va + vn], np.float32)
                f = np.asarray(m.mesh_face[fa:fa + fn], np.int32).reshape(-1)
            else:  # ellipsoid: Newton's unit sphere tessellation scaled per axis
                s = nt.Mesh.create_sphere(1.0, compute_normals=False, compute_uvs=False, compute_inertia=False)
                v = np.asarray(s.vertices, np.float32) * np.asarray(m.geom_size[g, :3], np.float32)
                f = np.asarray(s.indices, np.int32)
            self._cache[g] = nt.Mesh(v, f, compute_inertia=False)
        return self._cache[g]


def add_geom(newton, builder, model: mujoco.MjModel, g: int, body: int, cfg, xf: wp.transform | None = None,
             meshes: GeomMeshes | None = None) -> int:
    """Add MuJoCo geom ``g`` to ``builder`` as a shape of Newton body ``body`` (-1: static) at ``xf`` (default: the
    geom's pose in its body); returns the shape id."""
    t = int(model.geom_type[g])
    s = model.geom_size[g]
    if xf is None:
        xf = _xform(model.geom_pos[g], model.geom_quat[g])
    if t == _PLANE:
        # MuJoCo: half-extents, 0 = infinite; Newton's MPM collider meshes a non-positive extent as 1000 m
        return builder.add_shape_plane(xform=xf, width=2.0 * float(s[0]) if s[0] > 0 else 0.0,
                                       length=2.0 * float(s[1]) if s[1] > 0 else 0.0, body=body, cfg=cfg)
    if t == _SPHERE:
        return builder.add_shape_sphere(body, xform=xf, radius=float(s[0]), cfg=cfg)
    if t == _CAPSULE:
        return builder.add_shape_capsule(body, xform=xf, radius=float(s[0]), half_height=float(s[1]), cfg=cfg)
    if t == _CYLINDER:
        return builder.add_shape_cylinder(body, xform=xf, radius=float(s[0]), half_height=float(s[1]), cfg=cfg)
    if t == _BOX:
        return builder.add_shape_box(body, xform=xf, hx=float(s[0]), hy=float(s[1]), hz=float(s[2]), cfg=cfg)
    if t in (_ELLIPSOID, _MESH):
        meshes = meshes or GeomMeshes(newton, model)
        return builder.add_shape_mesh(body, xform=xf, mesh=meshes.get(g), cfg=cfg)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
    raise NotImplementedError(f"geom {name!r}: type {_G(t).name} is not supported as an MPM collider")


def add_world_colliders(newton, builder, model: mujoco.MjModel, data: mujoco.MjData, lay: ColliderLayout,
                        friction: float | None = None, margin: float = 0.0,
                        meshes: GeomMeshes | None = None) -> np.ndarray:
    """Add one world's proxy bodies (``lay.body_map`` order, at ``data``'s pose) and static shapes to ``builder``
    (inside ``begin_world`` / ``end_world``). ``friction``: sand-collider Coulomb friction of every shape (default:
    each geom's sliding friction); ``margin``: shape margin (the collider thickness; 0 = none). Returns the Newton
    body ids in ``lay.body_map`` order."""
    meshes = meshes or GeomMeshes(newton, model)

    def cfg(g):
        mu = float(model.geom_friction[g, 0]) if friction is None else float(friction)
        return newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0, margin=float(margin), has_shape_collision=False,
                                               has_particle_collision=True)

    ids = []
    for b in lay.body_map:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.body_iquat[b])
        R = R.reshape(3, 3)
        inertia = R @ np.diag(model.body_inertia[b]) @ R.T
        ids.append(builder.add_body(xform=_xform(data.xpos[b], data.xquat[b]), com=wp.vec3(*model.body_ipos[b]),
                                    inertia=wp.mat33(*inertia.astype(np.float32).ravel()),
                                    mass=float(model.body_mass[b]), lock_inertia=True,
                                    label=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"body{b}"))
    index = dict(zip((int(b) for b in lay.body_map), ids))
    for g in lay.geoms:
        add_geom(newton, builder, model, int(g), index[int(model.geom_bodyid[g])], cfg(g), meshes=meshes)
    for g in lay.static_geoms:
        pose = _xform(data.geom_xpos[g], _mat2quat(data.geom_xmat[g]))
        add_geom(newton, builder, model, int(g), -1, cfg(g), pose, meshes=meshes)
    return np.asarray(ids, dtype=int)
