"""MuJoCo geoms as MPM colliders: a Newton model of the bodies and geoms the granular material touches.

The MPM solver (Newton 1.5.2's ``SolverImplicitMPM``) rasterizes colliders from a Newton ``Model``: one triangle mesh
per body (its collision shapes merged in the body frame, Newton's own ``Mesh.create_*`` tessellations), moved every
step by ``state.body_q``. MetalSim's robot lives in a MuJoCo model, so this module builds that Newton model from the
``MjModel``: per world, one Newton body per selected MuJoCo body (body frame = MuJoCo's ``xpos`` / ``xquat`` frame,
centre of mass ``body_ipos``, mass and inertia as MuJoCo's), one Newton shape per selected geom with the geom's
pose in its body, type and size, and the static geoms of the world body as static shapes. ``body_map`` gives, for
each Newton body ``w * nb + k``, the MuJoCo body it mirrors.

Geom types: plane (``size`` 0 = Newton's 1000 m plane), sphere, capsule, cylinder, box, ellipsoid (as a mesh: Newton
1.5.2's collider meshing has no ellipsoid) and mesh. Height fields and SDF geoms raise.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

_PLANE, _HFIELD, _SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER, _BOX, _MESH, _SDF = (
    int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_SDF))


def _xform(pos, quat_wxyz) -> wp.transform:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return wp.transform(wp.vec3(*(float(v) for v in pos)), wp.quat(x, y, z, w))


def default_geoms(model: mujoco.MjModel) -> np.ndarray:
    """Geoms that collide in MuJoCo (``contype`` or ``conaffinity`` non-zero)."""
    return np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))


def _ellipsoid_mesh(newton, r):
    m = newton.Mesh.create_sphere(1.0, compute_normals=False, compute_uvs=False, compute_inertia=False)
    v = np.asarray(m.vertices, np.float32) * np.asarray(r, np.float32)
    return newton.Mesh(v, np.asarray(m.indices), compute_inertia=False)


def _geom_mesh(newton, model: mujoco.MjModel, g: int):
    d = int(model.geom_dataid[g])
    va, vn = int(model.mesh_vertadr[d]), int(model.mesh_vertnum[d])
    fa, fn = int(model.mesh_faceadr[d]), int(model.mesh_facenum[d])
    v = np.asarray(model.mesh_vert[va:va + vn], np.float32)
    f = np.asarray(model.mesh_face[fa:fa + fn], np.int32).reshape(-1)
    return newton.Mesh(v, f, compute_inertia=False)


def add_geom(newton, builder, model: mujoco.MjModel, g: int, body: int, cfg, xf: wp.transform | None = None) -> int:
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
    if t == _ELLIPSOID:
        return builder.add_shape_mesh(body, xform=xf, mesh=_ellipsoid_mesh(newton, s[:3]), cfg=cfg)
    if t == _MESH:
        return builder.add_shape_mesh(body, xform=xf, mesh=_geom_mesh(newton, model, g), cfg=cfg)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
    raise NotImplementedError(f"geom {name!r}: type {mujoco.mjtGeom(t).name} is not supported as an MPM collider")


@dataclass
class ColliderModel:
    """The Newton collider model and how it maps onto the MuJoCo model."""

    model: object                 # newton.Model: world_count worlds, nb bodies each
    body_map: np.ndarray          # (nb,) MuJoCo body id of Newton body k in every world (Newton body = w * nb + k)
    geoms: np.ndarray             # MuJoCo geom ids used
    static_geoms: np.ndarray      # of those, the world body's (static colliders)
    nb: int


def build(model: mujoco.MjModel, num_worlds: int, geoms=None, friction: float | None = None,
          margin: float = 0.0, device=None) -> ColliderModel:
    """Newton model mirroring ``geoms`` (default :func:`default_geoms`) of ``model`` in ``num_worlds`` worlds.

    ``friction``: sand-collider Coulomb friction for every shape (default: each geom's sliding friction).
    ``margin``: shape margin (Newton's collider thickness; 0 = the MPM solver's default)."""
    import newton  # noqa: PLC0415

    geoms = default_geoms(model) if geoms is None else np.asarray(sorted(int(g) for g in geoms))
    gbody = model.geom_bodyid[geoms]
    # a geom on a body welded to the world (no dofs on the path) is static too
    static_body = np.array([model.body_weldid[b] == 0 for b in range(model.nbody)])
    static = geoms[static_body[gbody]]
    moving = geoms[~static_body[gbody]]
    bodies = np.array(sorted({int(b) for b in model.geom_bodyid[moving]}), dtype=int)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    one = newton.ModelBuilder()
    nb_index = {}
    for b in bodies:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.body_iquat[b])
        R = R.reshape(3, 3)
        inertia = R @ np.diag(model.body_inertia[b]) @ R.T
        nb_index[int(b)] = one.add_body(xform=_xform(data.xpos[b], data.xquat[b]), com=wp.vec3(*model.body_ipos[b]),
                                        inertia=wp.mat33(*inertia.astype(np.float32).ravel()),
                                        mass=float(model.body_mass[b]), lock_inertia=True,
                                        label=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"body{b}")
    for g in geoms:
        b = int(model.geom_bodyid[g])
        mu = float(model.geom_friction[g, 0]) if friction is None else float(friction)
        cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0, margin=float(margin), has_shape_collision=False,
                                              has_particle_collision=True)
        if g in set(static.tolist()):
            # static geoms in world coordinates (MuJoCo world frame = body frame of the welded root)
            pose = _xform(data.geom_xpos[g], _mat2quat(data.geom_xmat[g]))
            shape = add_geom(newton, one, model, int(g), -1, cfg, pose)
        else:
            shape = add_geom(newton, one, model, int(g), nb_index[b], cfg)
        del shape
    builder = newton.ModelBuilder()
    builder.replicate(one, num_worlds)
    with wp.ScopedDevice(device):
        nm = builder.finalize()
    return ColliderModel(model=nm, body_map=bodies, geoms=geoms, static_geoms=static, nb=len(bodies))


def _mat2quat(xmat) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, np.float64).ravel())
    return q

