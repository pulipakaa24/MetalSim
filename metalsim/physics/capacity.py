"""Provable per-world capacity bounds for MuJoCo Warp (contacts and constraint rows) and the overflow guard.

MuJoCo Warp sizes its capacity-dependent launches by ``njmax`` (constraint rows per world) and ``nconmax``
(contacts per world): every Newton iteration launches ``(nworld, njmax)`` grids, the contact-constraint builders
``(nworld * nconmax, ...)`` grids, and on Metal each launch is a full dispatch with a barrier, so a capacity above
what the model can produce is paid on every substep (MuJoCo's own guidance: "set as small as possible while ensuring
the simulation does not exceed these limits"). A capacity that cannot be exceeded changes no result: the rows,
their order and every kernel's arithmetic are the same for any capacity that does not overflow.

``bounds(mjm, wm)`` derives the largest number of contacts and constraint rows any world can hold from the model
alone: the collision pairs MuJoCo Warp actually tests (``Model.nxn_geom_pair_filtered`` after body / parent-child /
contype-conaffinity / explicit exclude filtering, contact pairs only), the largest contact count each narrowphase
routine writes for the pair's geom types (``collision_primitive``: plane-convex 4, box pairs 8, capsule pairs 2, the
rest 1; GJK/EPA convex pairs 1 with MULTICCD disabled, 4 with it: ``collision_gjk.multicontact`` returns its
witness points in a 4 x 3 matrix), the rows per contact from the pair's condim and the cone type, plus one row per limited joint or tendon, the equality rows and the friction-loss
rows. Heightfields, SDFs and flex are refused (their per-pair contact counts are capped by MuJoCo Warp's kernels
with an overflow flag, not bounded by the model): give ``njmax`` / ``nconmax`` explicitly there.

The guard: ``BatchSim.check_overflow()`` raises on any capacity overflow flag (dropped rows or contacts are never
silent); the training loop calls it at every log point.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# largest number of contacts each primitive narrowphase routine writes per pair (mujoco_warp collision_primitive.py
# wrappers: write_contact call count), keyed by (type1, type2) with type1 <= type2 in MuJoCo's geom-type order
G = mujoco.mjtGeom
PRIMITIVE_MAX = {
    (G.mjGEOM_PLANE, G.mjGEOM_SPHERE): 1, (G.mjGEOM_PLANE, G.mjGEOM_CAPSULE): 2, (G.mjGEOM_PLANE, G.mjGEOM_ELLIPSOID): 1,
    (G.mjGEOM_PLANE, G.mjGEOM_CYLINDER): 4, (G.mjGEOM_PLANE, G.mjGEOM_BOX): 8, (G.mjGEOM_PLANE, G.mjGEOM_MESH): 4,
    (G.mjGEOM_SPHERE, G.mjGEOM_SPHERE): 1, (G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE): 1, (G.mjGEOM_SPHERE, G.mjGEOM_CYLINDER): 1,
    (G.mjGEOM_SPHERE, G.mjGEOM_BOX): 1, (G.mjGEOM_CAPSULE, G.mjGEOM_CAPSULE): 2, (G.mjGEOM_CAPSULE, G.mjGEOM_BOX): 2,
}
BOX_BOX_PRIMITIVE_MAX = 8            # box-box takes the primitive routine only when NATIVECCD is disabled
CONVEX_MAX = 1                       # GJK / EPA: one contact per pair when MULTICCD is disabled
CONVEX_MULTICCD_MAX = 4              # multicontact() returns at most 4 witness pairs (mat43) per pair
UNBOUNDED_TYPES = (G.mjGEOM_HFIELD, G.mjGEOM_SDF)
CAPACITY_FLAGS = ("NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD", "EPA_HORIZON", "CONTACT_MATCH")


@dataclass(frozen=True)
class Bounds:
    ncon: int              # contacts per world
    nefc: int              # constraint rows per world
    nlimit: int
    neq_rows: int
    nfriction: int
    pairs: tuple           # (geom1, geom2, max contacts, rows per contact)

    @property
    def njmax(self) -> int:
        """nefc rounded up to MuJoCo Warp's constraint tile (16); at least 16."""
        return max(16, ((self.nefc + 15) // 16) * 16)

    @property
    def nconmax(self) -> int:
        return max(1, self.ncon)


def _rows_per_contact(condim: int, elliptic: bool) -> int:
    if condim == 1:
        return 1
    return condim if elliptic else 2 * (condim - 1)


def bounds(mjm: mujoco.MjModel, wm) -> Bounds:
    """Per-world contact and constraint-row bounds of ``mjm`` as MuJoCo Warp's ``Model`` ``wm`` will collide it.
    Raises ValueError when the model has geoms or elements whose contact count the kernels cap instead of the model
    bounding (heightfields, SDFs, flex)."""
    if mjm.nflex:
        raise ValueError("capacity bounds: flex contacts are capped per pair by the kernels, not bounded by the model")
    if (np.isin(mjm.geom_type, UNBOUNDED_TYPES)).any():
        raise ValueError("capacity bounds: heightfield / SDF pair contact counts are capped by the kernels, not bounded by the model")
    multiccd = not (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    native_ccd = not (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    elliptic = mjm.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    pairs_arr = wm.nxn_geom_pair_filtered.numpy()
    pairid = wm.nxn_pairid_filtered.numpy()
    pairs = []
    ncon = 0
    for (g1, g2), (pid, _) in zip(pairs_arr, pairid):
        if pid <= -2:                       # collision-sensor pair without contacts
            continue
        t1, t2 = int(mjm.geom_type[g1]), int(mjm.geom_type[g2])
        key = (min(t1, t2), max(t1, t2))
        if key in PRIMITIVE_MAX:
            k = PRIMITIVE_MAX[key]
        elif key == (G.mjGEOM_BOX, G.mjGEOM_BOX) and not native_ccd:
            k = BOX_BOX_PRIMITIVE_MAX
        else:
            k = CONVEX_MULTICCD_MAX if multiccd else CONVEX_MAX
        if pid >= 0:
            condim = int(mjm.pair_dim[pid])
        else:
            p1, p2 = int(mjm.geom_priority[g1]), int(mjm.geom_priority[g2])
            c1, c2 = int(mjm.geom_condim[g1]), int(mjm.geom_condim[g2])
            condim = c1 if p1 > p2 else c2 if p2 > p1 else max(c1, c2)
        pairs.append((int(g1), int(g2), k, _rows_per_contact(condim, elliptic)))
        ncon += k
    nlimit = int(((mjm.jnt_limited != 0) & np.isin(mjm.jnt_type, (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE,
                                                                    mujoco.mjtJoint.mjJNT_BALL))).sum())
    nlimit += int((mjm.tendon_limited != 0).sum()) if mjm.ntendon else 0
    eq_rows = {mujoco.mjtEq.mjEQ_CONNECT: 3, mujoco.mjtEq.mjEQ_WELD: 6, mujoco.mjtEq.mjEQ_JOINT: 1, mujoco.mjtEq.mjEQ_TENDON: 1}
    neq_rows = 0
    for e in range(mjm.neq):
        t = int(mjm.eq_type[e])
        if t not in eq_rows:
            raise ValueError(f"capacity bounds: equality type {t} rows are not bounded here")
        neq_rows += eq_rows[t]
    nfriction = int((mjm.dof_frictionloss > 0).sum()) + (int((mjm.tendon_frictionloss > 0).sum()) if mjm.ntendon else 0)
    nefc = nlimit + neq_rows + nfriction + sum(k * r for _, _, k, r in pairs)
    return Bounds(ncon=ncon, nefc=nefc, nlimit=nlimit, neq_rows=neq_rows, nfriction=nfriction, pairs=tuple(pairs))


def capacity_overflows(flags: dict) -> dict:
    """The subset of ``BatchSim.overflow_flags()`` that means dropped rows or contacts (ITERATIONS / LS_ITERATIONS
    are solver caps, reported separately)."""
    return {k: v for k, v in flags.items() if k in CAPACITY_FLAGS}
