"""Contact and joint-limit stiffness settings for MuJoCo (C and Warp) models, to approach PhysX's
rigid contacts and hard joint limits.

MuJoCo constraints are soft: each is a spring-damper in constraint space with time constant
``solref[0]`` (clamped by MuJoCo to >= 2 dt) and damping ratio ``solref[1]``, and an impedance
``solimp = (dmin, dmax, width, midpoint, power)`` giving the fraction of the constraint force that is
"hard" (0.9-0.95 by default; 0.99+ is stiffer; for joint limits 0.999 oscillates and 0.9999 diverges on
the G1 at 2.5 ms). Contact geometry (MuJoCo 3.14, C and Warp, measured): a contact is active when
``dist < margin``, ``gap`` only adds inactive (detected) contacts out to ``margin + gap``, and the
constraint acts on ``dist - margin``. Consequence: any ``margin > 0`` makes the robot rest ~``margin``
above the floor (G1: pelvis 0.7191 vs 0.7090 m with 1 cm, with or without gap = margin); MuJoCo has no
equivalent of PhysX's speculative contact with rest offset 0.

PhysX (Isaac Lab defaults for the G1): TGS, contact offset 0.02 m / rest offset 0 (speculative
detection, resting at zero distance), rigid unilateral contacts, hard joint limits.

Usage (compiled model, before ``BatchSim`` / ``put_model``)::

    from metalsim.physics import contact_tuning
    m, info = build_g1_model(...)
    contact_tuning.apply(m, "tau5_imp99_hardlimits")

``apply`` also accepts an ``mujoco.MjSpec`` (edits its geoms and joints; compile afterwards).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

MUJOCO_DEFAULT_SOLREF = (0.02, 1.0)
MUJOCO_DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class Tuning:
    """None leaves the model's value unchanged."""
    contact_solref: tuple | None = None
    contact_solimp: tuple | None = None
    margin: float | None = None
    gap: float | None = None
    limit_solref: tuple | None = None
    limit_solimp: tuple | None = None
    cone: str | None = None             # "pyramidal" | "elliptic"
    impratio: float | None = None       # frictional-to-normal impedance ratio (elliptic cones only)
    friction: float | None = None       # sliding friction on all geoms (pair value = max of the two)
    margin_geoms: tuple | None = None   # geom names that get margin/gap (None: all). A pair's margin/gap is
                                        # the SUM of its geoms' (C engine_collision_driver.c:166/175, MuJoCo
                                        # Warp collision_core.contact_margin_gap), so margin on the ground alone
                                        # gives every ground pair exactly that margin; MuJoCo Warp rejects
                                        # margin on mesh-mesh pairs with MULTICCD (e.g. foot-foot)
    note: str = ""


PRESETS: dict[str, Tuning] = {
    "default": Tuning(note="MuJoCo defaults (contact and limit solref 0.02/1, solimp 0.9/0.95)"),
    "tau5_imp99": Tuning(contact_solref=(0.005, 1.0), contact_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                         note="contact time constant 5 ms (2 dt at 2.5 ms), impedance 0.99-0.999"),
    "tau5_imp99_margin_gap1cm": Tuning(contact_solref=(0.005, 1.0), contact_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                                       margin=0.01, gap=0.01, margin_geoms=("ground",),
                                       note="+ margin = gap = 1 cm on the ground (acts from 1 cm; rests ~1 cm above the floor)"),
    "tau5_imp99_margin1cm": Tuning(contact_solref=(0.005, 1.0), contact_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                                   margin=0.01, gap=0.0, margin_geoms=("ground",),
                                   note="+ margin 1 cm, gap 0 (acts from 1 cm; rests ~1 cm above the floor)"),
    "tau5_imp99_hardlimits": Tuning(contact_solref=(0.005, 1.0), contact_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                                    limit_solref=(0.005, 1.0), limit_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                                    note="tau5_imp99 + joint limits at time constant 5 ms, impedance 0.99-0.999"),
    "hardlimits": Tuning(limit_solref=(0.005, 1.0), limit_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                         note="default contacts, joint limits at 5 ms / 0.99-0.999"),
}


def _fit5(v):
    v = list(v)
    return v + list(MUJOCO_DEFAULT_SOLIMP[len(v):])


def set_contacts(m, solref=None, solimp=None, margin=None, gap=None, geoms=None, margin_geoms=None):
    """Contact solref/solimp/margin/gap on all geoms (or the given geom ids) of an MjModel or MjSpec,
    and on explicit contact pairs if any; margin/gap only on ``margin_geoms`` (names) when given."""
    if margin_geoms is not None and (margin is not None or gap is not None):
        if isinstance(m, mujoco.MjSpec):
            for g in m.geoms:
                if g.name in margin_geoms:
                    if margin is not None: g.margin = margin
                    if gap is not None: g.gap = gap
        else:
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm) for nm in margin_geoms]
            if min(ids) < 0:
                raise ValueError(f"margin_geoms {margin_geoms} not all in the model")
            if margin is not None: m.geom_margin[ids] = margin
            if gap is not None: m.geom_gap[ids] = gap
        margin = gap = None
    if isinstance(m, mujoco.MjSpec):
        for g in m.geoms:
            if geoms is not None and g.name not in geoms:
                continue
            if solref is not None: g.solref = list(solref)
            if solimp is not None: g.solimp = _fit5(solimp)
            if margin is not None: g.margin = margin
            if gap is not None: g.gap = gap
        for p in m.pairs:
            if solref is not None: p.solref = list(solref)
            if solimp is not None: p.solimp = _fit5(solimp)
            if margin is not None: p.margin = margin
            if gap is not None: p.gap = gap
        return m
    idx = slice(None) if geoms is None else np.asarray(geoms)
    if solref is not None: m.geom_solref[idx] = solref
    if solimp is not None: m.geom_solimp[idx] = _fit5(solimp)
    if margin is not None: m.geom_margin[idx] = margin
    if gap is not None: m.geom_gap[idx] = gap
    if m.npair:
        if solref is not None: m.pair_solref[:] = solref
        if solimp is not None: m.pair_solimp[:] = _fit5(solimp)
        if margin is not None: m.pair_margin[:] = margin
        if gap is not None: m.pair_gap[:] = gap
    return m


def set_joint_limits(m, solref=None, solimp=None, joints=None):
    """Joint-limit solref/solimp (MuJoCo's jnt_solref/jnt_solimp act on limit constraints) on all
    limited joints (or the given joint ids) of an MjModel or MjSpec."""
    if isinstance(m, mujoco.MjSpec):
        for j in m.joints:
            if joints is not None and j.name not in joints:
                continue
            if solref is not None: j.solref_limit = list(solref)
            if solimp is not None: j.solimp_limit = _fit5(solimp)
        return m
    idx = slice(None) if joints is None else np.asarray(joints)
    if solref is not None: m.jnt_solref[idx] = solref
    if solimp is not None: m.jnt_solimp[idx] = _fit5(solimp)
    return m


def set_solver(m, cone=None, impratio=None):
    """Friction cone type and impratio on an MjModel or MjSpec."""
    opt = m.option if isinstance(m, mujoco.MjSpec) else m.opt
    if cone is not None:
        opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[cone]
    if impratio is not None:
        opt.impratio = impratio
    return m


def set_friction(m, sliding):
    """Sliding friction coefficient on every geom (MuJoCo combines a pair's friction by the max)."""
    if isinstance(m, mujoco.MjSpec):
        for g in m.geoms:
            g.friction = [sliding, g.friction[1], g.friction[2]]
    else:
        m.geom_friction[:, 0] = sliding
    return m


def apply(m, tuning: str | Tuning):
    """Apply a preset name or a ``Tuning`` to an MjModel (in place, returned) or MjSpec."""
    t = PRESETS[tuning] if isinstance(tuning, str) else tuning
    set_contacts(m, t.contact_solref, t.contact_solimp, t.margin, t.gap, margin_geoms=t.margin_geoms)
    set_joint_limits(m, t.limit_solref, t.limit_solimp)
    set_solver(m, t.cone, t.impratio)
    if t.friction is not None:
        set_friction(m, t.friction)
    return m


def check_timestep(m, tuning: str | Tuning) -> list[str]:
    """Warnings for time constants below MuJoCo's 2 dt floor (MuJoCo clamps them silently)."""
    t = PRESETS[tuning] if isinstance(tuning, str) else tuning
    out = []
    for nm, sr in (("contact", t.contact_solref), ("limit", t.limit_solref)):
        if sr is not None and sr[0] > 0 and sr[0] < 2 * m.opt.timestep:
            out.append(f"{nm} time constant {sr[0]} s < 2 dt = {2 * m.opt.timestep} s: MuJoCo uses 2 dt")
    return out


class g1_model_tuning:
    """Context manager: every ``build_g1_model`` call made through ``metalsim.learn.g1_velocity``
    (e.g. inside ``G1VelocityTask.__init__``) returns a tuned model, without editing the task::

        with contact_tuning.g1_model_tuning("tau5_imp99_hardlimits"):
            task = G1VelocityTask(4096, ...)

    The task owner's permanent one-line equivalent, in ``G1VelocityTask.__init__`` right after
    ``build_g1_model(...)``: ``contact_tuning.apply(self.model, tuning)``.
    """

    def __init__(self, tuning: str | Tuning):
        self.tuning = tuning

    def __enter__(self):
        import metalsim.learn.g1_velocity as g1
        global _ACTIVE_OVERRIDE
        self._g1, self._orig = g1, g1.build_g1_model
        self._prev_override, _ACTIVE_OVERRIDE = _ACTIVE_OVERRIDE, self.tuning
        tuning = self.tuning

        def tuned(*args, **kw):
            m, info = self._orig(*args, **kw)
            return apply(m, tuning), info
        g1.build_g1_model = tuned
        return self

    def __exit__(self, *exc):
        global _ACTIVE_OVERRIDE
        self._g1.build_g1_model = self._orig
        _ACTIVE_OVERRIDE = self._prev_override
        return False


_ACTIVE_OVERRIDE = None


def active_override():
    """The tuning of an enclosing ``g1_model_tuning`` context (G1VelocityTask then keeps it instead of its
    ``contact_cfg``), or None."""
    return _ACTIVE_OVERRIDE


_HL = dict(limit_solref=(0.005, 1.0), limit_solimp=(0.99, 0.999, 0.001, 0.5, 2.0))
PRESETS.update({
    # MuJoCo docs: "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large
    # impratio, and the Newton algorithm" (modeling.html); MuJoCo Warp has no noslip solver
    "elliptic_hardlimits": Tuning(cone="elliptic", impratio=1.0, **_HL, note="elliptic cone, impratio 1, hard limits"),
    "elliptic_imp10_hardlimits": Tuning(cone="elliptic", impratio=10.0, **_HL, note="elliptic cone, impratio 10, hard limits"),
    "tau5_imp99_elliptic_hardlimits": Tuning(contact_solref=(0.005, 1.0), contact_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                                             cone="elliptic", impratio=1.0, **_HL,
                                             note="tau5_imp99 + elliptic cone, impratio 1, hard limits"),
    # tau5_imp99 + elliptic impratio 10 diverges in the MuJoCo C drop (2.08 s at 2.5 ms): not offered
    # Isaac Lab develop's own MuJoCo Warp config for the velocity task: NewtonShapeCfg(ke=160000, kd=1100), 2.5 ms;
    # mapped here to MuJoCo's direct solref form (-stiffness, -damping), which is an assumption about Newton's mapping;
    # measured in the MuJoCo C drop the G1 does not fall (pelvis 0.688 m at 3 s vs Isaac 0.054) with 16.6 kN peaks, so
    # this mapping is not Newton's and the preset is kept only as a record
    "isaaclab_mjw_kekd_hardlimits": Tuning(contact_solref=(-160000.0, -1100.0), **_HL,
                                           note="direct solref (-1.6e5, -1.1e3) as Isaac Lab's MJWarp ke/kd, hard limits"),
})

PRESETS.update({
    # cheaper routes to stiffer contacts (throughput rule): tau 10 ms instead of the 5 ms floor, and an impedance
    # that is MuJoCo's 0.9 at touch-down and ramps to 0.999 over 5 mm of penetration (stiff only on impact)
    "tau10_imp99_hardlimits": Tuning(contact_solref=(0.01, 1.0), contact_solimp=(0.99, 0.999, 0.001, 0.5, 2.0), **_HL,
                                     note="contact tau 10 ms, impedance 0.99-0.999, hard limits"),
    "tau10_impact_hardlimits": Tuning(contact_solref=(0.01, 1.0), contact_solimp=(0.9, 0.999, 0.005, 0.5, 2.0), **_HL,
                                      note="contact tau 10 ms, impedance 0.9 -> 0.999 over 5 mm (stiff on impact), hard limits"),
})

# Measured against Isaac's PhysX recordings (parity_out2, protocols A_hold / B_random / C_drop):
# see scripts/diagnostics/bench_contact_tuning.py and runs/parity/tuning/report_*.
# 2026-09-25 decision (docs/DECISIONS.md): tau10_impact_hardlimits, the only preset reproducing Isaac's PhysX-trained
# policy transfer within noise (3.3 cm), feet_slide 28 % closer, hard limits, 1.02x cost (runs/contact_research/
# final_ranking_2026-09-25.md). G1VelocityTask applies it by default (contact_cfg="recommended").
PRESETS["recommended"] = PRESETS["tau10_impact_hardlimits"]

# Elliptic friction cones on top of the adopted preset (2026-09-25 evening, docs/research/elliptic_cones_2026-09-25.md):
# with the fork's per-world cone Hessian (MuJoCo Warp fork c301880) elliptic cones cost 1.41-1.53x instead of 7x on the
# G1 task. On the PhysX protocol, impratio 10 brings the drop-torso / hold 20 ms forces from 1.70x / 1.23x of PhysX's
# to 1.08x / 1.07x, penetration 1.64 -> 1.55 cm, feet_slide of Isaac's checkpoint -0.0144 -> -0.0131 (Isaac -0.0127),
# transfer unchanged; +7-10 % falls of the PhysX-trained checkpoint (3 seeds, within one seed sd). impratio 100 leaves ~90 %
# of worlds at the 10-iteration cap; impratio 1 is the plain elliptic cone. Candidate default for the G1 task
# (owner's decision after the confirming training run runs/g1_flat_flatcfg_ellip10.log).
_T10 = PRESETS["tau10_impact_hardlimits"]
PRESETS.update({
    f"tau10_impact_hardlimits_ellip{imp}": Tuning(contact_solref=_T10.contact_solref, contact_solimp=_T10.contact_solimp,
                                                  limit_solref=_T10.limit_solref, limit_solimp=_T10.limit_solimp,
                                                  cone="elliptic", impratio=float(imp),
                                                  note=f"{_T10.note} + elliptic cone, impratio {imp}")
    for imp in (1, 10, 100)
})
