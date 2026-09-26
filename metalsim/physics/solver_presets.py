"""MuJoCo Warp solver presets taken from another simulator's live configuration.

``isaaclab3``: the MuJoCo Warp settings Isaac Lab 3.0.0-EA runs for Isaac-Velocity-Flat-G1 on its default
``newton_mjwarp`` backend, as recorded live on the L4 (``runs/parity3/isaac/newton_mjwarp_settings.json``,
``runs/parity3/isaac/fidelity/newton_mjwarp/{meta.json, newton_generated.xml}``) and read from the sources
(Isaac Lab v3.0.0-EA ae37b028 ``source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py``
``RoughPhysicsCfg`` and ``config/g1/flat_env_cfg.py``; Newton 1.5.2 cf5378db
``newton/_src/solvers/mujoco/kernels.py`` and ``solver_mujoco.py``). Item by item:

=====================================  ==============================================  ========================================
Isaac Lab 3.0 / Newton 1.5.2            where                                            here
=====================================  ==============================================  ========================================
5 ms tick x 2 substeps (2.5 ms)         NewtonCfg.num_substeps = 2, sim.dt 0.005          2.5 ms substeps (the task's physics_dt)
solver Newton, 100 / 50 iteration caps  MJWarpSolverCfg defaults                          same caps (``iterations``)
early exit at tolerance 1e-6            MJWarpSolverCfg.tolerance; CUDA graph             same tolerance, same per-world exit:
  (ls_tolerance 0.01)                     conditional node stops launching                   MuJoCo Warp marks each converged world
                                                                                             done and skips it; without the
                                                                                             conditional node (Metal) the remaining
                                                                                             iterations still launch (cost, not result)
implicitfast, pyramidal cone,           RoughPhysicsCfg.newton_mjwarp                     same (already the task's)
  impratio 1
contact solref (1.82 ms, 1.375)         convert_solref(ke 1.6e5, kd 1100, 1, 1)          same geom solref; MuJoCo's refsafe floors
                                          (shape solref_mode default MJCF_DEFAULT)           the time constant at 2 dt = 5 ms in both
contact solimp (0.9, 0.95, 0.001)       MuJoCo default                                    same
friction: max(robot 0.8, ground 1.0)    contact_params: max of the two geoms'            already 1.0 here (robot geoms 1.0, ground
  = 1.0 (not PhysX's 0.8 x 1.0)           geom_friction when priorities are equal           0.8; MuJoCo also takes the max)
margin 0, gap 0.01 per shape            NewtonShapeCfg; gap_sum = gap_a + gap_b           geom gap 0.01 (pair 0.02; MuJoCo sums too)
per-joint limit solref                   update_jnt_solref_from_invweight0 (force-space   the recorded per-joint (timeconst, damp
  (4 ms .. 0.46 s, damping 0.03 .. 0.35)   joint_limit_ke/kd scaled by dof_invweight0)       ratio); refsafe floors 4 ms to 5 ms
collision once per 5 ms tick, reused    NewtonManager._simulate_*: collide() then 2       ``collision_every=2``: MuJoCo Warp's own
  by both substeps, contact dist/pos      solver steps; convert_newton_contacts fast        collision on the first substep of each
  refreshed from body poses               path recomputes dist/pos from the shape-local     tick; on the second, the same contacts with
                                          contact points, keeps normal and frame            dist/pos from their body-local witness
                                                                                             points (Newton's fast path)
Newton CollisionPipeline (explicit      use_mujoco_contacts=False                         NOT honoured: MuJoCo Warp's collision
  broad phase, contact reduction)                                                            (C-exact plane-convex contact set)
njmax 95 / nconmax 10 (flat)            G1FlatEnvCfg                                      capacities, not physics; larger buffers
                                                                                             here (overflow would drop constraints)
=====================================  ==============================================  ========================================

Order of application (the one place it is defined; ``G1VelocityTask`` follows it): the task's ``contact_cfg``
(``metalsim.physics.contact_tuning``, default "recommended") is applied first, then ``solver_cfg``. A solver preset here
therefore sets EVERY contact and joint-limit field it owns — geom solref / solimp / gap / margin and jnt solref / solimp —
so nothing of the contact preset underneath survives (``test_effective_model_fields_per_preset_combination``). Before
2026-09-25 19:50 ``apply`` did not set jnt_solimp or margin, and the il3 training runs composed Isaac's limit solref with
the recommended preset's 0.99–0.999 limit impedance (superseded; research note §3).

Usage::

    from metalsim.physics import solver_presets
    solver_presets.apply(m, "isaaclab3")                     # MjModel, before BatchSim / put_model
    sim = BatchSim(m, n, options=BatchSimOptions(**solver_presets.batch_options("isaaclab3", substeps=8, ...)))
    solver_presets.install(sim, "isaaclab3")                 # once-per-tick collision (re-captures the step graph)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISAACLAB3_LIMIT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)   # live mjw_model.jnt_solimp, every joint (meta.json)
ISAACLAB3_SETTINGS = os.path.join(ROOT, "runs", "parity3", "isaac", "newton_mjwarp_settings.json")


@dataclass(frozen=True)
class SolverPreset:
    iterations: int | None = None          # solver iteration cap
    ls_iterations: int | None = None       # line-search iteration cap
    tolerance: float | None = None         # early-exit tolerance (per world)
    ls_tolerance: float | None = None
    contact_solref: tuple | None = None
    contact_solimp: tuple | None = None
    geom_gap: float | None = None          # per geom (a pair's gap is the sum)
    geom_margin: float | None = None
    limit_solref: str | None = None        # "isaaclab3_live": per-joint values recorded from Isaac's live model
    limit_solimp: tuple | None = None      # overrides the limit impedance set with limit_solref (archived mixed variant only)
    limit_override: tuple | None = None    # (joint-name regex, solref, solimp) applied last (diagnostic variants)
    effort_limit: str | None = None        # "joint": the effort limit as the joint's actuatorfrcrange (Newton's MJCF) instead of
                                           # the actuator's forcerange (MetalSim's build); same total clamp, different place
    collision_every: int = 1               # substeps per collision pass (2 = once per 5 ms tick at 2.5 ms)
    newton_force_space_limits: bool = False   # joint-limit solref follows dof_invweight0 (after mass changes)
    note: str = ""


_IL3 = dict(tolerance=1e-6, ls_tolerance=0.01, contact_solref=(0.0018182, 1.375), contact_solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
            geom_gap=0.01, geom_margin=0.0, limit_solref="isaaclab3_live", newton_force_space_limits=True)
PRESETS: dict[str, SolverPreset] = {
    "isaaclab3": SolverPreset(iterations=100, ls_iterations=50, collision_every=2, **_IL3,
                              note="Isaac Lab 3.0-EA newton_mjwarp settings for the G1 (caps 100/50, tolerance 1e-6, "
                                   "soft per-joint limits, contact solref 1.82 ms/1.375, collision once per 5 ms tick)"),
    # the same numerics with smaller iteration caps: the same result (to float noise, measured 1.2e-7) for every world that converges within the cap
    # (MuJoCo Warp exits each world at the tolerance); worlds that hit the cap are counted by BatchSim.overflow_flags()
    "isaaclab3_cap20": SolverPreset(iterations=20, ls_iterations=50, collision_every=2, **_IL3,
                                    note="isaaclab3 with an iteration cap of 20 (cost)"),
    # the preset the like-for-like training uses (2026-09-25): isaaclab3's numerics with MuJoCo Warp's collision on every
    # substep (closer to Isaac's Newton recording on 7 of 11 protocol rows than the once-per-tick emulation) and an
    # iteration cap of 20 (1.8x the throughput of the cap of 100; per-world exit at the same tolerance)
    "isaaclab3_every_substep_cap20": SolverPreset(iterations=20, ls_iterations=50, collision_every=1, **_IL3,
                                                  note="isaaclab3, collision every substep, iteration cap 20"),
    # archived: what both il3 training runs of 2026-09-25 17:16-19:36 actually ran (the task's contact_cfg "recommended"
    # left its 0.99-0.999 limit impedance under Isaac's soft limit solref; superseded, kept to reproduce those runs)
    "isaaclab3_mixed_recommended_limits": SolverPreset(iterations=20, ls_iterations=50, collision_every=1,
                                                       **{**_IL3, "limit_solimp": (0.99, 0.999, 0.001, 0.5, 2.0)},
                                                       note="isaaclab3_every_substep_cap20 with the recommended preset's limit impedance"),
    # single-item variants of the training preset for the rough blow-up diagnosis (2026-09-25 evening)
    "isaaclab3_every_substep_cap20_hardlimits": SolverPreset(iterations=20, ls_iterations=50, collision_every=1,
                                                             **{**_IL3, "limit_solref": "hard", "newton_force_space_limits": False},
                                                             note="training preset with the hard-limit preset's joint limits"),
    "isaaclab3_every_substep_cap20_nogap": SolverPreset(iterations=20, ls_iterations=50, collision_every=1, **{**_IL3, "geom_gap": 0.0},
                                                        note="training preset without Isaac's 1 cm per-geom contact gap"),
    "isaaclab3_every_substep_cap20_mjcontact": SolverPreset(iterations=20, ls_iterations=50, collision_every=1,
                                                            **{**_IL3, "contact_solref": (0.02, 1.0), "geom_gap": 0.0},
                                                            note="training preset with MuJoCo's default contact solref and no gap (limits and caps kept)"),
    "isaaclab3_every_substep_cap20_hardfingers": SolverPreset(iterations=20, ls_iterations=50, collision_every=1,
                                                              **{**_IL3, "limit_override": (r".*_(zero|one|two|three|four|five|six)_joint",
                                                                                            (0.005, 1.0), (0.99, 0.999, 0.001, 0.5, 2.0))},
                                                              note="training preset with the hard-limit preset's limits on the 14 finger joints only"),
    "isaaclab3_every_substep_cap20_jointeffort": SolverPreset(iterations=20, ls_iterations=50, collision_every=1,
                                                              **{**_IL3, "effort_limit": "joint"},
                                                              note="training preset with the effort limit on the joint (Newton's layout)"),
    # archived A/B variants of single items
    "isaaclab3_collide_every_substep": SolverPreset(iterations=100, ls_iterations=50, collision_every=1, **_IL3,
                                                    note="isaaclab3 with MuJoCo Warp's collision on every substep"),
    "isaaclab3_hardlimits": SolverPreset(iterations=100, ls_iterations=50, collision_every=2,
                                         **{**_IL3, "limit_solref": "hard", "newton_force_space_limits": False},
                                         note="isaaclab3 with the hard-limit preset's joint limits (5 ms / 0.99-0.999)"),
}


def isaaclab3_joint_limits(path: str = ISAACLAB3_SETTINGS) -> dict[str, tuple[float, float]]:
    """Per-joint limit solref (timeconst, dampratio) read from Isaac's live mjw_model.jnt_solref."""
    d = json.load(open(path))
    return {nm: tuple(v["solreflimit_live"]) for nm, v in d["joint_limits"]["joints"].items()}


def apply(m: mujoco.MjModel, name: str | SolverPreset) -> mujoco.MjModel:
    """Model-level part of a preset (in place): options, contact and joint-limit parameters."""
    p = PRESETS[name] if isinstance(name, str) else name
    if p.tolerance is not None: m.opt.tolerance = p.tolerance
    if p.ls_tolerance is not None: m.opt.ls_tolerance = p.ls_tolerance
    if p.iterations is not None: m.opt.iterations = p.iterations
    if p.ls_iterations is not None: m.opt.ls_iterations = p.ls_iterations
    if p.contact_solref is not None: m.geom_solref[:] = p.contact_solref
    if p.contact_solimp is not None: m.geom_solimp[:] = p.contact_solimp
    if p.geom_gap is not None: m.geom_gap[:] = p.geom_gap
    if p.geom_margin is not None: m.geom_margin[:] = p.geom_margin
    if p.limit_solref == "isaaclab3_live":
        lim = isaaclab3_joint_limits()
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                # no limit on the root; Isaac's model carries MuJoCo's defaults there (recorded jnt_solref[0] = (0.02, 1),
                # jnt_solimp[0] default). Set them so the preset's model does not depend on what a contact preset left
                m.jnt_solref[j] = (0.02, 1.0); m.jnt_solimp[j] = ISAACLAB3_LIMIT_SOLIMP
                continue
            if nm not in lim:
                raise KeyError(f"joint {nm} has no recorded Isaac limit solref")
            m.jnt_solref[j] = lim[nm]
            m.jnt_solimp[j] = p.limit_solimp or ISAACLAB3_LIMIT_SOLIMP   # recorded jnt_solimp (a contact preset applied before may have set 0.99+)
    if p.effort_limit == "joint":
        for a in range(m.nu):
            if not m.actuator_forcelimited[a]:
                continue                          # already on the joint (build_g1_model's default since 2026-09-26)
            j = m.actuator_trnid[a][0]
            m.jnt_actfrclimited[j] = 1; m.jnt_actfrcrange[j] = m.actuator_forcerange[a]
            m.actuator_forcelimited[a] = 0
    if p.limit_override is not None:
        import re as _re
        rx, sr, si = p.limit_override
        for j in range(m.njnt):
            if _re.fullmatch(rx, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""):
                m.jnt_solref[j] = sr; m.jnt_solimp[j] = si
    if p.limit_solref == "hard":
        from metalsim.physics import contact_tuning     # read-only use of the contact agent's hard-limit values
        hl = contact_tuning.PRESETS["hardlimits"]
        contact_tuning.set_joint_limits(m, hl.limit_solref, hl.limit_solimp)
    return m


def batch_options(name: str | SolverPreset, **base) -> dict:
    """BatchSimOptions keyword arguments with the preset's iteration caps (``base`` supplies the rest)."""
    p = PRESETS[name] if isinstance(name, str) else name
    out = dict(base)
    if p.iterations is not None: out["solver_iterations"] = p.iterations
    if p.ls_iterations is not None: out["ls_iterations"] = p.ls_iterations
    return out


# --------------------------------------------------------------------------------------------------
# collision once per physics tick (Newton's collide() + num_substeps solver steps with the same contacts)

@wp.kernel
def _save_witness(nacon: wp.array(dtype=int), dist: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3),
                  frame: wp.array(dtype=wp.mat33), geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=int),
                  geom_bodyid: wp.array(dtype=int), xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                  la: wp.array(dtype=wp.vec3), lb: wp.array(dtype=wp.vec3)):
    """Body-local witness points of each contact at collision time (MuJoCo: normal = frame row 0 from geom[0]
    to geom[1], dist the signed distance along it, pos the midpoint): pa = pos - dist/2 n on geom[0]'s body,
    pb = pos + dist/2 n on geom[1]'s. After mj_step the body poses in d are still those the collision used."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    w = worldid[c]
    fr = frame[c]
    n = wp.vec3(fr[0, 0], fr[0, 1], fr[0, 2])
    pa = pos[c] - 0.5 * dist[c] * n
    pb = pos[c] + 0.5 * dist[c] * n
    b0 = geom_bodyid[geom[c][0]]
    b1 = geom_bodyid[geom[c][1]]
    la[c] = wp.transpose(xmat[w, b0]) * (pa - xpos[w, b0])
    lb[c] = wp.transpose(xmat[w, b1]) * (pb - xpos[w, b1])


@wp.kernel
def _refresh_contacts(nacon: wp.array(dtype=int), frame: wp.array(dtype=wp.mat33), geom: wp.array(dtype=wp.vec2i),
                      worldid: wp.array(dtype=int), geom_bodyid: wp.array(dtype=int), xpos: wp.array2d(dtype=wp.vec3),
                      xmat: wp.array2d(dtype=wp.mat33), la: wp.array(dtype=wp.vec3), lb: wp.array(dtype=wp.vec3),
                      dist: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3), efc_address: wp.array2d(dtype=int)):
    """Newton 1.5.2 convert_newton_contacts_to_mjwarp_kernel fast path: same contact set, normal and frame;
    dist and pos from the witness points moved with the current body poses; efc_address reset."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    w = worldid[c]
    fr = frame[c]
    n = wp.vec3(fr[0, 0], fr[0, 1], fr[0, 2])
    b0 = geom_bodyid[geom[c][0]]
    b1 = geom_bodyid[geom[c][1]]
    pa = xpos[w, b0] + xmat[w, b0] * la[c]
    pb = xpos[w, b1] + xmat[w, b1] * lb[c]
    dist[c] = wp.dot(n, pb - pa)
    pos[c] = 0.5 * (pa + pb)
    for i in range(efc_address.shape[1]):
        efc_address[c, i] = -1


_reuse = {"fn": None}
_patched = {"done": False}


def _patch_collision():
    """Route MuJoCo Warp's collision call through a switch that a capture can turn into the refresh kernel."""
    if _patched["done"]:
        return
    from mujoco_warp._src import collision_driver
    orig = collision_driver.collision

    def collision(m, d, *a, **k):
        fn = _reuse["fn"]
        if fn is not None:
            return fn(m, d)
        return orig(m, d, *a, **k)
    collision_driver.collision = collision
    _patched["done"] = True


def install(sim, name: str | SolverPreset) -> None:
    """Runtime part of a preset on a BatchSim: collision once per ``collision_every`` substeps (no-op for 1)."""
    p = PRESETS[name] if isinstance(name, str) else name
    every = int(p.collision_every)
    sim.collision_every = every
    if every <= 1:
        return
    if sim.opt.substeps % every:
        raise ValueError(f"substeps {sim.opt.substeps} not a multiple of collision_every {every}")
    tick = float(sim.mj_model.opt.timestep) * every
    if abs(tick - 0.005) > 1e-9:
        raise ValueError(f"collision_every {every} at timestep {sim.mj_model.opt.timestep} gives a {tick * 1e3:.2f} ms tick; "
                         "Isaac's is 5 ms (2 substeps of 2.5 ms): use physics_dt 0.0025")
    import types
    import mujoco_warp as mjw
    _patch_collision()
    d, m = sim.d, sim.m
    dev = sim.device
    la = wp.zeros(d.naconmax, dtype=wp.vec3, device=dev); lb = wp.zeros(d.naconmax, dtype=wp.vec3, device=dev)
    sim._witness = (la, lb)

    def refresh(mm, dd):
        wp.launch(_refresh_contacts, dim=dd.naconmax, device=dev, inputs=[
            dd.nacon, dd.contact.frame, dd.contact.geom, dd.contact.worldid, mm.geom_bodyid, dd.xpos, dd.xmat, la, lb,
            dd.contact.dist, dd.contact.pos, dd.contact.efc_address])

    def _launch_substeps(self):
        for i in range(self.opt.substeps):
            if i % every == 0:
                mjw.step(self.m, self.d)
                wp.launch(_save_witness, dim=self.d.naconmax, device=dev, inputs=[
                    self.d.nacon, self.d.contact.dist, self.d.contact.pos, self.d.contact.frame, self.d.contact.geom,
                    self.d.contact.worldid, self.m.geom_bodyid, self.d.xpos, self.d.xmat, la, lb])
            else:
                _reuse["fn"] = refresh
                try:
                    mjw.step(self.m, self.d)
                finally:
                    _reuse["fn"] = None
            for h in self._substep_hooks:
                h()
    sim._launch_substeps = types.MethodType(_launch_substeps, sim)
    with wp.ScopedDevice(dev):
        sim._capture()
