"""Actuation dynamics between a controller and MuJoCo position actuators, shared by MuJoCo-backed robots.

The same layers as TRON1's realism layer (``metalsim/tron1/sim.py``, ``metalsim/tron1/realism.py``; names kept where
they mean the same thing), cut so that a batched MuJoCo Warp sim can use them without a host round trip per physics
step:

* ``voltage_servo``: a voltage-driven position servo (firmware P loop -> PWM duty -> DC motor; hobby / bus servos)
  written as MuJoCo's native position actuator *exactly*: the duty clip is the actuator's ``forcerange``, the back-EMF
  (kt^2/R) plus viscous friction is the joint ``damping``. MuJoCo then produces the four-quadrant torque-speed line
  itself (TRON1's ``speed_no_load`` / ``torque_limit`` envelope, Isaac Lab's ``DCMotor``), on CPU and on the GPU, with
  no host code. It is Rhoban BAM's ``VoltageControlledActuator`` (bam/actuator.py; its MuJoCo export bam/to_mujoco.py
  ``voltage_controlled_to_mujoco``), with the firmware gain split out so other P registers can be modelled.
* ``SetpointLayer``: per physics step, on the host: command delay (``cmd_delay``, fractional, linear interpolation
  between physics steps, as BAM's ``fractional_delay_shift``; Isaac Lab's ``DelayedPDActuator`` delays by whole steps),
  then the servo firmware's setpoint profile (its internal target moves toward the goal at most ``speed_limit``, and,
  if set, ``accel_limit``; BAM's STS3215 model, bam/feetech/actuator.py ``q_target_smooth``). Both are open loop in
  the command, so a whole control period's per-physics-step targets are planned at once (``plan``).
* ``CtrlPlanFeeder``: feeds such a plan to a ``metalsim.physics.batch.BatchSim``: the first substep's targets are
  written with the command, the rest by a substep hook inside the captured step graph (one small copy per substep).
* ``bam_friction_budget`` / ``BamFriction``: Rhoban BAM's load-dependent gearbox friction (M1..M6: Stribeck,
  friction growing with the torque the gearbox carries), written to ``dof_frictionloss`` before every physics step
  on CPU MuJoCo, as BAM's own MuJoCo controller does. A geared hobby servo holds a gravity load partly by friction, so
  where it settles depends on the side it came from (so101-sag: real SO-101 followers, 2026-10-31).
  ``BamFrictionWarp`` is the same update on a ``BatchSim``: a Warp kernel over (world, dof) in the captured step graph
  after every physics substep, writing the per-world ``dof_frictionloss`` (so101-bam-gpu, 2026-10-31).

Not here, because they need the state at every physics step (TRON1 runs them in Python at 2 kHz): a first-order
current-loop lag on the *torque* (``current_lag``) and encoder quantization. For a voltage servo the electrical time
constant L/R is far below a 5 ms physics step, so it is folded into ``cmd_delay``; a torque-controlled drive (G1,
TRON1) that needs it on the GPU would add it as an actuator ``dyntype`` or a Warp hook.
"""

from dataclasses import dataclass

import numpy as np


# -- voltage-driven position servo as a native MuJoCo position actuator ----------------------------------------------

def voltage_servo(stall_torque: float, no_load_speed: float, duty_per_rad: float, max_duty: float = 1.0,
                  torque_limit: float = 1.0, coulomb: float = 0.0, kd_duty_per_rad_s: float = 0.0) -> dict:
    """MuJoCo parameters of a firmware-P-loop voltage servo, at the joint (output) side.

    Motor: tau = tau_full * duty - (kt^2 / R + b) * w - coulomb, with tau_full = kt V / R the zero-speed torque at 100 %
    duty. Firmware: duty = clip(duty_per_rad * error - kd_duty_per_rad_s * w, +-min(max_duty, torque_limit)).
    ``stall_torque`` and ``no_load_speed`` are the datasheet's two points of the torque-speed line, taken at the
    firmware's highest duty ``max_duty``, friction included: tau_full = stall_torque / max_duty, and the damping
    (back-EMF + viscous) is (stall_torque - coulomb) / no_load_speed so that the line, with the Coulomb friction, runs
    through both points. ``torque_limit`` is a duty cap below ``max_duty`` (a torque-limit register).

    Returns kp, kv (MuJoCo ``position`` actuator gains: the firmware's P and D as torque), forcerange (the duty clip,
    as torque), damping (a joint property: it acts outside the clip, as back-EMF does) and frictionloss."""
    full = stall_torque / max_duty
    return dict(kp=full * duty_per_rad, kv=full * kd_duty_per_rad_s, forcerange=full * min(max_duty, torque_limit),
                damping=(stall_torque - coulomb) / no_load_speed, frictionloss=coulomb)


# -- load-dependent gearbox friction (Rhoban BAM's friction models M1..M6) --------------------------------------------

_BAM_FLAGS = dict(m1=(), m2=("stribeck",), m3=("load",), m4=("load", "stribeck"),
                  m5=("load", "stribeck", "directional"), m6=("load", "stribeck", "directional", "quadratic"))


def bam_friction_budget(p: dict, tau_m, tau_e, dtheta):
    """Rhoban BAM's friction budget tau_fm (N.m, joint side): the largest torque friction can resist this step.

    A copy of ``bam/model.py`` ``Model.compute_frictions`` (frictionloss part; github.com/Rhoban/bam b9e1111, lines
    125-208), vectorised; checked equal to it on random inputs for m1..m6 (so101-sag ``check_budget.py``). ``p`` is a BAM
    parameter file as a dict (``bam/params/<actuator>/m1..m6.json``, its ``model`` key selects the terms); ``tau_m`` the
    motor torque, ``tau_e`` the external torque (gravity, contacts, limits; net = tau_m + tau_e), ``dtheta`` the joint
    velocity. M1 Coulomb; M2 + Stribeck; M3 + load-dependent (|tau_m - tau_e|); M4 both; M5 directional (motor- and
    external-side load coefficients); M6 + a quadratic term (``docs/theory/models.rst``). The viscous part is not here:
    it is a constant joint damping."""
    f = _BAM_FLAGS[p["model"]]
    tau_m, tau_e, dtheta = (np.asarray(x, float) for x in (tau_m, tau_e, dtheta))
    if "directional" in f:
        gb = np.abs(tau_e * p["load_friction_external"] - tau_m * p["load_friction_motor"])
        if "stribeck" in f:
            gbs = np.abs(tau_e * p["load_friction_external_stribeck"] - tau_m * p["load_friction_motor_stribeck"])
    else:
        gb = np.abs(tau_e - tau_m)
    fl = np.full(np.broadcast(tau_m, tau_e, dtheta).shape, p["friction_base"], float)
    if "load" in f:
        fl = fl + (gb if "directional" in f else p["load_friction_base"] * gb)
    if "stribeck" in f:
        sc = np.exp(-(np.abs(dtheta / p["dtheta_stribeck"]) ** p["alpha"]))      # 1 at rest, 0 when moving
        fl = fl + sc * p["friction_stribeck"]
        if "load" in f:
            fl = fl + (gbs * sc if "directional" in f else p["load_friction_stribeck"] * gb * sc)
            if "quadratic" in f:
                on = np.sign(tau_e) != np.sign(tau_m)
                fl = fl + sc * on * ((np.abs(tau_e) < np.abs(tau_m)) * p["load_friction_external_quad"] * tau_e ** 2
                                     + (np.abs(tau_e) > np.abs(tau_m)) * p["load_friction_motor_quad"] * tau_m ** 2)
    return fl


class BamFriction:
    """BAM's load-dependent friction on CPU MuJoCo: ``update(d)`` before every ``mj_step`` writes ``dof_frictionloss``
    of ``dofs`` from the previous step's data, as ``bam/mujoco.py`` ``MujocoController.update`` (lines 170-203) does:
    external torque = -qfrc_bias + qfrc_constraint - this dof's own friction-loss constraint force, motor torque =
    qfrc_actuator - emf * qvel (BAM's motor torque includes the back-EMF, which ``voltage_servo`` puts in the joint
    damping; ``emf`` = kt^2 / R). MuJoCo's friction-loss constraint then does BAM's static-friction clipping.

    ``stiffen(m)`` sets the dofs' friction-constraint solref / solimp to BAM's stiff values (``bam/mjlab.py``
    ``_STIFF_SOLREF_FRICTION`` / ``_STIFF_SOLIMP_FRICTION``, lines 252-263; its CPU testbench uses one noslip iteration
    instead): with MuJoCo's default soft friction constraint a statically held joint creeps through the band (so101-sag
    hold test: 11 mrad in 2.5 s at the shoulder, against < 1 mrad stiffened).

    The same update inside a MuJoCo Warp step graph: ``BamFrictionWarp``."""

    STIFF_SOLREF = (-5.0e4, -2.0e2)
    STIFF_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)

    def __init__(self, m, dofs, params: dict, emf: float = 0.0):
        self.m, self.dofs, self.p, self.emf = m, np.asarray(dofs, int), dict(params), float(emf)
        self.reset()

    def stiffen(self, m=None) -> None:
        m = self.m if m is None else m
        m.dof_solref[self.dofs] = self.STIFF_SOLREF
        m.dof_solimp[self.dofs] = self.STIFF_SOLIMP

    def reset(self) -> None:
        """The budget at rest with no load (call after a reset of the data)."""
        self.m.dof_frictionloss[self.dofs] = bam_friction_budget(self.p, np.zeros(len(self.dofs)),
                                                                 np.zeros(len(self.dofs)), np.zeros(len(self.dofs)))

    def update(self, d) -> None:
        import mujoco

        m, dofs = self.m, self.dofs
        own = np.zeros(m.nv)
        n = int(d.nefc)
        if n:
            k = d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
            np.add.at(own, d.efc_id[:n][k], d.efc_force[:n][k])
        tau_e = -d.qfrc_bias[dofs] + d.qfrc_constraint[dofs] - own[dofs]
        tau_m = d.qfrc_actuator[dofs] - self.emf * d.qvel[dofs]
        m.dof_frictionloss[dofs] = bam_friction_budget(self.p, tau_m, tau_e, d.qvel[dofs])


# bam_friction_budget's terms as the Warp kernel's flag bits and coefficient slots
_BAM_BITS = dict(load=1, stribeck=2, directional=4, quadratic=8)
_BAM_COEF = ("friction_base", "friction_stribeck", "load_friction_base", "load_friction_stribeck", "load_friction_external",
             "load_friction_motor", "load_friction_external_stribeck", "load_friction_motor_stribeck",
             "load_friction_external_quad", "load_friction_motor_quad", "dtheta_stribeck", "alpha")


class BamFrictionWarp:
    """``BamFriction`` on a ``metalsim.physics.batch.BatchSim`` (MuJoCo Warp): the same update, per world, inside the
    captured step graph. A Warp kernel over (world, dof) runs after every physics substep (``add_substep_hook``) and
    writes the per-world ``dof_frictionloss`` the next substep's constraint rows are built from, from that substep's
    data: the friction-loss rows of ``d.efc`` (``type`` FRICTION_DOF, ``id`` the dof, ``force``; rows below
    ``d.nefc[w]``), ``qfrc_bias``, ``qfrc_constraint``, ``qfrc_actuator`` and ``qvel``. After ``mjw.step`` (forward,
    then the integrator) the forces are the step's forward pass and ``qvel`` the integrated velocity, as in a CPU
    ``MjData`` after ``mj_step``: "after substep k" is ``BamFriction.update(d)`` "before step k + 1". The kernel is
    ``BamFriction.update`` + ``bam_friction_budget`` line for line (BAM's GPU version is ``bam/mjlab.py``
    ``BamActuator.compute`` / ``_dof_friction_force`` at github.com/Rhoban/bam b9e1111, lines 601-749; its m6 quadratic
    term, lines 516-530, drops ``bam/model.py``'s sign condition, lines 183-203; this follows ``model.py``, the
    identification code).

    The BatchSim must be built with ``"dof_frictionloss"`` in ``per_world_fields`` (one row per world; a shared row
    would be written by every world's threads) and the friction constraint already stiffened (``BamFriction.stiffen``
    on the MjModel before the BatchSim). ``update()`` launches the kernel once outside the graph: call it after a host
    write of the state and ``forward()`` (``BatchSim.set_state``), so the first step's budget comes from that state as
    on CPU; the worlds a ``reset`` selects get the budget at rest (``BamFriction.reset``) inside the reset graph."""

    def __init__(self, sim, dofs, params: dict, emf: float = 0.0):
        import mujoco
        import warp as wp

        self.wp, self.sim, self.p, self.emf = wp, sim, dict(params), float(emf)
        self.dofs_np = np.asarray(dofs, np.int32)
        fl = sim.m.dof_frictionloss
        if fl.shape[0] != sim.n:
            raise ValueError(f"BamFrictionWarp: dof_frictionloss has {fl.shape[0]} row(s) for {sim.n} worlds; build the "
                             f"BatchSim with 'dof_frictionloss' in BatchSimOptions.per_world_fields")
        f = _BAM_FLAGS[self.p["model"]]
        self.flags = int(sum(b for k, b in _BAM_BITS.items() if k in f))
        z = np.zeros(len(self.dofs_np))
        self.rest_np = np.asarray(bam_friction_budget(self.p, z, z, z), np.float32)
        self.friction_dof = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
        with wp.ScopedDevice(sim.device):
            self.dofs = wp.array(self.dofs_np, dtype=wp.int32)
            self.coef = wp.array(np.array([float(self.p.get(k, 0.0)) for k in _BAM_COEF], np.float32), dtype=wp.float32)
            self.rest = wp.array(self.rest_np, dtype=wp.float32)
        self.reset()
        sim.add_substep_hook(self._launch, self._launch_reset)

    def _launch(self) -> None:
        wp, d, m = self.wp, self.sim.d, self.sim.m
        wp.launch(_bam_kernels()["update"], dim=(self.sim.n, len(self.dofs_np)),
                  inputs=[d.nefc, d.efc.type, d.efc.id, d.efc.force, d.qfrc_bias, d.qfrc_constraint, d.qfrc_actuator,
                          d.qvel, self.dofs, self.friction_dof, self.emf, self.flags, self.coef],
                  outputs=[m.dof_frictionloss])

    def _launch_reset(self, mask) -> None:
        self.wp.launch(_bam_kernels()["rest"], dim=(self.sim.n, len(self.dofs_np)), inputs=[mask, self.dofs, self.rest],
                       outputs=[self.sim.m.dof_frictionloss])

    def update(self) -> None:
        """Launch the update once now, outside the step graph (after ``set_state`` / ``forward``)."""
        with self.wp.ScopedDevice(self.sim.device):
            self._launch()

    def reset(self) -> None:
        """Host write of the budget at rest to every world (synchronizes)."""
        self.sim.synchronize()
        fl = self.sim.m.dof_frictionloss.numpy()
        fl[:, self.dofs_np] = self.rest_np
        self.sim.m.dof_frictionloss.assign(fl)


def _bam_kernels():
    if "bam" not in _KERNELS:
        import warp as wp

        @wp.func
        def np_sign(x: wp.float32):
            # numpy's sign (0 at 0; wp.sign(0) is 1), for bam_friction_budget's quadratic-term condition
            s = wp.float32(0.0)
            if x > 0.0:
                s = wp.float32(1.0)
            if x < 0.0:
                s = wp.float32(-1.0)
            return s

        @wp.kernel
        def bam_update(nefc: wp.array(dtype=wp.int32), efc_type: wp.array2d(dtype=wp.int32),
                       efc_id: wp.array2d(dtype=wp.int32), efc_force: wp.array2d(dtype=wp.float32),
                       qfrc_bias: wp.array2d(dtype=wp.float32), qfrc_constraint: wp.array2d(dtype=wp.float32),
                       qfrc_actuator: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
                       dofs: wp.array(dtype=wp.int32), friction_dof: wp.int32, emf: wp.float32, flags: wp.int32,
                       c: wp.array(dtype=wp.float32), frictionloss: wp.array2d(dtype=wp.float32)):
            w, j = wp.tid()
            dof = dofs[j]
            # BamFriction.update: this dof's own friction-loss constraint force (rows below nefc; nefc > njmax overflows)
            n = wp.min(nefc[w], efc_type.shape[1])
            own = wp.float32(0.0)
            for i in range(n):
                if efc_type[w, i] == friction_dof:
                    if efc_id[w, i] == dof:
                        own += efc_force[w, i]
            tau_e = -qfrc_bias[w, dof] + qfrc_constraint[w, dof] - own
            dtheta = qvel[w, dof]
            tau_m = qfrc_actuator[w, dof] - emf * dtheta
            # bam_friction_budget; c = _BAM_COEF
            load = (flags & 1) != 0
            stribeck = (flags & 2) != 0
            directional = (flags & 4) != 0
            quadratic = (flags & 8) != 0
            gb = wp.float32(0.0)
            gbs = wp.float32(0.0)
            if directional:
                gb = wp.abs(tau_e * c[4] - tau_m * c[5])
                if stribeck:
                    gbs = wp.abs(tau_e * c[6] - tau_m * c[7])
            else:
                gb = wp.abs(tau_e - tau_m)
            fl = c[0]
            if load:
                if directional:
                    fl = fl + gb
                else:
                    fl = fl + c[2] * gb
            if stribeck:
                sc = wp.exp(-wp.pow(wp.abs(dtheta / c[10]), c[11]))      # 1 at rest, 0 when moving
                fl = fl + sc * c[1]
                if load:
                    if directional:
                        fl = fl + gbs * sc
                    else:
                        fl = fl + c[3] * gb * sc
                    if quadratic:
                        if np_sign(tau_e) != np_sign(tau_m):
                            q = wp.float32(0.0)
                            if wp.abs(tau_e) < wp.abs(tau_m):
                                q = c[8] * tau_e * tau_e
                            if wp.abs(tau_e) > wp.abs(tau_m):
                                q = c[9] * tau_m * tau_m
                            fl = fl + sc * q
            frictionloss[w, dof] = fl

        @wp.kernel
        def bam_rest(mask: wp.array(dtype=wp.bool), dofs: wp.array(dtype=wp.int32), rest: wp.array(dtype=wp.float32),
                     frictionloss: wp.array2d(dtype=wp.float32)):
            w, j = wp.tid()
            if mask[w]:
                frictionloss[w, dofs[j]] = rest[j]

        _KERNELS["bam"] = dict(update=bam_update, rest=bam_rest)
    return _KERNELS["bam"]


# -- host-side setpoint pipeline (command delay -> firmware profile) ---------------------------------------------------

@dataclass
class SetpointParams:
    """``cmd_delay`` s (controller -> servo, transport plus firmware), ``speed_limit`` rad/s and ``accel_limit``
    rad/s^2 of the firmware's internal target (np.inf: no limit). Scalars or per-actuator arrays."""
    cmd_delay: float = 0.0
    speed_limit: float | np.ndarray = np.inf
    accel_limit: float | np.ndarray = np.inf


class SetpointLayer:
    """(n, nu) position targets per physics step, from commands held over a control period.

    Order: the command is delayed by ``cmd_delay`` (a pure delay commutes with the time-invariant profile, so where the
    real delay sits, bus or firmware, does not matter), then the firmware's internal target ``s`` moves toward it:

    * ``accel_limit`` infinite: ``s`` steps toward the goal by at most ``speed_limit * dt`` (BAM's STS3215 model);
    * finite: the velocity of ``s`` follows the braking curve ``sign(e) min(speed_limit, sqrt(2 a |e|))`` with at most
      ``a dt`` change per step, and lands on the goal (a trapezoidal / triangular profile toward a moving goal).

    ``reset(q0)`` puts the internal target at the joint positions (the servo powers on holding where it is) and fills
    the delay line with them. Only ``mask``-ed actuators are processed; the others pass the command through."""

    def __init__(self, n: int, nu: int, dt: float, p: SetpointParams, mask=None):
        self.n, self.nu, self.dt, self.p = n, nu, float(dt), p
        self.mask = np.ones(nu, bool) if mask is None else np.asarray(mask, bool)
        self.vmax = np.broadcast_to(np.asarray(p.speed_limit, float), (nu,)).copy()
        self.amax = np.broadcast_to(np.asarray(p.accel_limit, float), (nu,)).copy()
        d = max(0.0, float(p.cmd_delay)) / self.dt
        self.k0, self.frac = int(np.floor(d)), d - np.floor(d)
        self.hist = np.zeros((self.k0 + 2, n, nu))          # hist[0] newest goal (this physics step)
        self.s = np.zeros((n, nu))
        self.v = np.zeros((n, nu))

    def reset(self, q0: np.ndarray, worlds=None) -> None:
        w = slice(None) if worlds is None else worlds
        q0 = np.asarray(q0, float).reshape(-1, self.nu) if np.ndim(q0) > 1 else np.asarray(q0, float)[None]
        self.hist[:, w] = q0
        self.s[w] = q0
        self.v[w] = 0.0

    def tick(self, goal: np.ndarray) -> np.ndarray:
        """Advance one physics step with ``goal`` (n, nu) the controller's current command; return the targets."""
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = goal
        g = (1.0 - self.frac) * self.hist[self.k0] + self.frac * self.hist[self.k0 + 1]
        dt, s, v = self.dt, self.s, self.v
        e = g - s
        fin = np.isfinite(self.amax)
        vmax = np.where(np.isfinite(self.vmax), self.vmax, 1e12)
        # velocity-limited only (BAM): internal target clamped toward the goal
        step_v = np.clip(e, -vmax * dt, vmax * dt)
        if fin.any():
            a = np.where(fin, self.amax, 1.0)
            v_brake = np.sign(e) * np.minimum(vmax, np.sqrt(2.0 * a * np.abs(e)))
            v_new = np.clip(v_brake, v - a * dt, v + a * dt)
            st = v_new * dt
            land = (np.abs(st) >= np.abs(e)) & (st * e >= 0)          # reaches (or would pass) the goal this step
            st = np.where(land, e, st)
            v_new = np.where(land, 0.0, v_new)
            step_a = st
            step = np.where(fin, step_a, step_v)
            self.v = np.where(fin, v_new, step_v / dt)
        else:
            step = step_v
            self.v = step_v / dt
        self.s = np.where(self.mask, s + step, g)
        return self.s.copy()

    def plan(self, goal: np.ndarray, nsub: int) -> np.ndarray:
        """(n, nsub, nu) targets for the ``nsub`` physics steps of one control period holding ``goal``."""
        return np.stack([self.tick(goal) for _ in range(nsub)], 1)


# -- feeding a plan to a batched MuJoCo Warp sim ----------------------------------------------------------------------

class CtrlPlanFeeder:
    """Per-substep ``ctrl`` from a host plan inside BatchSim's captured step graph.

    ``load(plan)`` (n, nsub, nu): writes ``plan[:, 0]`` to ``d.ctrl`` and the rest to a device buffer; after physics
    substep k the hook copies ``plan[:, k + 1]`` into ``d.ctrl`` (after the last substep it keeps the last one).
    BatchSim must be built with ``substeps == nsub``."""

    def __init__(self, sim, nsub: int):
        import warp as wp

        self.wp, self.sim, self.nsub = wp, sim, int(nsub)
        if int(sim.opt.substeps) != self.nsub:
            raise ValueError(f"BatchSim substeps {sim.opt.substeps} != plan length {nsub}")
        n, nu = sim.n, int(sim.d.ctrl.shape[1])
        with wp.ScopedDevice(sim.device):
            self.buf = wp.zeros((n, self.nsub, nu), dtype=wp.float32)
            self.k = wp.zeros(1, dtype=wp.int32)
        kernel = _copy_kernel()

        def hook():
            wp.launch(kernel, dim=(n, nu), inputs=[self.buf, self.k, self.nsub], outputs=[sim.d.ctrl])
            wp.launch(_inc_kernel(), dim=1, inputs=[self.k])

        sim.add_substep_hook(hook)

    def load(self, plan: np.ndarray) -> None:
        wp, sim = self.wp, self.sim
        p = np.ascontiguousarray(plan, dtype=np.float32)
        with wp.ScopedDevice(sim.device):
            sim.d.ctrl.assign(p[:, 0])
            self.buf.assign(p)
            self.k.assign(np.zeros(1, np.int32))


_KERNELS = {}


def _copy_kernel():
    if "copy" not in _KERNELS:
        import warp as wp

        @wp.kernel
        def copy_next(buf: wp.array3d(dtype=wp.float32), k: wp.array(dtype=wp.int32), nsub: int,
                      ctrl: wp.array2d(dtype=wp.float32)):
            w, j = wp.tid()
            i = wp.min(k[0] + 1, nsub - 1)
            ctrl[w, j] = buf[w, i, j]

        _KERNELS["copy"] = copy_next
    return _KERNELS["copy"]


def _inc_kernel():
    if "inc" not in _KERNELS:
        import warp as wp

        @wp.kernel
        def inc(k: wp.array(dtype=wp.int32)):
            k[0] = k[0] + 1

        _KERNELS["inc"] = inc
    return _KERNELS["inc"]
