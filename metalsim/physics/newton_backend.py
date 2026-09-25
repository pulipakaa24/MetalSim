"""Newton (XPBD) backend pieces for Isaac Lab's Unitree G1 on the Metal Warp fork.

What this module gets right that the first feasibility scripts did not (see
``scripts/diagnostics/newton_xpbd_drives.py`` for the evidence):

* ``ModelBuilder.joint_target_q`` is in the *coordinate* layout (indexed by ``joint_q_start``, 7 slots
  for the free root joint), while ``joint_target_ke``/``kd``/``mode`` are in the *DOF* layout
  (``joint_qd_start``, 6 slots for the root). Writing targets by DOF index shifts every G1 target by
  one joint (hip yaw got the knee's 0.42 rad, the knee the ankle's -0.23 rad ...).
* ``SolverXPBD`` integrates maximal coordinates only: ``State.joint_q``/``joint_qd`` are never written.
  Read joint coordinates with ``newton.eval_ik`` (``joint_state``), not from ``State.joint_q``.

* Newton's default joint relaxation (linear 0.7, angular 0.4) scales the linear and angular parts of the
  same joint impulse differently, so joints transmit torque wrongly (one-joint pendulum: response to a
  joint torque +43 %, to gravity -18 %, at any iteration count). Equal factors are exact; angular > 0.4
  diverges on the G1, so both are 0.4 here.

Drive models (``G1XPBD.drive``):

* ``"ipd"`` (recommended): Isaac Lab's actuator computed every substep by ``ActuatorPD`` and applied
  through ``Control.joint_f`` (XPBD's own drives off): ``tau = clip(kp (q* - q) - kd qd / (1 + kd dt / I),
  +-effort)`` with I the local two-body inertia about the joint axis (backward Euler on the damper's own
  effect: exact at rest, stable for Isaac's kd 10 on 1e-6 kg m^2 hand links). Armature is added
  isotropically to the child body's inertia (``armature_inertia="iso"``); without it the explicit
  stiffness diverges on the light links. Against the exact static PD equilibrium (fixed base):
  <= 0.006 rad at 1 iteration, <= 0.001 rad at 4.
* ``"xpbd"``: XPBD's own compliance drive (compliance 1/ke, damping kd/ke). Corrections are velocity
  impulses without lambda accumulation, so the effective stiffness is not ke (pendulum: 72..1240 Nm/rad
  for ke 200 over 1..16 iterations; G1 fixed base 0.03-0.4 rad off). No effort limit, no armature.
* ``"pd"``: plain explicit PD via ``joint_f`` (diverges at 2.5 ms on the hand links; reference only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

G1_USD = "assets/isaac/G1/g1_minimal.usd"
POS = newton.JointTargetMode.POSITION

# Isaac Lab G1_MINIMAL_CFG actuators: (joint-name regex, kp, kd, effort limit [Nm], armature [kg m^2])
G1_ACTUATORS = [
    (r".*_hip_yaw_joint", 150.0, 5.0, 300.0, 0.01), (r".*_hip_roll_joint", 150.0, 5.0, 300.0, 0.01),
    (r".*_hip_pitch_joint", 200.0, 5.0, 300.0, 0.01), (r".*_knee_joint", 200.0, 5.0, 300.0, 0.01),
    (r"torso_joint", 200.0, 5.0, 300.0, 0.01),
    (r".*_ankle_.*", 20.0, 2.0, 20.0, 0.01),
    (r".*_shoulder_.*", 40.0, 10.0, 300.0, 0.01), (r".*_elbow_.*", 40.0, 10.0, 300.0, 0.01),
    (r".*_(five|three|six|four|zero|one|two)_joint", 40.0, 10.0, 300.0, 0.001),
]
# Isaac Lab G1 init_state.joint_pos (unlisted joints default to 0)
G1_INIT = [
    (r".*_hip_pitch_joint", -0.20), (r".*_knee_joint", 0.42), (r".*_ankle_pitch_joint", -0.23),
    (r".*_elbow_pitch_joint", 0.87), (r"left_shoulder_roll_joint", 0.16), (r".*_shoulder_pitch_joint", 0.35),
    (r"right_shoulder_roll_joint", -0.16), (r"left_one_joint", 1.0), (r"right_one_joint", -1.0),
    (r"left_two_joint", 0.52), (r"right_two_joint", -0.52),
]


def _meshes_to_boxes(b: newton.ModelBuilder) -> None:
    """Replace mesh colliders by their bounding boxes (the G1's three mesh colliders are boxes). Needed on
    Metal before Warp fork 786cdae (no fixed-size arrays for the GJK/MPR narrow phase); optional since."""
    types = list(b.shape_type); srcs = list(b.shape_source); scales = list(b.shape_scale); xf = list(b.shape_transform)
    for i, t in enumerate(types):
        if srcs[i] is not None and t in (newton.GeoType.MESH, getattr(newton.GeoType, "CONVEX_MESH", -1)):
            v = np.asarray(srcs[i].vertices, float) * np.asarray(scales[i], float)
            lo, hi = v.min(0), v.max(0); c = 0.5 * (lo + hi); he = 0.5 * (hi - lo)
            types[i] = newton.GeoType.BOX; scales[i] = wp.vec3(*he.tolist()); srcs[i] = None
            p0, q0 = wp.transform_get_translation(xf[i]), wp.transform_get_rotation(xf[i])
            p = np.array(p0) + np.array(wp.quat_rotate(q0, wp.vec3(*c.tolist())))
            xf[i] = wp.transform(wp.vec3(*p.tolist()), q0)
    b.shape_type = types; b.shape_source = srcs; b.shape_scale = scales; b.shape_transform = xf


def g1_builder(z0: float = 0.74, mesh_to_box: bool = True, collapse_fixed_joints: bool = True,
               armature_inertia: bool | str = False, floating: bool = True,
               limit_margin: float | None = 0.15, recenter: bool = False) -> tuple[newton.ModelBuilder, dict]:
    """Isaac's G1 USD with Isaac Lab's actuator gains and default pose, targets correctly indexed.

    Returns the builder and per-DOF actuator arrays (``kp``, ``kd``, ``effort``, ``armature``,
    ``target``; length ``joint_dof_count``, zeros on the root's 6 DOFs)."""
    b = newton.ModelBuilder()
    b.add_usd(G1_USD, floating=floating, xform=wp.transform((0, 0, z0), wp.quat_identity()),
              enable_self_collisions=False, load_visual_shapes=False, collapse_fixed_joints=collapse_fixed_joints)
    b.gravity = (0.0, 0.0, -9.81)                    # the USD authors gravityMagnitude 0 (PhysX default); the importer copies it
    if mesh_to_box:
        _meshes_to_boxes(b)
    nd = b.joint_dof_count
    act = {k: np.zeros(nd) for k in ("kp", "kd", "effort", "armature", "target", "offset")}
    ke = list(b.joint_target_ke); kd = list(b.joint_target_kd); mode = list(b.joint_target_mode)
    tq = list(b.joint_target_q); q = list(b.joint_q)
    qd_start = list(b.joint_qd_start) + [nd]; q_start = list(b.joint_q_start) + [b.joint_coord_count]
    for j, lab in enumerate(b.joint_label):
        name = lab.split("/")[-1]
        for pat, kp, kv, eff, arm in G1_ACTUATORS:
            if not re.fullmatch(pat, name):
                continue
            val = next((v for ip, v in G1_INIT if re.fullmatch(ip, name)), 0.0)
            assert qd_start[j + 1] - qd_start[j] == 1 and q_start[j + 1] - q_start[j] == 1, name
            d, c = qd_start[j], q_start[j]
            ke[d] = kp; kd[d] = kv; mode[d] = POS
            tq[c] = val; q[c] = val                    # coordinate layout, NOT the DOF index
            act["kp"][d], act["kd"][d], act["effort"][d], act["armature"][d], act["target"][d] = kp, kv, eff, arm, val
            break
    b.joint_target_ke, b.joint_target_kd, b.joint_target_mode, b.joint_target_q, b.joint_q = ke, kd, mode, tq, q
    b.joint_effort_limit = [float(e) if e > 0 else old for e, old in zip(act["effort"], b.joint_effort_limit)]
    b.joint_armature = [float(a) for a in act["armature"]]
    if recenter:
        # XPBD's angle wraps at +-pi from the joint zero; move each revolute joint's zero to the middle of its limit
        # range (parent joint frame rotated by mid about the axis) so the wrap point is as far as possible from both
        # limits. Newton angle = MuJoCo/Isaac angle - offset; NewtonSim converts coordinates and targets.
        Xp = list(b.joint_X_p); lo_l = list(b.joint_limit_lower); hi_l = list(b.joint_limit_upper)
        tq = list(b.joint_target_q); q = list(b.joint_q)
        for j in range(b.joint_count):
            if b.joint_type[j] != newton.JointType.REVOLUTE:
                continue
            d, c = qd_start[j], q_start[j]
            mid = 0.5 * (lo_l[d] + hi_l[d])
            ax = wp.vec3(*np.asarray(b.joint_axis[d], float).tolist())
            X = Xp[j]
            Xp[j] = wp.transform(wp.transform_get_translation(X), wp.transform_get_rotation(X) * wp.quat_from_axis_angle(wp.normalize(ax), float(mid)))
            lo_l[d] -= mid; hi_l[d] -= mid; tq[c] -= mid; q[c] -= mid
            act["offset"][d] = mid; act["target"][d] -= mid
        b.joint_X_p, b.joint_limit_lower, b.joint_limit_upper, b.joint_target_q, b.joint_q = Xp, lo_l, hi_l, tq, q
    if limit_margin is not None:
        # XPBD measures a revolute angle as 2 asin(twist) in (-pi, pi]: a joint crossing pi (G1 elbow pitch range
        # up to 3.421 rad; hip pitch 3.05) wraps to -pi, reads as a ~2 pi limit violation and is "corrected" with a
        # huge impulse (energy x300 in one substep, the 3-sigma blow-ups). Keep every revolute limit inside
        # +-(pi - limit_margin): elbow pitch upper 3.421 -> 2.99, hip pitch upper 3.05 -> 2.99 (Newton model only).
        lim = np.pi - limit_margin
        for j in range(b.joint_count):
            if b.joint_type[j] == newton.JointType.REVOLUTE:
                d = qd_start[j]
                b.joint_limit_lower[d] = max(b.joint_limit_lower[d], -lim); b.joint_limit_upper[d] = min(b.joint_limit_upper[d], lim)
    if armature_inertia:
        add_armature_inertia(b, act["armature"], isotropic=armature_inertia == "iso")
    return b, act


def add_armature_inertia(b: newton.ModelBuilder, armature: np.ndarray, isotropic: bool = False) -> None:
    """Approximate joint armature (reflected rotor inertia) for maximal-coordinate solvers: add
    ``armature * a a^T`` to the child body's inertia (a = joint axis in the child body frame).

    Exact for a joint whose parent is fixed; otherwise it also adds the rotor inertia to the body's
    motion through the parent's rotation, where a real rotor would contribute only its spin. For the
    G1 (0.01 kg m^2 vs link inertias of 1e-3..1e-1) that side effect is small.

    Measured: NOT usable. On 32 of the G1's links the result violates the triangle inequality; Newton's
    finalize() then rebalances (e.g. hip-pitch link 0.0009/0.0014/0.0109 -> 0.0095/0.0100/0.0195 kg m^2)
    and with ``balance_inertia = False`` XPBD returns NaN in the first 0.25 s. Use ``drive="ipd"``,
    which puts the armature into the actuator's joint-space model instead. ``isotropic=True`` adds
    ``armature * I3`` (always a valid inertia; also adds the rotor inertia about the two other axes)."""
    nd = b.joint_dof_count; qd_start = list(b.joint_qd_start) + [nd]
    inertia = [np.array(b._coerce_mat33(I), float).reshape(3, 3) for I in b.body_inertia]
    for j in range(b.joint_count):
        if b.joint_type[j] != newton.JointType.REVOLUTE:
            continue
        d = qd_start[j]; arm = float(armature[d])
        if arm <= 0.0:
            continue
        c = b.joint_child[j]
        q_cj = np.array(wp.transform_get_rotation(b.joint_X_c[j]))  # joint frame in child body frame
        a = np.array(wp.quat_rotate(wp.quat(*q_cj), wp.vec3(*np.asarray(b.joint_axis[d], float))))
        inertia[c] = inertia[c] + arm * (np.eye(3) if isotropic else np.outer(a, a))
    b.body_inertia = [wp.mat33(*I.flatten().tolist()) for I in inertia]
    b.body_inv_inertia = [wp.mat33(*np.linalg.inv(I).flatten().tolist()) for I in inertia]


def heightfield_from_mujoco(hf: dict) -> tuple[newton.Heightfield, wp.transform]:
    """Newton heightfield with the same surface as the task's MuJoCo hfield dict (metalsim.learn.terrain):
    MuJoCo puts the field at geom z = zmin, spans x in [-size_x, size_x] over the ncol samples and y over the
    nrow rows, with surface z = zmin + data * elevation."""
    sx, sy, elev = float(hf["size"][0]), float(hf["size"][1]), float(hf["size"][2])
    zmin = float(hf["zmin"])
    h = newton.Heightfield(data=np.asarray(hf["data"], np.float32).reshape(hf["nrow"], hf["ncol"]), nrow=int(hf["nrow"]),
                           ncol=int(hf["ncol"]), hx=sx, hy=sy, min_z=zmin, max_z=zmin + elev)
    return h, wp.transform_identity()


def scene(n_envs: int, spacing: float = 2.5, hfield: dict | None = None, **g1_kw) -> tuple[newton.ModelBuilder, dict]:
    rb, act = g1_builder(**g1_kw)
    s = newton.ModelBuilder(); s.gravity = (0.0, 0.0, -9.81)
    if hfield is None:
        s.add_ground_plane()
    else:                                   # static, shared by all worlds (added before replicate, like the plane)
        h, X = heightfield_from_mujoco(hfield)
        s.add_shape_heightfield(xform=X, heightfield=h, label="terrain")     # default shape cfg, as the ground plane
    s.replicate(rb, n_envs, spacing=(spacing, spacing, 0.0)); s.gravity = (0.0, 0.0, -9.81)
    return s, {k: np.tile(v, n_envs) for k, v in act.items()}


@wp.kernel
def _pd_torque(joint_q: wp.array[float], joint_qd: wp.array[float], q_start: wp.array[int], qd_start: wp.array[int],
               target: wp.array[float], kp: wp.array[float], kd: wp.array[float], effort: wp.array[float],
               joint_f: wp.array[float]):
    j = wp.tid()
    d = qd_start[j]
    if qd_start[j + 1] - d != 1 or kp[d] == 0.0 and kd[d] == 0.0:
        return
    c = q_start[j]
    tau = kp[d] * (target[d] - joint_q[c]) - kd[d] * joint_qd[d]
    joint_f[d] = wp.clamp(tau, -effort[d], effort[d])


@wp.kernel
def _implicit_pd_torque(body_q: wp.array[wp.transform], body_inv_I: wp.array[wp.mat33], joint_parent: wp.array[int],
                        joint_child: wp.array[int], joint_X_p: wp.array[wp.transform], joint_axis: wp.array[wp.vec3],
                        joint_q: wp.array[float], joint_qd: wp.array[float], q_start: wp.array[int], qd_start: wp.array[int],
                        target: wp.array[float], kp: wp.array[float], kd: wp.array[float], effort: wp.array[float],
                        armature: wp.array[float], dt: float, ext_filter: float, use_ext: int,
                        grav_flag: wp.array[int], tau_g: wp.array[float], stiff_implicit: int,
                        qd_prev: wp.array[float], tau_prev: wp.array[float], tau_ext_f: wp.array[float], joint_f: wp.array[float]):
    """Isaac's actuator (PD + effort clip + armature) for a maximal-coordinate solver, per revolute DOF.

    Local joint-space model: I q'' = tau_app + tau_ext, with I = 1 / (a^T (I_p^-1 + I_c^-1) a) the two-body
    inertia about the axis and tau_ext = I a_prev - tau_app_prev the rest of last step's joint acceleration
    (optional low-pass ``ext_filter``, default off; clamped to the effort limit so a hard joint-limit impulse is not
    fought). The actuator sees I + armature: backward-Euler damping
    qd' = (qd + dt (tau_ext + kp e) / (I + arm)) / (1 + kd dt / (I + arm)), tau = clip(kp e - kd qd', +-effort),
    and the torque transmitted to the link is tau_app = (I tau - arm tau_ext) / (I + arm).
    At equilibrium tau_app = tau = kp e exactly (the XPBD compliance drive is biased there)."""
    j = wp.tid()
    d = qd_start[j]
    if qd_start[j + 1] - d != 1 or (kp[d] == 0.0 and kd[d] == 0.0):
        return
    c_id = joint_child[j]
    p_id = joint_parent[j]
    X_wp = joint_X_p[j]
    R_p = wp.identity(3, dtype=float)
    if p_id >= 0:
        X_wp = body_q[p_id] * X_wp
        R_p = wp.quat_to_matrix(wp.transform_get_rotation(body_q[p_id]))
    a = wp.transform_vector(X_wp, joint_axis[d])
    R_c = wp.quat_to_matrix(wp.transform_get_rotation(body_q[c_id]))
    a_c = wp.transpose(R_c) * a
    w = wp.dot(a_c, body_inv_I[c_id] * a_c)
    if p_id >= 0:
        a_p = wp.transpose(R_p) * a
        w += wp.dot(a_p, body_inv_I[p_id] * a_p)
    I_loc = 1.0 / w
    I_tot = I_loc + armature[d]
    e = target[d] - joint_q[q_start[j]]
    qd = joint_qd[d]
    tau = float(0.0)
    tau_ext = float(0.0)
    tau_app = float(0.0)
    if use_ext != 0:
        raw = wp.clamp((qd - qd_prev[d]) / dt * I_loc - tau_prev[d], -effort[d], effort[d])
        tau_ext = tau_ext_f[d] + ext_filter * (raw - tau_ext_f[d])
        qd_next = (qd + dt * (tau_ext + kp[d] * e) / I_tot) / (1.0 + kd[d] * dt / I_tot)
        tau = wp.clamp(kp[d] * e - kd[d] * qd_next, -effort[d], effort[d])
        tau_app = (I_loc * tau - armature[d] * tau_ext) / I_tot
    elif grav_flag[d] != 0:
        # no collider below this joint: the external torque on its subtree is gravity, known exactly.
        # Backward Euler on the joint: qd' = (qd + dt (tau_g + kp e) / I) / (1 + kd dt / I); exact at rest
        # (tau = kp e = -tau_g) and creeps at (tau_g + kp e) / kd when damping dominates (Isaac's hands)
        qd_next = (qd + dt * (tau_g[d] + kp[d] * e) / I_tot) / (1.0 + kd[d] * dt / I_tot)
        tau = wp.clamp(kp[d] * e - kd[d] * qd_next, -effort[d], effort[d])
        tau_app = tau
    elif stiff_implicit != 0:
        # contacts below this joint, stiffness implicit too (Tan et al. stable PD without the external term):
        # tau = (kp (e - dt qd) - kd qd) / (1 + (kp dt^2 + kd dt) / I); static bias 1 / (1 + kp dt^2 / I)
        tau = wp.clamp((kp[d] * (e - dt * qd) - kd[d] * qd) / (1.0 + (kp[d] * dt * dt + kd[d] * dt) / I_tot), -effort[d], effort[d])
        tau_app = tau
    else:
        # contacts below this joint (legs): backward Euler on the damper's own effect only,
        # tau_d = -kd (qd + dt tau_d / I) (exact at rest whatever the contact load)
        tau = wp.clamp(kp[d] * e - kd[d] * qd / (1.0 + kd[d] * dt / I_tot), -effort[d], effort[d])
        tau_app = tau
    joint_f[d] = tau_app
    tau_prev[d] = tau_app
    tau_ext_f[d] = tau_ext
    qd_prev[d] = qd


@wp.kernel
def _subtree_gravity_torque(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], body_mass: wp.array[float],
                            gravity: wp.vec3, joint_parent: wp.array[int], joint_X_p: wp.array[wp.transform],
                            joint_axis: wp.array[wp.vec3], qd_start: wp.array[int], grav_flag: wp.array[int],
                            sub_start: wp.array[int], sub_body: wp.array[int], tau_g: wp.array[float]):
    """Static gravity torque about each flagged revolute joint's axis: sum over its subtree of (c_b - p_j) x m_b g."""
    j = wp.tid()
    d = qd_start[j]
    if qd_start[j + 1] - d != 1 or grav_flag[d] == 0:
        return
    X_wp = joint_X_p[j]
    if joint_parent[j] >= 0:
        X_wp = body_q[joint_parent[j]] * X_wp
    p = wp.transform_get_translation(X_wp)
    a = wp.transform_vector(X_wp, joint_axis[d])
    t = wp.vec3(0.0)
    for k in range(sub_start[j], sub_start[j + 1]):
        b = sub_body[k]
        c = wp.transform_point(body_q[b], body_com[b])
        t += wp.cross(c - p, gravity * body_mass[b])
    tau_g[d] = wp.dot(t, a)


class ActuatorPD:
    """Per-substep joint torques for revolute DOFs of any Newton model, written to ``Control.joint_f``.

    ``mode="ipd"``: ``_implicit_pd_torque`` (implicit damping, armature, effort clip; recommended).
    ``mode="pd"``: plain explicit ``clip(kp e - kd qd)`` (unstable at 2.5 ms for Isaac's hand gains).
    Arrays are per DOF (``model.joint_dof_count``); the solver's own drives should be off (ke = kd = 0)."""

    def __init__(self, model, kp, kd, effort, target, armature=None, dt=0.0025, mode="ipd", ext_filter=1.0, use_ext=False,
                 gravity_implicit=True, stiff_implicit=False):
        self.model, self.dt, self.mode, self.ext_filter, self.use_ext = model, dt, mode, ext_filter, int(use_ext)
        self.stiff_implicit = int(stiff_implicit)
        self._init_subtrees(gravity_implicit)
        f = lambda a: wp.array(np.asarray(a, np.float32), dtype=float, device=model.device)
        self.kp, self.kd, self.effort, self.target = f(kp), f(kd), f(effort), f(target)
        self.armature = f(armature if armature is not None else np.zeros(model.joint_dof_count))
        z = lambda: wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        self.qd_prev, self.tau_prev, self.tau_ext_f = z(), z(), z()
        self.joint_q = wp.clone(model.joint_q); self.joint_qd = wp.clone(model.joint_qd)

    def apply(self, state, control):
        m = self.model
        newton.eval_ik(m, state, self.joint_q, self.joint_qd)
        if self.mode == "pd":
            wp.launch(_pd_torque, dim=m.joint_count, inputs=[self.joint_q, self.joint_qd, m.joint_q_start, m.joint_qd_start,
                      self.target, self.kp, self.kd, self.effort], outputs=[control.joint_f], device=m.device)
        else:
            if self.n_grav:
                wp.launch(_subtree_gravity_torque, dim=m.joint_count, inputs=[state.body_q, m.body_com, m.body_mass, self.gravity,
                          m.joint_parent, m.joint_X_p, m.joint_axis, m.joint_qd_start, self.grav_flag, self.sub_start, self.sub_body],
                          outputs=[self.tau_g], device=m.device)
            wp.launch(_implicit_pd_torque, dim=m.joint_count, inputs=[state.body_q, m.body_inv_inertia, m.joint_parent, m.joint_child,
                      m.joint_X_p, m.joint_axis, self.joint_q, self.joint_qd, m.joint_q_start, m.joint_qd_start, self.target, self.kp,
                      self.kd, self.effort, self.armature, self.dt, self.ext_filter, self.use_ext, self.grav_flag, self.tau_g,
                      self.stiff_implicit],
                      outputs=[self.qd_prev, self.tau_prev, self.tau_ext_f, control.joint_f], device=m.device)

    def _init_subtrees(self, enabled):
        """Per revolute DOF: flag = no collision shape anywhere below the joint; CSR list of the subtree's bodies."""
        m = self.model
        parent, child = m.joint_parent.numpy(), m.joint_child.numpy(); qds = m.joint_qd_start.numpy()
        shape_body = m.shape_body.numpy()
        has_shape = np.zeros(m.body_count, bool); has_shape[shape_body[shape_body >= 0]] = True
        kids = [[] for _ in range(m.body_count)]
        for j in range(m.joint_count):
            if parent[j] >= 0:
                kids[parent[j]].append(child[j])
        flag = np.zeros(m.joint_dof_count, np.int32); start = [0]; bodies = []
        for j in range(m.joint_count):
            sub, stack = [], [child[j]]
            while stack:
                b = stack.pop(); sub.append(b); stack.extend(kids[b])
            if enabled and qds[j + 1] - qds[j] == 1 and not has_shape[sub].any():
                flag[qds[j]] = 1; bodies.extend(sub)
            start.append(len(bodies))
        dev = m.device
        self.grav_flag = wp.array(flag, dtype=int, device=dev)
        self.sub_start = wp.array(np.array(start, np.int32), dtype=int, device=dev)
        self.sub_body = wp.array(np.array(bodies or [0], np.int32), dtype=int, device=dev)
        self.tau_g = wp.zeros(m.joint_dof_count, dtype=float, device=dev)
        g = np.asarray(m.gravity.numpy() if hasattr(m.gravity, "numpy") else m.gravity, float).reshape(-1, 3)[0]
        self.gravity = wp.vec3(*g.tolist())
        self.n_grav = int(flag.sum())


@dataclass
class G1XPBD:
    """N G1 robots under SolverXPBD. ``step()`` advances ``substeps`` physics steps of ``dt``."""
    n_envs: int
    iterations: int = 8
    dt: float = 0.0025
    drive: str = "xpbd"               # "xpbd" | "pd" (explicit) | "ipd" (implicit damping)
    armature_inertia: bool | str = "iso"  # armature as added child-body inertia: "iso" (default, valid), True (axis only: invalid)
    armature_model: bool = False      # "ipd": armature (also) in the actuator model; double-counts with armature_inertia
    relaxation: float = 0.4           # joint linear AND angular relaxation. Newton's defaults (0.7 / 0.4) give wrong
                                      # articulated dynamics (pendulum: torque response +43 %, gravity -18 %);
                                      # angular > 0.4 diverges on the G1 (Jacobi sum over a body's joints)
    solver_kw: dict | None = None
    device: str = "metal:0"
    z0: float = 0.74                  # Isaac Lab G1 init_state.pos
    floating: bool = True
    mesh_to_box: bool = True          # False: the USD's convex-mesh colliders via GJK/MPR (needs Warp fork 786cdae on Metal)
    use_ext: bool = False             # "ipd": external-torque estimate (measured: unstable on the hands); default:
                                      # backward Euler on the damper's own effect
    ext_filter: float = 1.0           # "ipd": low-pass factor of the tau_ext estimate (1 = none; <1 lags and destabilizes, measured)

    def __post_init__(self):
        with wp.ScopedDevice(self.device):
            builder, act = scene(self.n_envs, armature_inertia=self.armature_inertia, z0=self.z0, floating=self.floating,
                                 mesh_to_box=self.mesh_to_box)
            if self.drive in ("pd", "ipd"):                 # XPBD's own drives off; torques via joint_f
                builder.joint_target_ke = [0.0] * builder.joint_dof_count
                builder.joint_target_kd = [0.0] * builder.joint_dof_count
            self.model = m = builder.finalize()
            kw = {"joint_linear_relaxation": self.relaxation, "joint_angular_relaxation": self.relaxation, **(self.solver_kw or {})}
            self.solver = newton.solvers.SolverXPBD(m, iterations=self.iterations, **kw)
            self.s0, self.s1 = m.state(), m.state(); self.control = m.control()
            newton.eval_fk(m, m.joint_q, m.joint_qd, self.s0)
            self.pipeline = newton.CollisionPipeline(m, broad_phase="explicit")    # what Model.collide() builds
            self.contacts = self.pipeline.contacts(); self.pipeline.collide(self.s0, self.contacts)
            self.joint_q = wp.clone(m.joint_q); self.joint_qd = wp.clone(m.joint_qd)
            self.act = act
            self.actuator = None
            if self.drive in ("pd", "ipd"):
                self.actuator = ActuatorPD(m, act["kp"], act["kd"], act["effort"], act["target"],
                                           act["armature"] if self.armature_model else None, self.dt, self.drive, self.ext_filter,
                                           self.use_ext)
        self.graph = None

    def _substep(self):
        m = self.model
        if self.actuator is not None:
            self.actuator.apply(self.s0, self.control)
        self.s0.clear_forces()
        self.pipeline.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0

    def step(self, substeps: int = 1, graph: bool = False):
        """Advance ``substeps`` physics steps. With ``graph`` the (even) block is captured once and replayed."""
        with wp.ScopedDevice(self.device):
            if not graph:
                for _ in range(substeps):
                    self._substep()
                return
            assert substeps % 2 == 0, "state ping-pong must return to s0 for replay"
            if self.graph is None or self.graph[0] != substeps:
                with wp.ScopedCapture() as cap:
                    for _ in range(substeps):
                        self._substep()
                self.graph = (substeps, cap.graph)
            wp.capture_launch(self.graph[1])

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint coordinates from the maximal-coordinate state (XPBD never writes State.joint_q)."""
        with wp.ScopedDevice(self.device):
            newton.eval_ik(self.model, self.s0, self.joint_q, self.joint_qd)
            return self.joint_q.numpy().reshape(self.n_envs, -1), self.joint_qd.numpy().reshape(self.n_envs, -1)

    def pelvis_z(self) -> np.ndarray:
        return self.s0.body_q.numpy().reshape(self.n_envs, -1, 7)[:, 0, 2]


# --------------------------------------------------------------------------------------------------
# NewtonSim: a drop-in for metalsim.physics.batch.BatchSim in the G1 task. The task's kernels read and
# write MuJoCo-layout arrays (qpos [x y z qw qx qy qz, joints], qvel [v_world, w_body, joint rates],
# qacc, qfrc_actuator, sensordata, ctrl) plus foot_vel (feet body-origin world velocity for feet_slide); NewtonSim keeps those arrays and converts to / from
# Newton's maximal-coordinate state inside the captured graph, so observation, reward, termination and
# reset code is shared verbatim between the engines. Joints are matched by name (the MuJoCo model built
# from the same USD is the metadata source; FK of both agrees to 5e-7 m).

@wp.kernel
def _ctrl_to_target(ctrl: wp.array2d[float], dof_of: wp.array[int], q_offset: wp.array[float], nd: int, target: wp.array[float]):
    e, i = wp.tid()
    target[e * nd + dof_of[i]] = ctrl[e, i] - q_offset[i]


@wp.kernel
def _copy_joint_qd(src: wp.array[float], dst: wp.array[float]):
    dst[wp.tid()] = src[wp.tid()]


@wp.kernel
def _to_mujoco(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
               joint_q: wp.array[float], joint_qd: wp.array[float], joint_qd_prev: wp.array[float], joint_f: wp.array[float],
               nb: int, nc: int, nd: int, coord_of: wp.array[int], dof_of: wp.array[int], q_offset: wp.array[float],
               foot_nb: wp.vec2i, inv_dt: float,
               qpos: wp.array2d[float], qvel: wp.array2d[float], qacc: wp.array2d[float], qfrc: wp.array2d[float],
               foot_vel: wp.array2d[wp.vec3]):
    e = wp.tid()
    r = e * nb                                           # root body (pelvis) is local body 0
    X = body_q[r]
    p = wp.transform_get_translation(X); qr = wp.transform_get_rotation(X)
    v_com = wp.spatial_top(body_qd[r]); w_w = wp.spatial_bottom(body_qd[r])
    v_o = v_com - wp.cross(w_w, wp.quat_rotate(qr, body_com[r]))     # MuJoCo: velocity of the body origin
    w_b = wp.quat_rotate_inv(qr, w_w)                                  # MuJoCo free joint: body-frame angular velocity
    qpos[e, 0] = p[0]; qpos[e, 1] = p[1]; qpos[e, 2] = p[2]
    qpos[e, 3] = qr[3]; qpos[e, 4] = qr[0]; qpos[e, 5] = qr[1]; qpos[e, 6] = qr[2]
    qvel[e, 0] = v_o[0]; qvel[e, 1] = v_o[1]; qvel[e, 2] = v_o[2]
    qvel[e, 3] = w_b[0]; qvel[e, 4] = w_b[1]; qvel[e, 5] = w_b[2]
    for i in range(coord_of.shape[0]):
        c = e * nc + coord_of[i]; d = e * nd + dof_of[i]
        qpos[e, 7 + i] = joint_q[c] + q_offset[i]
        qvel[e, 6 + i] = joint_qd[d]
        qacc[e, 6 + i] = (joint_qd[d] - joint_qd_prev[d]) * inv_dt
        qfrc[e, 6 + i] = joint_f[d]
    for f in range(2):                                   # feet_slide: foot body frame origin, world linear velocity
        b = e * nb + foot_nb[f]
        Xf = body_q[b]
        w = wp.spatial_bottom(body_qd[b])
        foot_vel[e, f] = wp.spatial_top(body_qd[b]) - wp.cross(w, wp.quat_rotate(wp.transform_get_rotation(Xf), body_com[b]))


@wp.kernel
def _from_mujoco(mask: wp.array[wp.bool], qpos: wp.array2d[float], qvel: wp.array2d[float], body_com: wp.array[wp.vec3],
                 nb: int, nc: int, nd: int, coord_of: wp.array[int], dof_of: wp.array[int], q_offset: wp.array[float],
                 joint_q: wp.array[float], joint_qd: wp.array[float]):
    e = wp.tid()
    if not mask[e]:
        return
    c0 = e * nc; d0 = e * nd
    qr = wp.quat(qpos[e, 4], qpos[e, 5], qpos[e, 6], qpos[e, 3])
    for k in range(3):
        joint_q[c0 + k] = qpos[e, k]
    joint_q[c0 + 3] = qr[0]; joint_q[c0 + 4] = qr[1]; joint_q[c0 + 5] = qr[2]; joint_q[c0 + 6] = qr[3]
    w_w = wp.quat_rotate(qr, wp.vec3(qvel[e, 3], qvel[e, 4], qvel[e, 5]))
    v_com = wp.vec3(qvel[e, 0], qvel[e, 1], qvel[e, 2]) + wp.cross(w_w, wp.quat_rotate(qr, body_com[e * nb]))
    joint_qd[d0 + 0] = v_com[0]; joint_qd[d0 + 1] = v_com[1]; joint_qd[d0 + 2] = v_com[2]
    joint_qd[d0 + 3] = w_w[0]; joint_qd[d0 + 4] = w_w[1]; joint_qd[d0 + 5] = w_w[2]
    for i in range(coord_of.shape[0]):
        joint_q[c0 + coord_of[i]] = qpos[e, 7 + i] - q_offset[i]
        joint_qd[d0 + dof_of[i]] = qvel[e, 6 + i]


@wp.kernel
def _touch(count: wp.array[int], shape0: wp.array[int], shape1: wp.array[int], force: wp.array[wp.spatial_vector],
           shape_slot: wp.array[int], shape_env: wp.array[int], adr: wp.vec3i, sensordata: wp.array2d[float]):
    k = wp.tid()
    if k >= count[0]:
        return
    f = wp.length(wp.spatial_top(force[k]))
    for s in range(2):
        sh = shape0[k]
        if s == 1:
            sh = shape1[k]
        if sh >= 0:
            slot = shape_slot[sh]
            if slot >= 0:
                wp.atomic_add(sensordata, shape_env[sh], adr[slot], f)


@wp.kernel
def _touch_hist_max(count: wp.array[int], shape0: wp.array[int], shape1: wp.array[int], force: wp.array[wp.spatial_vector],
                    shape_slot: wp.array[int], shape_env: wp.array[int], slot_sel: int, hist: wp.array[float]):
    """Running max over substeps of the per-env summed contact force on one touch slot (contact history)."""
    k = wp.tid()
    if k >= count[0]:
        return
    f = wp.length(wp.spatial_top(force[k]))
    for s in range(2):
        sh = shape0[k]
        if s == 1:
            sh = shape1[k]
        if sh >= 0:
            if shape_slot[sh] == slot_sel:
                wp.atomic_max(hist, shape_env[sh], f)


@wp.kernel
def _body_pose(body_q: wp.array[wp.transform], nb: int, pose_nb: wp.array[int], pose_mj: wp.array[int],
               xpos: wp.array2d[wp.vec3], xmat: wp.array2d[wp.mat33]):
    e, k = wp.tid()
    X = body_q[e * nb + pose_nb[k]]
    xpos[e, pose_mj[k]] = wp.transform_get_translation(X)
    xmat[e, pose_mj[k]] = wp.quat_to_matrix(wp.transform_get_rotation(X))


class _NewtonData:
    """MuJoCo-layout arrays the task kernels use (subset of mujoco_warp.Data)."""


class NewtonSim:
    """N G1 worlds on Newton XPBD with ``BatchSim``'s interface for the G1 task (``launch_step``,
    ``d.*`` MuJoCo-layout arrays, ``_reset_mask``, event ordering). One ``launch_step`` = one control
    step = ``substeps`` XPBD steps of ``dt`` with Isaac's actuator (``ActuatorPD``) reading ``d.ctrl``.
    ``sensordata`` holds contact normal+friction force magnitudes summed per touch slot (feet, torso)."""

    def __init__(self, mj_model, num_envs: int, iterations: int = 4, dt: float = 0.00125, control_dt: float = 0.02,
                 device: str = "metal:0", mesh_to_box: bool = True, touch_bodies=("left_ankle_roll_link", "right_ankle_roll_link", "torso_link"),
                 hfield: dict | None = None, pose_bodies=("torso_link",), relaxation: float = 0.4,
                 actuator_kw: dict | None = None, solver_kw: dict | None = None, solver: str = "xpbd", project: bool = False,
                 recenter: bool = False, limit_margin: float | None = 0.15):
        import mujoco
        from metalsim.interop import warp_metal as wm
        self.mj_model = mj = mj_model
        self.n = n = num_envs; self.dt_phys = dt; self.iterations = iterations
        self.substeps = int(round(control_dt / dt))
        # joint accelerations (dof_acc penalty) over the last 2.5 ms: the time resolution of the MuJoCo Warp
        # path's physics step; a single 1.25 ms XPBD substep difference is dominated by projection jitter
        self.acc_window = max(1, min(self.substeps, int(round(0.0025 / dt))))
        self.device = wp.get_device(device)
        z0 = float(mj.key_qpos[0][2]) if mj.nkey else 0.74
        with wp.ScopedDevice(self.device):
            builder, act = scene(n, spacing=0.0, z0=z0, armature_inertia=False if solver == "featherstone" else "iso",
                                 mesh_to_box=mesh_to_box, hfield=hfield, recenter=recenter, limit_margin=limit_margin)
            builder.joint_target_ke = [0.0] * builder.joint_dof_count; builder.joint_target_kd = [0.0] * builder.joint_dof_count
            self.model = m = builder.finalize()
            m.request_contact_attributes("force")
            self.solver_kind, self.project = solver, project
            if solver == "featherstone":        # reduced coordinates: no joint drift by construction
                # armature enters Featherstone's joint-space inertia natively (not as added body inertia)
                self.solver = newton.solvers.SolverFeatherstone(m, **(solver_kw or {"angular_damping": 0.0}))
            else:
                self.solver = newton.solvers.SolverXPBD(m, iterations=iterations, joint_linear_relaxation=relaxation,
                                                        joint_angular_relaxation=relaxation, **(solver_kw or {}))
            self.s0, self.s1 = m.state(), m.state(); self.control = m.control()
            newton.eval_fk(m, m.joint_q, m.joint_qd, self.s0)
            self.pipeline = newton.CollisionPipeline(m, broad_phase="explicit")    # what Model.collide() builds
            self.contacts = self.pipeline.contacts(); self.pipeline.collide(self.s0, self.contacts)
            assert m.articulation_count == n
            self.nb, self.nc, self.nd = m.body_count // n, m.joint_coord_count // n, m.joint_dof_count // n
            # per-env name maps: MuJoCo actuator i (joint i + 1) -> Newton coordinate / DOF offset in an env
            lab = [l.split("/")[-1] for l in m.joint_label[: m.joint_count // n]]
            qs, qds = m.joint_q_start.numpy(), m.joint_qd_start.numpy()
            names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, mj.actuator_trnid[i, 0]) for i in range(mj.nu)]
            assert all(mj.jnt_qposadr[mj.actuator_trnid[i, 0]] == 7 + i for i in range(mj.nu)), "MuJoCo qpos must follow actuator order"
            self.coord_of = wp.array([int(qs[lab.index(nm)]) for nm in names], dtype=int)
            self.dof_of = wp.array([int(qds[lab.index(nm)]) for nm in names], dtype=int)
            # per MuJoCo actuator: Newton angle = MuJoCo angle - offset (recentred joint zeros; 0 otherwise)
            self.q_offset = wp.array(np.array([act["offset"][int(qds[lab.index(nm)])] for nm in names], np.float32), dtype=float)
            blab = [l.split("/")[-1] for l in m.body_label[: self.nb]]
            assert blab[0] == "pelvis"
            mjb = lambda nm: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, nm)
            self.foot_nb = wp.vec2i(blab.index(touch_bodies[0]), blab.index(touch_bodies[1]))
            self.foot_mj = wp.vec2i(mjb(touch_bodies[0]), mjb(touch_bodies[1]))
            sb = m.shape_body.numpy()
            self.shape_slot = wp.array([touch_bodies.index(blab[b % self.nb]) if b >= 0 and blab[b % self.nb] in touch_bodies else -1 for b in sb], dtype=int)
            self.shape_env = wp.array([b // self.nb if b >= 0 else 0 for b in sb], dtype=int)
            # bodies whose MuJoCo-layout pose (xpos, xmat) the task reads (height scanner)
            self.pose_nb = wp.array([blab.index(b) for b in pose_bodies], dtype=int)
            self.pose_mj = wp.array([mjb(b) for b in pose_bodies], dtype=int)
            sadr = lambda nm: int(mj.sensor_adr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, nm + "_touch")])
            self.touch_adr = wp.vec3i(*[sadr(b) for b in touch_bodies])
            # actuator: per-DOF gains from the same builder (Isaac's groups), target written from ctrl
            self.actuator = ActuatorPD(m, act["kp"], act["kd"], act["effort"], act["target"], None, dt, "ipd", **(actuator_kw or {}))
            self.joint_qd_prev = wp.zeros(m.joint_dof_count, dtype=float)
            self.joint_q = wp.clone(m.joint_q); self.joint_qd = wp.clone(m.joint_qd)
            # MuJoCo-layout data
            z = lambda *s, dtype=float: wp.zeros(*s, dtype=dtype)
            d = self.d = _NewtonData()
            d.qpos = wp.array(np.tile(mj.key_qpos[0], (n, 1)).astype(np.float32), dtype=float)
            d.qvel = z((n, mj.nv)); d.qacc = z((n, mj.nv)); d.qfrc_actuator = z((n, mj.nv))
            d.sensordata = z((n, mj.nsensordata)); d.site_xpos = z((n, 1), dtype=wp.vec3)
            d.foot_vel = z((n, 2), dtype=wp.vec3); d.ctrl = z((n, mj.nu))
            d.xpos = z((n, mj.nbody), dtype=wp.vec3); d.xmat = z((n, mj.nbody), dtype=wp.mat33)   # filled for pose_bodies   # feet: world linear velocity of the body origin
            d.ctrl.assign(np.tile(mj.key_qpos[0][7:], (n, 1)).astype(np.float32))
            self._reset_mask = z(n, dtype=wp.bool)
        # optional contact history (Isaac's ContactSensor history): per-env max over the last ``hist_substeps``
        # substeps of the contact force on touch slot ``hist_slot`` (2 = torso), written to ``hist_out``.
        # Per-contact max, not the per-shape sum, on the substeps before the last (summed on the last).
        self.hist_out, self.hist_substeps, self.hist_slot = None, 0, 2
        self.event = wm.SharedEvent(device, "metalsim.physics.newton")
        self._wm = wm

    # -- BatchSim interface ---------------------------------------------------------------------------
    def launch_step(self) -> None:
        m = self.model
        with wp.ScopedDevice(self.device):
            wp.launch(_ctrl_to_target, dim=(self.n, self.mj_model.nu), inputs=[self.d.ctrl, self.dof_of, self.q_offset, self.nd, self.actuator.target])
            for k in range(self.substeps):
                self.actuator.apply(self.s0, self.control)          # eval_ik -> actuator.joint_q/qd, torques
                if k == self.substeps - self.acc_window:
                    wp.launch(_copy_joint_qd, dim=m.joint_dof_count, inputs=[self.actuator.joint_qd, self.joint_qd_prev])
                self.s0.clear_forces()
                self.pipeline.collide(self.s0, self.contacts)
                self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt_phys)
                self.s0, self.s1 = self.s1, self.s0
                if self.project:                 # rebuild body poses/twists from the joint state (removes constraint drift)
                    newton.eval_ik(m, self.s0, self.joint_q, self.joint_qd)
                    newton.eval_fk(m, self.joint_q, self.joint_qd, self.s0)
                j = k - (self.substeps - self.hist_substeps)
                if self.hist_out is not None and j >= 0 and self.solver_kind != "featherstone":
                    if j == 0:
                        self.hist_out.zero_()
                    self.solver.update_contacts(self.contacts)
                    wp.launch(_touch_hist_max, dim=self.contacts.rigid_contact_max, inputs=[self.contacts.rigid_contact_count,
                              self.contacts.rigid_contact_shape0, self.contacts.rigid_contact_shape1, self.contacts.force,
                              self.shape_slot, self.shape_env, self.hist_slot], outputs=[self.hist_out])
            if self.solver_kind != "featherstone":     # SolverFeatherstone has no contact-force reporting
                self.solver.update_contacts(self.contacts)
            self._sync_out()

    def _sync_out(self):
        m, d = self.model, self.d
        newton.eval_ik(m, self.s0, self.joint_q, self.joint_qd)
        wp.launch(_to_mujoco, dim=self.n, inputs=[self.s0.body_q, self.s0.body_qd, m.body_com, self.joint_q, self.joint_qd,
                  self.joint_qd_prev, self.control.joint_f, self.nb, self.nc, self.nd, self.coord_of, self.dof_of, self.q_offset,
                  self.foot_nb, 1.0 / (self.acc_window * self.dt_phys)], outputs=[d.qpos, d.qvel, d.qacc, d.qfrc_actuator, d.foot_vel])
        wp.launch(_body_pose, dim=(self.n, self.pose_nb.shape[0]), inputs=[self.s0.body_q, self.nb, self.pose_nb, self.pose_mj],
                  outputs=[d.xpos, d.xmat])
        d.sensordata.zero_()
        if self.solver_kind == "featherstone":         # no contact forces: touch signals stay 0 (not usable for training)
            return
        wp.launch(_touch, dim=self.contacts.rigid_contact_max, inputs=[self.contacts.rigid_contact_count, self.contacts.rigid_contact_shape0,
                  self.contacts.rigid_contact_shape1, self.contacts.force, self.shape_slot, self.shape_env, self.touch_adr],
                  outputs=[d.sensordata])

    def launch_reset(self, mask=None) -> None:
        """Write the MuJoCo-layout qpos/qvel of the masked worlds into Newton's state (after g1_reset)."""
        m = self.model; mask = self._reset_mask if mask is None else mask
        with wp.ScopedDevice(self.device):
            wp.launch(_from_mujoco, dim=self.n, inputs=[mask, self.d.qpos, self.d.qvel, m.body_com, self.nb, self.nc, self.nd,
                      self.coord_of, self.dof_of, self.q_offset], outputs=[self.joint_q, self.joint_qd])
            newton.eval_fk(m, self.joint_q, self.joint_qd, self.s0, mask=mask)
            if self.solver_kind == "featherstone":   # Featherstone integrates State.joint_q / joint_qd
                wp.copy(self.s0.joint_q, self.joint_q); wp.copy(self.s0.joint_qd, self.joint_qd)
            wp.launch(_body_pose, dim=(self.n, self.pose_nb.shape[0]), inputs=[self.s0.body_q, self.nb, self.pose_nb, self.pose_mj],
                      outputs=[self.d.xpos, self.d.xmat])

    def step(self) -> int:
        """One control step from a captured graph (as BatchSim.step); returns the completion event value."""
        assert self.substeps % 2 == 0, "state ping-pong must return to s0 for graph replay"
        with wp.ScopedDevice(self.device):
            if getattr(self, "_graph", None) is None:
                with wp.ScopedCapture() as cap:
                    self.launch_step()
                self._graph = cap.graph
            wp.capture_launch(self._graph)
        return self._signal()

    def wait(self, event, value: int) -> None:
        self._wm.wait(event, value, self.device)

    def _signal(self) -> int:
        v = self.event.next_value()
        self._wm.signal(self.event, v, self.device)
        return v

    def after(self, value: int) -> None:
        from metalsim.interop import torch_bridge as tb
        tb.wait_event(self.event, value)

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)
