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
               armature_inertia: bool | str = False, floating: bool = True) -> tuple[newton.ModelBuilder, dict]:
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
    act = {k: np.zeros(nd) for k in ("kp", "kd", "effort", "armature", "target")}
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


def scene(n_envs: int, spacing: float = 2.5, **g1_kw) -> tuple[newton.ModelBuilder, dict]:
    rb, act = g1_builder(**g1_kw)
    s = newton.ModelBuilder(); s.gravity = (0.0, 0.0, -9.81); s.add_ground_plane()
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
    else:
        # backward Euler on the damper's own effect only: tau_d = -kd (qd + dt tau_d / I) (exact at rest)
        tau = wp.clamp(kp[d] * e - kd[d] * qd / (1.0 + kd[d] * dt / I_tot), -effort[d], effort[d])
        tau_app = tau
    joint_f[d] = tau_app
    tau_prev[d] = tau_app
    tau_ext_f[d] = tau_ext
    qd_prev[d] = qd


class ActuatorPD:
    """Per-substep joint torques for revolute DOFs of any Newton model, written to ``Control.joint_f``.

    ``mode="ipd"``: ``_implicit_pd_torque`` (implicit damping, armature, effort clip; recommended).
    ``mode="pd"``: plain explicit ``clip(kp e - kd qd)`` (unstable at 2.5 ms for Isaac's hand gains).
    Arrays are per DOF (``model.joint_dof_count``); the solver's own drives should be off (ke = kd = 0)."""

    def __init__(self, model, kp, kd, effort, target, armature=None, dt=0.0025, mode="ipd", ext_filter=1.0, use_ext=False):
        self.model, self.dt, self.mode, self.ext_filter, self.use_ext = model, dt, mode, ext_filter, int(use_ext)
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
            wp.launch(_implicit_pd_torque, dim=m.joint_count, inputs=[state.body_q, m.body_inv_inertia, m.joint_parent, m.joint_child,
                      m.joint_X_p, m.joint_axis, self.joint_q, self.joint_qd, m.joint_q_start, m.joint_qd_start, self.target, self.kp,
                      self.kd, self.effort, self.armature, self.dt, self.ext_filter, self.use_ext],
                      outputs=[self.qd_prev, self.tau_prev, self.tau_ext_f, control.joint_f], device=m.device)


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
# qacc, qfrc_actuator, sensordata, cvel, ctrl); NewtonSim keeps those arrays and converts to / from
# Newton's maximal-coordinate state inside the captured graph, so observation, reward, termination and
# reset code is shared verbatim between the engines. Joints are matched by name (the MuJoCo model built
# from the same USD is the metadata source; FK of both agrees to 5e-7 m).

@wp.kernel
def _ctrl_to_target(ctrl: wp.array2d[float], dof_of: wp.array[int], nd: int, target: wp.array[float]):
    e, i = wp.tid()
    target[e * nd + dof_of[i]] = ctrl[e, i]


@wp.kernel
def _copy_joint_qd(src: wp.array[float], dst: wp.array[float]):
    dst[wp.tid()] = src[wp.tid()]


@wp.kernel
def _to_mujoco(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
               joint_q: wp.array[float], joint_qd: wp.array[float], joint_qd_prev: wp.array[float], joint_f: wp.array[float],
               nb: int, nc: int, nd: int, coord_of: wp.array[int], dof_of: wp.array[int],
               foot_nb: wp.vec2i, foot_mj: wp.vec2i, inv_dt: float,
               qpos: wp.array2d[float], qvel: wp.array2d[float], qacc: wp.array2d[float], qfrc: wp.array2d[float],
               cvel: wp.array2d[wp.spatial_vector]):
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
        qpos[e, 7 + i] = joint_q[c]
        qvel[e, 6 + i] = joint_qd[d]
        qacc[e, 6 + i] = (joint_qd[d] - joint_qd_prev[d]) * inv_dt
        qfrc[e, 6 + i] = joint_f[d]
    for f in range(2):                                   # feet: [angular, linear COM velocity] (feet_slide)
        cv = body_qd[e * nb + foot_nb[f]]
        cvel[e, foot_mj[f]] = wp.spatial_vector(wp.spatial_bottom(cv), wp.spatial_top(cv))


@wp.kernel
def _from_mujoco(mask: wp.array[wp.bool], qpos: wp.array2d[float], qvel: wp.array2d[float], body_com: wp.array[wp.vec3],
                 nb: int, nc: int, nd: int, coord_of: wp.array[int], dof_of: wp.array[int],
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
        joint_q[c0 + coord_of[i]] = qpos[e, 7 + i]
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


class _NewtonData:
    """MuJoCo-layout arrays the task kernels use (subset of mujoco_warp.Data)."""


class NewtonSim:
    """N G1 worlds on Newton XPBD with ``BatchSim``'s interface for the G1 task (``launch_step``,
    ``d.*`` MuJoCo-layout arrays, ``_reset_mask``, event ordering). One ``launch_step`` = one control
    step = ``substeps`` XPBD steps of ``dt`` with Isaac's actuator (``ActuatorPD``) reading ``d.ctrl``.
    ``sensordata`` holds contact normal+friction force magnitudes summed per touch slot (feet, torso)."""

    def __init__(self, mj_model, num_envs: int, iterations: int = 4, dt: float = 0.00125, control_dt: float = 0.02,
                 device: str = "metal:0", mesh_to_box: bool = True, touch_bodies=("left_ankle_roll_link", "right_ankle_roll_link", "torso_link")):
        import mujoco
        from metalsim.interop import warp_metal as wm
        self.mj_model = mj = mj_model
        self.n = n = num_envs; self.dt_phys = dt; self.iterations = iterations
        self.substeps = int(round(control_dt / dt))
        self.device = wp.get_device(device)
        z0 = float(mj.key_qpos[0][2]) if mj.nkey else 0.74
        with wp.ScopedDevice(self.device):
            builder, act = scene(n, spacing=0.0, z0=z0, armature_inertia="iso", mesh_to_box=mesh_to_box)
            builder.joint_target_ke = [0.0] * builder.joint_dof_count; builder.joint_target_kd = [0.0] * builder.joint_dof_count
            self.model = m = builder.finalize()
            m.request_contact_attributes("force")
            self.solver = newton.solvers.SolverXPBD(m, iterations=iterations, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
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
            blab = [l.split("/")[-1] for l in m.body_label[: self.nb]]
            assert blab[0] == "pelvis"
            mjb = lambda nm: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, nm)
            self.foot_nb = wp.vec2i(blab.index(touch_bodies[0]), blab.index(touch_bodies[1]))
            self.foot_mj = wp.vec2i(mjb(touch_bodies[0]), mjb(touch_bodies[1]))
            sb = m.shape_body.numpy()
            self.shape_slot = wp.array([touch_bodies.index(blab[b % self.nb]) if b >= 0 and blab[b % self.nb] in touch_bodies else -1 for b in sb], dtype=int)
            self.shape_env = wp.array([b // self.nb if b >= 0 else 0 for b in sb], dtype=int)
            sadr = lambda nm: int(mj.sensor_adr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, nm + "_touch")])
            self.touch_adr = wp.vec3i(*[sadr(b) for b in touch_bodies])
            # actuator: per-DOF gains from the same builder (Isaac's groups), target written from ctrl
            self.actuator = ActuatorPD(m, act["kp"], act["kd"], act["effort"], act["target"], None, dt, "ipd")
            self.joint_qd_prev = wp.zeros(m.joint_dof_count, dtype=float)
            self.joint_q = wp.clone(m.joint_q); self.joint_qd = wp.clone(m.joint_qd)
            # MuJoCo-layout data
            z = lambda *s, dtype=float: wp.zeros(*s, dtype=dtype)
            d = self.d = _NewtonData()
            d.qpos = wp.array(np.tile(mj.key_qpos[0], (n, 1)).astype(np.float32), dtype=float)
            d.qvel = z((n, mj.nv)); d.qacc = z((n, mj.nv)); d.qfrc_actuator = z((n, mj.nv))
            d.sensordata = z((n, mj.nsensordata)); d.site_xpos = z((n, 1), dtype=wp.vec3)
            d.cvel = z((n, mj.nbody), dtype=wp.spatial_vector); d.ctrl = z((n, mj.nu))
            d.ctrl.assign(np.tile(mj.key_qpos[0][7:], (n, 1)).astype(np.float32))
            self._reset_mask = z(n, dtype=wp.bool)
        self.event = wm.SharedEvent(device, "metalsim.physics.newton")
        self._wm = wm

    # -- BatchSim interface ---------------------------------------------------------------------------
    def launch_step(self) -> None:
        m = self.model
        with wp.ScopedDevice(self.device):
            wp.launch(_ctrl_to_target, dim=(self.n, self.mj_model.nu), inputs=[self.d.ctrl, self.dof_of, self.nd, self.actuator.target])
            for k in range(self.substeps):
                self.actuator.apply(self.s0, self.control)          # eval_ik -> actuator.joint_q/qd, torques
                if k == self.substeps - 1:
                    wp.launch(_copy_joint_qd, dim=m.joint_dof_count, inputs=[self.actuator.joint_qd, self.joint_qd_prev])
                self.s0.clear_forces()
                self.pipeline.collide(self.s0, self.contacts)
                self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt_phys)
                self.s0, self.s1 = self.s1, self.s0
            self.solver.update_contacts(self.contacts)
            self._sync_out()

    def _sync_out(self):
        m, d = self.model, self.d
        newton.eval_ik(m, self.s0, self.joint_q, self.joint_qd)
        wp.launch(_to_mujoco, dim=self.n, inputs=[self.s0.body_q, self.s0.body_qd, m.body_com, self.joint_q, self.joint_qd,
                  self.joint_qd_prev, self.control.joint_f, self.nb, self.nc, self.nd, self.coord_of, self.dof_of,
                  self.foot_nb, self.foot_mj, 1.0 / self.dt_phys], outputs=[d.qpos, d.qvel, d.qacc, d.qfrc_actuator, d.cvel])
        d.sensordata.zero_()
        wp.launch(_touch, dim=self.contacts.rigid_contact_max, inputs=[self.contacts.rigid_contact_count, self.contacts.rigid_contact_shape0,
                  self.contacts.rigid_contact_shape1, self.contacts.force, self.shape_slot, self.shape_env, self.touch_adr],
                  outputs=[d.sensordata])

    def launch_reset(self, mask=None) -> None:
        """Write the MuJoCo-layout qpos/qvel of the masked worlds into Newton's state (after g1_reset)."""
        m = self.model; mask = self._reset_mask if mask is None else mask
        with wp.ScopedDevice(self.device):
            wp.launch(_from_mujoco, dim=self.n, inputs=[mask, self.d.qpos, self.d.qvel, m.body_com, self.nb, self.nc, self.nd,
                      self.coord_of, self.dof_of], outputs=[self.joint_q, self.joint_qd])
            newton.eval_fk(m, self.joint_q, self.joint_qd, self.s0, mask=mask)

    def step(self) -> int:
        self.launch_step()
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
