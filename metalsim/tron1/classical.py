"""Classical (model-based, no learning) balance and locomotion controller for TRON1 wheel-foot.

Uses only what the SDK reports: joint angles and velocities from the encoders, and the IMU
(orientation, gyro). Three parts:

1. Leg inverse kinematics. For a commanded ride height h, `Tron1Model.stance(h)` solves hip and
   knee (legs mirrored, abad 0) so the wheel axles sit h below the base origin and the robot's
   centre of mass is straight above them with the base level. A table over h is built once.
   Roll is levelled by lengthening one leg and shortening the other (per-leg IK), and a gravity
   feedforward `tau = bias - J_wheel^T F_ground` from the nominal model cancels the static sag of
   the joint PD, with a small integral term for the unknown payload. The wheel drive load is fed
   forward to the legs only for the *differential* (turning) torque: for the common (balance)
   torque the pendulum leans so the ground force points from the tyre to the centre of mass and
   the legs carry no extra moment (checked in sim: feeding it forward made the hips deflect
   0.08 rad/N m instead of 0.004), whereas opposite wheel forces while turning scissor the legs.
2. Balance: a wheel-inverted-pendulum LQR. The pendulum parameters (body mass, COM distance from
   the axle, pitch inertia, wheel mass and inertia) are computed from LimX's model at the current
   stance. State = (x - x_ref, phi, xdot - v_ref, phidot), phi = base pitch from the IMU + the
   angle of the body COM from the axle given by the encoders; x and xdot come from wheel encoders
   plus pitch rate (rolling without slip). Output = total wheel torque.
   A balance-point estimator integrates the odometry position error into a pitch trim, so an
   unknown COM offset or IMU mounting error does not become a standing drift.
3. Heading: PD on heading from the wheel encoders (drift-free unless the tyres slip) with gyro
   yaw-rate damping, applied as a torque difference between the wheels.

Commands go out as SDK mode 0: legs `Kp, Kd, q_des, tau_ff`; wheels pure torque.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are

from .sdk import LEG, N_MOTORS, WF_MOTOR_NAMES, WHEEL, ImuData, RobotCmd, RobotState
from .sim import WF_XML, WHEEL_RADIUS, quat_from_rpy, quat_to_mat

G = 9.81


class Tron1Model:
    """Nominal kinematics/dynamics the controller believes (LimX model + estimated payload)."""

    def __init__(self, payload_mass=1.5, payload_pos=(0.0, 0.0, 0.08), wheel_armature=0.008):
        spec = mujoco.MjSpec.from_file(str(WF_XML))
        from .sim import pack_inertia
        self.payload_mass, self.payload_pos = float(payload_mass), tuple(float(v) for v in payload_pos)
        if payload_mass > 0:
            b = spec.body("base_Link").add_body(name="payload", pos=[0.0, 0.0, 0.0])
            b.mass = payload_mass
            b.ipos = list(payload_pos)
            b.inertia = pack_inertia(payload_mass)
            b.explicitinertial = True
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.jq = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in WF_MOTOR_NAMES])
        self.jv = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in WF_MOTOR_NAMES])
        self.wheel_bodies = [self.m.body(f"wheel_{s}_Link").id for s in "LR"]
        self.total_mass = float(self.m.body_subtreemass[1])
        self.m_wheel = float(sum(self.m.body_mass[w] for w in self.wheel_bodies)) / 2
        self.I_wheel = float(self.m.body_inertia[self.wheel_bodies[0]].max()) + wheel_armature
        self.lo = self.m.jnt_range[[self.m.joint(n).id for n in WF_MOTOR_NAMES], 0]
        self.hi = self.m.jnt_range[[self.m.joint(n).id for n in WF_MOTOR_NAMES], 1]

    def _set(self, q8):
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.jq] = q8
        mujoco.mj_forward(self.m, d)

    @staticmethod
    def mirror(hip, knee, abad_l=0.0, abad_r=0.0, hip_r=None, knee_r=None):
        hip_r = hip if hip_r is None else hip_r
        knee_r = knee if knee_r is None else knee_r
        return np.array([abad_l, hip, knee, 0.0, abad_r, -hip_r, -knee_r, 0.0])

    def wheel_centers(self, q8):
        self._set(q8)
        return np.array([self.d.xpos[w] for w in self.wheel_bodies])

    def body_com(self, q8):
        """COM of everything except the wheels, base frame."""
        self._set(q8)
        m, d = self.m, self.d
        mask = np.ones(m.nbody, bool)
        mask[0] = False
        mask[self.wheel_bodies] = False
        w = m.body_mass[mask]
        return (d.xipos[mask] * w[:, None]).sum(0) / w.sum(), float(w.sum())

    def pendulum(self, q8):
        """Wheel-inverted-pendulum parameters at configuration q8 (base level)."""
        c, M = self.body_com(q8)
        a = self.wheel_centers(q8).mean(0)
        m, d = self.m, self.d
        I = 0.0
        for b in range(1, m.nbody):
            if b in self.wheel_bodies:
                continue
            R = d.ximat[b].reshape(3, 3)
            Iw = R @ np.diag(m.body_inertia[b]) @ R.T
            r = d.xipos[b] - c
            I += Iw[1, 1] + m.body_mass[b] * (r[0] ** 2 + r[2] ** 2)
        v = c - a
        return dict(M=M, l=float(np.hypot(v[0], v[2])), I=I, alpha=float(np.arctan2(v[0], v[2])),
                    m_w=self.m_wheel, I_w=self.I_wheel, track=float(abs(np.diff(self.wheel_centers(q8)[:, 1])[0])))

    def lean_angle(self, q8):
        """Angle of the body COM from the axle midpoint, base frame (positive = COM ahead)."""
        c, _ = self.body_com(q8)
        a = self.wheel_centers(q8).mean(0)
        return float(np.arctan2(c[0] - a[0], c[2] - a[2]))

    def stance(self, h, iters=30):
        """Solve (hip, knee) so the axle is h below the base origin and total COM is above it."""
        x = np.array([0.3, 0.6])
        for _ in range(iters):
            f = self._stance_residual(x, h)
            if np.linalg.norm(f) < 1e-9:
                break
            J = np.zeros((2, 2))
            for k in range(2):
                e = np.zeros(2)
                e[k] = 1e-6
                J[:, k] = (self._stance_residual(x + e, h) - f) / 1e-6
            x = x - np.linalg.solve(J, f)
        return x

    def _stance_residual(self, x, h):
        q8 = self.mirror(*x)
        self._set(q8)
        a = np.array([self.d.xpos[w] for w in self.wheel_bodies]).mean(0)
        com = self.d.subtree_com[1]
        return np.array([a[2] + h, com[0] - a[0]])

    def leg_ik(self, q8_guess, side, target, iters=8):
        """Per-leg IK: (abad, hip, knee) of `side` (0 L, 1 R) placing its wheel centre at `target`."""
        idx = np.array([0, 1, 2]) + 4 * side
        q8 = q8_guess.copy()
        for _ in range(iters):
            self._set(q8)
            p = self.d.xpos[self.wheel_bodies[side]]
            err = target - p
            if np.linalg.norm(err) < 1e-7:
                break
            jac = np.zeros((3, self.m.nv))
            mujoco.mj_jacBody(self.m, self.d, jac, None, self.wheel_bodies[side])
            Jl = jac[:, self.jv[idx]]
            q8[idx] += np.linalg.lstsq(Jl, err, rcond=None)[0]
            q8[idx] = np.clip(q8[idx], self.lo[idx], self.hi[idx])
        return q8

    def wheel_reaction_ff(self, q8):
        """Leg joint torques per unit wheel torque, (8, 2), from the static wheel drive load.

        A wheel torque tau pushes the ground with tau / r and reacts with -tau on the knee link;
        this returns the joint torques that hold the leg against both.
        """
        self._set(q8)
        m, d = self.m, self.d
        out = np.zeros((N_MOTORS, 2))
        for side, w in enumerate(self.wheel_bodies):
            jp = np.zeros((3, m.nv))
            jr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, None, w)
            mujoco.mj_jacBody(m, d, None, jr, m.body_parentid[w])
            axis = d.xmat[w].reshape(3, 3)[:, 1]
            out[:, side] = -(jp[:, self.jv].T @ np.array([1.0 / WHEEL_RADIUS, 0.0, 0.0]) - jr[:, self.jv].T @ axis)
        out[WHEEL] = 0.0
        return out

    def gravity_ff(self, q8, quat=(1.0, 0.0, 0.0, 0.0)):
        """Leg torques holding the robot up statically at joint angles q8 and base tilt `quat`.

        tau = bias - sum_i J_i^T F_i, with the vertical ground forces F_L, F_R split by the moment
        balance about the line through the two tyres (world frame, yaw removed): when the robot
        rolls, the downhill leg carries more, and a fixed half-half split lets it sag and roll more.
        """
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[3:7] = quat
        d.qpos[self.jq] = q8
        mujoco.mj_forward(m, d)
        pL, pR = (d.xpos[w] for w in self.wheel_bodies)
        com = d.subtree_com[1]
        axis = pL - pR
        axis[2] = 0.0
        L2 = axis @ axis
        s = np.clip((com - pR)[:2] @ axis[:2] / L2, 0.0, 1.0)     # 0 = all on R, 1 = all on L
        W = self.total_mass * G
        tau = d.qfrc_bias[self.jv].copy()
        for w, F in zip(self.wheel_bodies, (s * W, (1 - s) * W)):
            jac = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jac, None, w)
            tau -= jac[:, self.jv].T @ np.array([0.0, 0.0, F])
        tau[WHEEL] = 0.0
        return tau

def wip_lqr(p, Q=(4.0, 60.0, 3.0, 4.0), R=0.2):
    """Continuous LQR for the linearized wheel-inverted pendulum. Returns K (1x4) and A, B."""
    M, l, I, mw, Iw, r = p["M"], p["l"], p["I"], 2 * p["m_w"], 2 * p["I_w"], WHEEL_RADIUS
    Mm = np.array([[M + mw + Iw / r ** 2, M * l], [M * l, I + M * l ** 2]])
    Minv = np.linalg.inv(Mm)
    # [xdd, phidd] = Minv @ ([0, M g l] phi + [1/r, -1] tau)
    A = np.zeros((4, 4))
    A[0, 2] = A[1, 3] = 1.0
    A[2:, 1] = Minv @ np.array([0.0, M * G * l])
    B = np.zeros((4, 1))
    B[2:, 0] = Minv @ np.array([1.0 / r, -1.0])
    P = solve_continuous_are(A, B, np.diag(Q), np.array([[R]]))
    K = np.linalg.solve(np.array([[R]]), B.T @ P)
    return K, A, B


@dataclass
class ClassicalGains:
    height: float = 0.70           # axle below base origin [m]; stance table covers 0.55-0.76
    leg_kp: float = 60.0
    leg_kd: float = 2.5
    # The two legs and the ground form a parallelogram: with point-like tyre contacts (LimX's
    # model has a 1 cm wide disc) the base can sway sideways with both abads turning together,
    # and gravity pushes that mode over unless 2 Kp_abad > M g h (about 150 N m/rad at h = 0.7).
    abad_kp: float = 150.0
    abad_kd: float = 4.0
    leg_ki: float = 20.0           # N m / (rad s), integral on leg joint error
    leg_ki_limit: float = 15.0
    # Sensitive: on 48 randomized robots, kp/kd 0/0 -> 8 falls, 0.5/0.05 -> 4, 0.5/0.1 -> 48.
    lateral_kp: float = 0.5        # wheel shift toward the fall, per unit tan(roll), x COM height
    lateral_kd: float = 0.05       # same, per unit roll rate (rad/s)
    lateral_limit: float = 0.10    # m
    roll_kp: float = 1.0           # leg length difference per unit roll (x half-track)
    roll_kd: float = 0.08
    heading_kp: float = 3.0        # N m per rad of heading error (heading from wheel encoders)
    heading_kd: float = 5.0        # N m per rad/s of yaw-rate error (gyro)
    heading_err_limit: float = 0.5 # rad; beyond this the reference is dragged (after a shove)
    trim_rate: float = 0.0         # balance-point estimator gain [rad / (m s)]
    trim_limit: float = 0.15       # rad
    accel_limit: float = 1.5       # m/s^2 on the velocity reference
    x_err_limit: float = 1.0       # m; beyond this the spot is re-anchored (a large shove)
    wheel_friction_comp: float = 0.0   # N m, Coulomb friction fed forward along the wheel speed
    wheel_friction_eps: float = 0.5    # rad/s, smoothing of the sign() near zero speed
    # Deadband on the position error: inside it no position correction at all (the robot only
    # balances, so a tiny offset cannot restart a hunting oscillation); outside it the hold
    # ramps in from zero, which bounds the drift to about the band.
    hold_deadband: float = 0.02    # m
    # Parking: position/heading are only held once the robot has stopped. While driving (or
    # still slowing down) the anchor follows the robot, so it never "catches up" to a moving
    # target; it parks after |v| < park_speed for park_time.
    park_speed: float = 0.05       # m/s
    park_yaw_rate: float = 0.05    # rad/s
    park_time: float = 0.3         # s
    park_timeout: float = 1.0      # s after the command reaches zero, park regardless of speed
    hold_kp: float = 1.0           # 1/s, outer position loop: speed correction per metre of error
    hold_speed_limit: float = 0.15 # m/s, cap on that correction
    latency_comp: float = 0.005    # s, predict the balance state this far ahead
    wheel_torque_limit: float = 40.0
    Q: tuple = (4.0, 60.0, 3.0, 4.0)
    R: float = 0.2
    # Manual balance gains (manual_K = 1 bypasses the LQR). Wheel torque =
    # -(K_pos e_x + K_lean phi + K_vel e_v + K_lean_rate phidot): K_pos / K_vel are the position
    # P / D terms, K_lean / K_lean_rate the lean P / D terms. Defaults = LQR result at h = 0.70.
    manual_K: float = 0.0
    K_pos: float = 4.47
    K_lean: float = 73.5
    K_vel: float = 8.67
    K_lean_rate: float = 13.9


class ClassicalController:
    def __init__(self, gains: ClassicalGains | None = None, model: Tron1Model | None = None, control_hz=500):
        self.g = gains or ClassicalGains()
        self.model = model or Tron1Model()
        self.dt = 1.0 / control_hz
        hs = np.linspace(0.55, 0.76, 22)
        self._h_table = hs
        self._stances = np.array([self.model.stance(h) for h in hs])
        self._alpha_ref = 0.0
        self.set_height(self.g.height)
        self.reset()

    def set_pack_belief(self, mass=None, com=None):
        """What the controller assumes about the sensor pack; rebuilds the model and stance table."""
        mass = self.model.payload_mass if mass is None else float(mass)
        com = self.model.payload_pos if com is None else tuple(com)
        self.model = Tron1Model(payload_mass=mass, payload_pos=com)
        self._stances = np.array([self.model.stance(h) for h in self._h_table])
        self.set_height(self.g.height)

    def set_height(self, h):
        self.g.height = h
        hip, knee = [np.interp(h, self._h_table, self._stances[:, k]) for k in range(2)]
        self.q_stance = self.model.mirror(hip, knee)
        self.pend = self.model.pendulum(self.q_stance)
        self.K, self.A, self.B = wip_lqr(self.pend, self.g.Q, self.g.R)
        if self.g.manual_K >= 0.5:
            self.K = -np.array([[self.g.K_pos, self.g.K_lean, self.g.K_vel, self.g.K_lean_rate]])
        self.tau_ff = self.model.gravity_ff(self.q_stance)
        self.wheel_ff = self.model.wheel_reaction_ff(self.q_stance)
        self.wheels_nom = self.model.wheel_centers(self.q_stance)

    def reset(self):
        self.x = 0.0
        self.x_ref = 0.0
        self.v_ref = 0.0
        self.leg_int = np.zeros(N_MOTORS)
        self.phi_trim = 0.0
        self.psi_ref = None
        self.parked = False
        self.still_time = 0.0
        self.idle_time = 0.0
        self.t_last = None
        self.tau_bal = 0.0
        self.prev_alpha = None
        self.q_des = self.q_stance.copy()

    # --------------------------------------------------------------------------------------------
    def step(self, st: RobotState, imu: ImuData, cmd_vel, t) -> RobotCmd:
        g = self.g
        # Use the measured time since the last call (the real loop does not tick at exactly
        # control_hz); fall back to the nominal period on the first call or a bad clock.
        dt = self.dt if self.t_last is None else float(np.clip(t - self.t_last, 0.2 * self.dt, 0.05))
        self.t_last = t
        R = quat_to_mat(imu.quat)
        pitch = float(np.arcsin(np.clip(-R[2, 0], -1, 1)))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        w_world = R @ imu.gyro
        c, s = np.cos(yaw), np.sin(yaw)
        pitch_rate = -s * w_world[0] + c * w_world[1]
        roll_rate = c * w_world[0] + s * w_world[1]
        yaw_rate = w_world[2]

        # --- balance state from encoders + IMU
        alpha = self.model.lean_angle(np.asarray(st.q))
        alpha_rate = 0.0 if self.prev_alpha is None else (alpha - self.prev_alpha) / dt
        self.prev_alpha = alpha
        phi = pitch + alpha
        phi_rate = pitch_rate + alpha_rate
        v = WHEEL_RADIUS * (float(np.mean(np.asarray(st.dq)[WHEEL])) + pitch_rate)
        self.x += v * dt

        vx_cmd, _, wz_cmd = cmd_vel
        dv = np.clip(vx_cmd - self.v_ref, -g.accel_limit * dt, g.accel_limit * dt)
        self.v_ref += dv
        commanded = abs(vx_cmd) > 1e-3 or abs(wz_cmd) > 1e-3 or abs(self.v_ref) > 1e-3
        if commanded:
            self.parked = False
            self.still_time = 0.0
            self.idle_time = 0.0
        elif not self.parked:
            # Park once it has settled, or after park_timeout regardless: a robot whose balance
            # point is off creeps on its own and would otherwise never count as settled.
            slow = abs(v) < g.park_speed and abs(yaw_rate) < g.park_yaw_rate
            self.still_time = self.still_time + dt if slow else 0.0
            self.idle_time += dt
            self.parked = self.still_time >= g.park_time or self.idle_time >= g.park_timeout
        if not self.parked:
            self.x_ref = self.x                       # track speed only; anchor follows
        elif abs(self.x - self.x_ref) > g.x_err_limit:
            self.x_ref = self.x - np.sign(self.x - self.x_ref) * g.x_err_limit

        # Balance-point estimator: a COM offset, payload or IMU mounting error moves the true
        # balance angle away from the model's; the LQR then needs a standing position error to
        # hold it, and the x_ref clamp turns that into drift. Integrating the odometry position
        # error into a pitch trim moves the set-point to the true balance angle instead.
        self.phi_trim = float(np.clip(self.phi_trim - g.trim_rate * (self.x - self.x_ref) * dt,
                                      -g.trim_limit, g.trim_limit))
        # Outer position loop (cascade): the encoder position error shifts the speed target of
        # the balance LQR, capped, so holding the spot never overrides balancing.
        e_x = self.x - self.x_ref
        e_x = float(np.sign(e_x) * max(abs(e_x) - g.hold_deadband, 0.0))
        v_hold = float(np.clip(-g.hold_kp * e_x, -g.hold_speed_limit, g.hold_speed_limit))
        s4 = np.array([e_x, phi - self.phi_trim, v - self.v_ref - v_hold, phi_rate])
        if g.latency_comp > 0:
            s4 = s4 + g.latency_comp * (self.A @ s4 + self.B[:, 0] * self.tau_bal)
        self.tau_bal = float(np.clip(-(self.K @ s4)[0], -2 * g.wheel_torque_limit, 2 * g.wheel_torque_limit))

        # --- heading
        # --- heading hold from the wheel encoders (no gyro drift; wrong only if the tyres slip):
        # psi = r (theta_R - theta_L) / track; body pitch cancels in the difference.
        qw = np.asarray(st.q)[WHEEL]
        psi = WHEEL_RADIUS * (qw[1] - qw[0]) / self.pend["track"]
        if self.psi_ref is None:
            self.psi_ref = psi
        if not self.parked:
            self.psi_ref = psi                        # yaw-rate control only; heading follows
        e_psi = self.psi_ref - psi
        if abs(e_psi) > g.heading_err_limit:
            self.psi_ref = psi + np.sign(e_psi) * g.heading_err_limit
            e_psi = self.psi_ref - psi
        tau_yaw = g.heading_kp * e_psi + g.heading_kd * (wz_cmd - yaw_rate)

        tau_l = np.clip(self.tau_bal / 2 - tau_yaw / 2, -g.wheel_torque_limit, g.wheel_torque_limit)
        tau_r = np.clip(self.tau_bal / 2 + tau_yaw / 2, -g.wheel_torque_limit, g.wheel_torque_limit)

        # --- legs: level the base in roll by per-leg IK, hold with PD + feedforward + integral
        half_track = self.pend["track"] / 2
        dz = np.clip(g.roll_kp * half_track * np.tan(roll) + g.roll_kd * half_track * roll_rate, -0.08, 0.08)
        # Sideways "step": shift both wheels toward the side the robot is tipping to (positive
        # roll = left side up = falling to -y), which puts the support back under the COM.
        dy = -np.clip((g.lateral_kp * np.tan(roll) + g.lateral_kd * roll_rate) * self.pend["l"],
                      -g.lateral_limit, g.lateral_limit)
        q_des = self.q_stance.copy()
        if abs(dz) > 1e-4 or abs(dy) > 1e-4:
            for side, sgn in ((0, 1.0), (1, -1.0)):
                tgt = self.wheels_nom[side] + np.array([0.0, dy, sgn * dz])
                q_des = self.model.leg_ik(q_des, side, tgt, iters=3)
        self.q_des = q_des
        q = np.asarray(st.q)
        e = q_des - q
        self.leg_int[LEG] = np.clip(self.leg_int[LEG] + g.leg_ki * e[LEG] * dt, -g.leg_ki_limit, g.leg_ki_limit)

        cmd = RobotCmd(stamp=int(t * 1e9))
        cmd.q[LEG] = q_des[LEG]
        cmd.Kp[LEG] = g.leg_kp
        cmd.Kd[LEG] = g.leg_kd
        cmd.Kp[[0, 4]] = g.abad_kp
        cmd.Kd[[0, 4]] = g.abad_kd
        q_tilt = quat_from_rpy(roll, pitch, 0.0)
        tau_legs = self.model.gravity_ff(q, q_tilt) + self.wheel_ff @ np.array([-tau_yaw / 2, tau_yaw / 2])
        cmd.tau[LEG] = tau_legs[LEG] + self.leg_int[LEG]
        dq_w = np.asarray(st.dq)[WHEEL]
        cmd.tau[WHEEL] = np.array([tau_l, tau_r]) + g.wheel_friction_comp * np.tanh(dq_w / g.wheel_friction_eps)
        return cmd
