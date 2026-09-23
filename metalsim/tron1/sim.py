"""TRON1 wheel-foot simulator that speaks the LimX SDK message types, with a realistic hardware layer.

Layers between a controller and the MuJoCo physics (`assets/tron1/WF_TRON1A/xml/robot.xml`, LimX's
official model), all parameterized by `metalsim.tron1.realism.SimParams`:

1. transport: state and IMU messages published at 1 kHz reach the controller after `sensor_delay`
   (+ exponential jitter); commands reach the drives after `cmd_delay` (+ jitter) and may be lost.
   Drives hold the newest command received.
2. drives (2 kHz here): mode-0 law `Kp (q - q_enc) + Kd (dq - dq_enc) + tau` evaluated on the
   *quantized* encoder angle and its finite-difference velocity, clipped to the torque ceilings,
   scaled by motor strength, limited by the torque-speed envelope, then a first-order current lag.
3. joints: Coulomb friction (MuJoCo frictionloss), viscous damping and rotor armature per joint.
4. sensors: encoder quantization, velocity from 1 kHz differences (the robot's EC-Master config has
   `qvel_alpha: 1`, i.e. no master-side filtering), torque *estimate* with a torque-constant error
   and noise, IMU with white noise, bias and bias random walk, fixed mounting misalignment, and an
   AHRS orientation with wandering tilt error and yaw drift.
5. body and ground: payload (sensor pack), base COM offset, link mass and inertia scaling, ground
   friction and tyre softness; optional terrain height field (`metalsim.tron1.scene`).

Physics runs at 0.5 ms (LimX's model file uses 1 ms). Time is kept in integer microseconds.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .realism import SimParams
from .sdk import N_MOTORS, WF_MOTOR_NAMES, WHEEL, ImuData, RobotCmd, RobotState

ROOT = Path(__file__).resolve().parents[2]
WF_XML = ROOT / "assets/tron1/WF_TRON1A/xml/robot.xml"
WHEEL_RADIUS = 0.127
PHYS_DT_US = 500
SDK_DT_US = 1000


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def build_model(params: SimParams, terrain=None) -> mujoco.MjModel:
    """LimX's WF_TRON1A model with the body/ground/joint parts of `params` applied."""
    spec = mujoco.MjSpec.from_file(str(WF_XML))
    spec.option.timestep = PHYS_DT_US * 1e-6
    base = spec.body("base_Link")
    for b in spec.bodies:
        if b.name in ("world", ""):
            continue
        b.mass *= params.link_mass_scale
        b.inertia = np.asarray(b.inertia) * params.inertia_scale
    base.ipos = np.asarray(base.ipos) + params.base_com_offset
    if params.payload_mass > 0:
        pay = base.add_body(name="payload", pos=params.payload_pos)
        pay.mass = params.payload_mass
        pay.inertia = [params.payload_mass * 0.01] * 3
        pay.explicitinertial = True
    for i, name in enumerate(WF_MOTOR_NAMES):
        j = spec.joint(name)
        j.frictionloss = float(params.coulomb[i])
        j.damping = [float(params.viscous[i]), 0.0, 0.0]
        j.armature = float(params.armature[i])
    for a in spec.actuators:
        a.ctrlrange = [-500, 500]
    floor = spec.geom("floor")
    floor.friction = [params.ground_friction, 0.005, 0.0001]
    for side in "LR":
        g = spec.geom(f"wheel_{side}_collision")
        g.friction = [params.ground_friction, 0.005, 0.0001]
        g.solref = [params.wheel_softness, 1.0]
        g.condim = 4                     # torsional friction for turning in place
        g.size = [WHEEL_RADIUS, params.tyre_half_width, 0.0]
    if terrain is not None:
        terrain.attach(spec)
    return spec.compile()


@dataclass
class _Msg:
    t_us: int
    seq: int


class Tron1Sim:
    """One simulated TRON1 WF. Drive it with `tick(cmd_or_None)` or `run(controller, ...)`."""

    def __init__(self, params: SimParams | None = None, seed: int = 0, terrain=None):
        self.p = params if params is not None else SimParams.nominal()
        self.rng = np.random.default_rng(seed)
        self.terrain = terrain
        self.m = build_model(self.p, terrain)
        self.d = mujoco.MjData(self.m)
        self.jq = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in WF_MOTOR_NAMES])
        self.jv = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in WF_MOTOR_NAMES])
        self.base_id = self.m.body("base_Link").id
        self.wheel_ids = [self.m.body(f"wheel_{s}_Link").id for s in "LR"]
        self._sens = {n: self.m.sensor_adr[self.m.sensor(n).id] for n in ("quat", "gyro", "acc")}
        self.total_mass = float(self.m.body_subtreemass[self.base_id])
        self._fall_geoms = {self.m.geom(g).id for g in ("base_collision", "abad_L_collision", "abad_R_collision",
                                                         "hip_L_collision", "hip_R_collision", "knee_L_collision",
                                                         "knee_R_collision")}
        self._ground_geoms = {self.m.geom("floor").id}
        if terrain is not None:
            self._ground_geoms |= set(terrain.geom_ids(self.m))
        self.mount_q = quat_from_rpy(*self.p.imu_mount_rpy)
        self.mount_R = quat_to_mat(self.mount_q)

    # ------------------------------------------------------------------ reset
    def reset(self, q_legs: np.ndarray, xy=(0.0, 0.0), yaw=0.0, pitch=0.0):
        """Place the robot on the ground in joint configuration `q_legs` (8 values, wheels ignored)."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.jq] = q_legs
        d.qpos[self.jq[WHEEL]] = 0.0
        d.qpos[3:7] = quat_from_rpy(0.0, pitch, yaw)
        d.qpos[0:2] = xy
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(m, d)
        low = min(d.xpos[w][2] for w in self.wheel_ids) - WHEEL_RADIUS
        ground = 0.0 if self.terrain is None else self.terrain.height_at(*d.qpos[:2])
        d.qpos[2] += ground - low + 0.002
        mujoco.mj_forward(m, d)

        self.t_us = 0
        self._seq = 0
        self._cmd_queue: list = []            # (arrival_us, seq, RobotCmd)
        self._state_queue: list = []          # (arrival_us, seq, RobotState, ImuData)
        self._active_cmd = RobotCmd.damping(0.0)
        self._active_seq = -1
        self.latest_state: RobotState | None = None
        self.latest_imu: ImuData | None = None
        self.tau_motor = np.zeros(N_MOTORS)
        self.q_enc = self._quantize(d.qpos[self.jq])
        self._q_enc_prev_1k = self.q_enc.copy()
        self.dq_enc = np.zeros(N_MOTORS)
        self.gyro_bias = self.p.gyro_bias.copy()
        self.ahrs_err = np.zeros(3)           # roll, pitch, yaw error of the onboard AHRS
        self.push_force = np.zeros(3)
        self.push_until_us = -1
        self.fallen = False
        self._publish()                       # a first state so controllers have something
        _, _, self.latest_state, self.latest_imu = heapq.heappop(self._state_queue)
        return self

    # ------------------------------------------------------------------ helpers
    def _quantize(self, q):
        if self.p.encoder_bits <= 0:
            return q.copy()
        step = 2 * np.pi / (1 << self.p.encoder_bits)
        return np.round(q / step) * step

    def _delay_us(self, base):
        extra = self.rng.exponential(self.p.jitter) if self.p.jitter > 0 else 0.0
        return int(round((base + extra) * 1e6))

    def _publish(self):
        """Drive + IMU publish at 1 kHz (called on the 1 ms boundary)."""
        p, d = self.p, self.d
        st = RobotState(stamp=self.t_us * 1000)
        st.q = self.q_enc.copy()
        st.dq = self.dq_enc.copy()
        st.tau = self.tau_motor * (1.0 + p.tau_gain_err)
        if p.tau_noise > 0:
            st.tau = st.tau + self.rng.normal(0.0, p.tau_noise, N_MOTORS)

        s = self._sens
        q_true = d.sensordata[s["quat"]:s["quat"] + 4].copy()
        gyro = d.sensordata[s["gyro"]:s["gyro"] + 3].copy()
        acc = d.sensordata[s["acc"]:s["acc"] + 3].copy()
        dt = SDK_DT_US * 1e-6
        if p.gyro_bias_walk > 0:
            self.gyro_bias += self.rng.normal(0.0, p.gyro_bias_walk * np.sqrt(dt), 3)
        imu = ImuData(stamp=self.t_us * 1000)
        imu.gyro = self.mount_R.T @ gyro + self.gyro_bias + (self.rng.normal(0, p.gyro_noise, 3) if p.gyro_noise else 0)
        imu.acc = self.mount_R.T @ acc + p.acc_bias + (self.rng.normal(0, p.acc_noise, 3) if p.acc_noise else 0)
        if p.ahrs_tilt_wander > 0:           # Ornstein-Uhlenbeck, 5 s correlation
            a = dt / 5.0
            self.ahrs_err[:2] += -a * self.ahrs_err[:2] + p.ahrs_tilt_wander * np.sqrt(2 * a) * self.rng.normal(size=2)
        if p.ahrs_yaw_walk > 0:
            self.ahrs_err[2] += p.ahrs_yaw_walk * np.sqrt(dt) * self.rng.normal()
        imu.quat = quat_mul(quat_mul(quat_from_rpy(*self.ahrs_err), q_true), self.mount_q)
        imu.quat /= np.linalg.norm(imu.quat)

        self._seq += 1
        heapq.heappush(self._state_queue, (self.t_us + self._delay_us(p.sensor_delay), self._seq, st, imu))

    def _deliver(self):
        while self._state_queue and self._state_queue[0][0] <= self.t_us:
            _, _, st, imu = heapq.heappop(self._state_queue)
            self.latest_state, self.latest_imu = st, imu
        while self._cmd_queue and self._cmd_queue[0][0] <= self.t_us:
            _, seq, cmd = heapq.heappop(self._cmd_queue)
            if seq > self._active_seq:
                self._active_cmd, self._active_seq = cmd, seq

    def send(self, cmd: RobotCmd):
        """Controller -> robot. Only mode 0 is modelled (the only mode LimX's controllers use)."""
        if np.any(np.asarray(cmd.mode) != 0):
            raise NotImplementedError("only SDK mode 0 (torque-position hybrid) is modelled")
        if self.p.packet_loss > 0 and self.rng.random() < self.p.packet_loss:
            return
        self._seq += 1
        heapq.heappush(self._cmd_queue, (self.t_us + self._delay_us(self.p.cmd_delay), self._seq, cmd.copy()))

    def push(self, dv_xy, duration=0.1):
        """Horizontal shove on the base that changes the robot's momentum by total_mass * dv."""
        f = self.total_mass * np.asarray(dv_xy, dtype=float) / duration
        self.push_force = np.array([f[0], f[1], 0.0])
        self.push_until_us = self.t_us + int(duration * 1e6)

    def _drive(self):
        p, d, c = self.p, self.d, self._active_cmd
        tau = p.kp_scale * c.Kp * (c.q - self.q_enc) + p.kd_scale * c.Kd * (c.dq - self.dq_enc) + c.tau
        tau = np.clip(tau, -p.torque_limit, p.torque_limit) * p.strength
        # Torque-speed envelope (only limits motoring; braking keeps full torque).
        w = d.qvel[self.jv]
        avail = p.torque_limit * p.strength * np.clip((p.speed_no_load - np.abs(w)) / (p.speed_no_load - p.speed_knee), 0.0, 1.0)
        motoring = tau * w > 0
        tau = np.where(motoring, np.clip(tau, -avail, avail), tau)
        dt = PHYS_DT_US * 1e-6
        a = dt / (p.current_lag + dt) if p.current_lag > 0 else 1.0
        self.tau_motor += a * (tau - self.tau_motor)
        d.ctrl[:] = self.tau_motor

    # ------------------------------------------------------------------ stepping
    def tick(self):
        """Advance 1 ms (two physics steps); publish state/IMU; deliver due messages."""
        d = self.d
        for _ in range(SDK_DT_US // PHYS_DT_US):
            self._deliver()
            self.q_enc = self._quantize(d.qpos[self.jq])
            if self.p.encoder_bits <= 0:        # ideal sensing: exact velocity, as LimX's simulator
                self.dq_enc = d.qvel[self.jv].copy()
            self._drive()
            d.xfrc_applied[self.base_id, :3] = self.push_force if self.t_us < self.push_until_us else 0.0
            mujoco.mj_step(self.m, d)
            self.t_us += PHYS_DT_US
        q_now = self._quantize(d.qpos[self.jq])
        self.dq_enc = ((q_now - self._q_enc_prev_1k) / (SDK_DT_US * 1e-6) if self.p.encoder_bits > 0
                       else d.qvel[self.jv].copy())
        self._q_enc_prev_1k = q_now
        self.q_enc = q_now
        self._publish()
        self._deliver()
        self._check_fall()

    def _check_fall(self):
        d = self.d
        up = quat_to_mat(d.qpos[3:7])[:, 2]
        if up[2] < np.cos(np.radians(70)):
            self.fallen = True
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 in self._fall_geoms and g2 in self._ground_geoms) or (g2 in self._fall_geoms and g1 in self._ground_geoms):
                self.fallen = True

    # ------------------------------------------------------------------ truth, for evaluation only
    def truth(self):
        d = self.d
        R = quat_to_mat(d.qpos[3:7])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v_world = d.qvel[0:3]
        c, s = np.cos(yaw), np.sin(yaw)
        v_head = np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]])
        pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
        roll = np.arctan2(R[2, 1], R[2, 2])
        return dict(t=self.t_us * 1e-6, pos=d.qpos[0:3].copy(), yaw=yaw, pitch=pitch, roll=roll,
                    v_head=v_head, yaw_rate=float(d.qvel[5]), tau=self.tau_motor.copy())

    def run(self, controller, duration, control_hz=500, commands=None, pushes=(), log_every_ms=10):
        """Closed loop: `controller.step(state, imu, cmd_vel, t) -> RobotCmd` at `control_hz`.

        `commands(t) -> (vx, vy, wz)`; `pushes` is a list of (t, (dvx, dvy)). Returns a log dict.
        """
        period = int(round(1000 / control_hz))
        pushes = sorted(pushes)
        log = []
        n_ms = int(round(duration * 1000))
        for k in range(n_ms):
            t = self.t_us * 1e-6
            while pushes and pushes[0][0] <= t:
                self.push(pushes.pop(0)[1])
            vel = commands(t) if commands is not None else (0.0, 0.0, 0.0)
            if k % period == 0:
                self.send(controller.step(self.latest_state, self.latest_imu, vel, t))
            self.tick()
            if k % log_every_ms == 0:
                tr = self.truth()
                tr["cmd"] = np.array(vel)
                log.append(tr)
            if self.fallen:
                break
        out = {key: np.array([r[key] for r in log]) for key in log[0]}
        out["fallen"] = self.fallen
        out["t_end"] = self.t_us * 1e-6
        return out
