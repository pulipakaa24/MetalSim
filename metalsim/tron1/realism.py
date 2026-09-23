"""Realism parameters for the TRON1 wheel-foot simulation: actuators, latency, sensors, body, ground.

Every parameter carries its provenance, because most of what the real robot does below the SDK is
not published:

* [robot] read from this robot's motion controller (EC-Master config at :8080, 2026-09-23)
* [limx]  LimX's model files or SDK (tron1-robot-description, limxsdk-lowlevel datatypes.h)
* [limx-dr] LimX's own training randomization (tron1-rl-isaacgym wheelfoot_flat_config.py)
* [est]   engineering estimate for this class of hardware; not measured on TRON1

`SimParams.ideal()` reproduces LimX's `tron1-mujoco-sim` (perfect sensors, no delay, ideal
torque). `SimParams.nominal()` is the best single guess of the real robot. `sample(rng)` draws one
robot from the randomization ranges, which are at least as wide as LimX's training ranges.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .sdk import N_MOTORS

LEG_MASK = np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)


def _per_motor(leg, wheel):
    return np.where(LEG_MASK, leg, wheel).astype(float)


@dataclass
class SimParams:
    # --- actuators -------------------------------------------------------------------------------
    # Torque ceilings: 80 N m legs / 40 N m wheels [limx: URDF effort, MJCF ctrlrange, SDK guide].
    # LimX's current spec page says 60 N m peak for the legs; `strength` covers that down to 0.75.
    # The drive's current limit (35 A x Kt, Kt = 2.55-2.63 legs / 1.2 wheels [robot]) is 89-92 / 42
    # N m, above these, so the software ceilings are the binding ones (inferred from units).
    torque_limit: np.ndarray = field(default_factory=lambda: _per_motor(80.0, 40.0))
    strength: np.ndarray = field(default_factory=lambda: np.ones(N_MOTORS))   # [limx-dr] 0.8-1.2
    kp_scale: np.ndarray = field(default_factory=lambda: np.ones(N_MOTORS))   # [limx-dr] 0.8-1.2
    kd_scale: np.ndarray = field(default_factory=lambda: np.ones(N_MOTORS))   # [limx-dr] 0.8-1.2
    # Torque-speed envelope when motoring: full torque below `knee`, falling linearly to zero at
    # `no_load` [est]. Leg velocity limit 15 rad/s and wheel 40 rad/s are LimX's rated speeds [limx].
    speed_knee: np.ndarray = field(default_factory=lambda: _per_motor(12.0, 35.0))
    speed_no_load: np.ndarray = field(default_factory=lambda: _per_motor(22.0, 55.0))
    current_lag: float = 0.0005          # s, current-loop first-order lag [est] 0.2-1 ms
    coulomb: np.ndarray = field(default_factory=lambda: _per_motor(0.01, 0.01))   # N m [limx MJCF] / [est] below
    viscous: np.ndarray = field(default_factory=lambda: _per_motor(0.01, 0.01))   # N m s/rad [limx MJCF]
    armature: np.ndarray = field(default_factory=lambda: _per_motor(0.01, 0.01))  # kg m^2 [limx MJCF]

    # --- latency (seconds) -----------------------------------------------------------------------
    cmd_delay: float = 0.0               # controller -> drive
    sensor_delay: float = 0.0            # drive/IMU -> controller
    jitter: float = 0.0                  # mean of an exponential extra delay per message
    packet_loss: float = 0.0             # probability a command message is dropped

    # --- sensors ---------------------------------------------------------------------------------
    encoder_bits: int = 0                # 0 = perfect; else output-side resolution 2 pi / 2^bits
    tau_gain_err: np.ndarray = field(default_factory=lambda: np.zeros(N_MOTORS))   # Kt error
    tau_noise: float = 0.0               # N m
    gyro_noise: float = 0.0              # rad/s white, per 1 kHz sample
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias_walk: float = 0.0          # rad/s/sqrt(s)
    acc_noise: float = 0.0               # m/s^2
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_mount_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad, fixed misalignment
    ahrs_tilt_wander: float = 0.0        # rad, std of the AHRS roll/pitch error (OU process, 5 s)
    ahrs_yaw_walk: float = 0.0           # rad/sqrt(s)

    # --- body and ground -------------------------------------------------------------------------
    payload_mass: float = 0.0            # kg on top of the base (the Jetson sensor pack)
    payload_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.05]))
    base_com_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    link_mass_scale: float = 1.0
    inertia_scale: float = 1.0
    ground_friction: float = 0.6         # [limx MJCF]
    wheel_softness: float = 0.02         # MuJoCo solref time constant of the tyre contact [est]
    # Tyre width: LimX trains on a 5 cm wide wheel cylinder (tron1-rl-isaacgym WF_TRON1A URDF);
    # their MuJoCo file has a 1 cm disc, on which the robot has almost no roll stiffness.
    tyre_half_width: float = 0.025

    @staticmethod
    def ideal() -> "SimParams":
        """LimX's tron1-mujoco-sim: exact state, zero delay, ideal torque source, 1 cm tyre disc."""
        return SimParams(speed_knee=np.full(N_MOTORS, 1e6), speed_no_load=np.full(N_MOTORS, 2e6),
                         current_lag=0.0, wheel_softness=0.02, tyre_half_width=0.005)

    @staticmethod
    def nominal() -> "SimParams":
        """Best single guess of the real robot (with its sensor pack)."""
        return SimParams(
            # Joint friction of a geared quasi-direct-drive leg actuator; wheel hub motor [est].
            coulomb=_per_motor(0.3, 0.1), viscous=_per_motor(0.03, 0.01),
            armature=_per_motor(0.012, 0.008),
            cmd_delay=0.004, sensor_delay=0.001, jitter=0.0005, packet_loss=0.002,
            # Motor-side encoder seen through the gearbox: ~17 bits at the joint [est].
            encoder_bits=17, tau_noise=0.3,
            # MEMS IMU of the BMI088 class sampled at 1 kHz [est]: 0.014 deg/s/rtHz gyro noise
            # density, 175 ug/rtHz accel, ~0.3 deg/s turn-on bias.
            gyro_noise=0.0055, gyro_bias_walk=1e-4, acc_noise=0.04,
            ahrs_tilt_wander=np.radians(0.2), ahrs_yaw_walk=np.radians(0.05),
            # Sensor pack: Jetson Orin NX dev kit, Livox MID-360, mount and cabling [est].
            payload_mass=1.5, payload_pos=np.array([0.0, 0.0, 0.08]),
            ground_friction=0.8)

    def sample(self, rng: np.random.Generator) -> "SimParams":
        """One randomized robot centred on these parameters."""
        u = lambda lo, hi, n=None: rng.uniform(lo, hi, n)
        p = replace(self)
        p.strength = u(0.75, 1.1, N_MOTORS)          # 0.75 covers the 60 N m spec-page figure
        p.kp_scale = u(0.8, 1.2, N_MOTORS)
        p.kd_scale = u(0.8, 1.2, N_MOTORS)
        p.speed_knee = self.speed_knee * u(0.8, 1.2)
        p.speed_no_load = np.maximum(self.speed_no_load * u(0.8, 1.2), p.speed_knee * 1.2)
        p.current_lag = u(0.0002, 0.001)
        p.coulomb = self.coulomb * u(0.3, 2.5, N_MOTORS)
        p.viscous = self.viscous * u(0.5, 2.0, N_MOTORS)
        p.armature = self.armature * u(0.5, 2.0, N_MOTORS)
        # LimX trains with 0-20 ms action delay, but their policy (which runs on the real robot)
        # falls in this sim from 15 ms of command delay on and never up to 10 ms
        # (docs/TRON1.md), so the real command path is taken to be under ~10 ms.
        p.cmd_delay = u(0.0, 0.010)
        p.sensor_delay = u(0.0005, 0.003)
        p.jitter = u(0.0, 0.001)
        p.packet_loss = u(0.0, 0.01)
        p.encoder_bits = int(rng.integers(14, 19))
        p.tau_gain_err = u(-0.1, 0.1, N_MOTORS)
        p.tau_noise = u(0.1, 0.6)
        p.gyro_noise = self.gyro_noise * u(0.5, 2.0)
        p.gyro_bias = rng.normal(0.0, np.radians(0.3), 3)
        p.acc_noise = self.acc_noise * u(0.5, 2.0)
        p.acc_bias = rng.normal(0.0, 0.05, 3)
        p.imu_mount_rpy = np.array([u(-1, 1), u(-1, 1), 0.0]) * np.radians(1.2)   # [limx-dr]
        p.payload_mass = u(0.5, 3.0)
        p.payload_pos = np.array([u(-0.05, 0.05), u(-0.03, 0.03), u(0.03, 0.12)])
        p.base_com_offset = np.array([u(-0.03, 0.03), u(-0.02, 0.02), u(-0.03, 0.03)])   # [limx-dr]
        p.link_mass_scale = u(0.9, 1.1)
        p.inertia_scale = u(0.8, 1.2)                # [limx-dr]
        p.ground_friction = u(0.3, 1.2)              # LimX trains 0.2-1.6; rubber on indoor floors
        p.wheel_softness = u(0.01, 0.03)
        p.tyre_half_width = u(0.015, 0.025)
        return p
