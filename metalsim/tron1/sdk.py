"""Mirror of the LimX low-level SDK message types for TRON1 (limxsdk-lowlevel 4.1.1).

Field names, units and orderings follow `limxsdk-lowlevel/include/limxsdk/datatypes.h`:

* `RobotCmd`: per motor `mode` (0 torque-position hybrid, 1 velocity, 2 position, 3 torque),
  `q` [rad], `dq` [rad/s], `tau` [N m], `Kp` [N m/rad], `Kd` [N m s/rad]. LimX's own controllers
  only ever send mode 0, where the drive applies `Kp (q - q_meas) + Kd (dq - dq_meas) + tau`.
* `RobotState`: per motor `q`, `dq`, and `tau` = the drive's *estimated* output torque.
* `ImuData`: `acc` [m/s^2], `gyro` [rad/s], `quat` in (w, x, y, z) order.

Motor order for the wheel-foot variant (WF_TRON1A), as in LimX's MuJoCo simulator, where
`q[i] = qpos[7 + i]` of `robot.xml`: abad_L, hip_L, knee_L, wheel_L, abad_R, hip_R, knee_R, wheel_R.

A controller written against these types (`step(state, imu, cmd_vel, t) -> RobotCmd`) runs in
`metalsim.tron1.sim.Tron1Sim`; on the robot the same call maps 1:1 onto limxsdk's
`subscribeRobotState` / `subscribeImuData` / `publishRobotCmd` (no bridge written yet).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WF_MOTOR_NAMES = ("abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
                  "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint")
N_MOTORS = len(WF_MOTOR_NAMES)
LEG = np.array([0, 1, 2, 4, 5, 6])   # abad, hip, knee (L then R)
WHEEL = np.array([3, 7])
MODE_HYBRID = 0


def _zeros():
    return np.zeros(N_MOTORS)


@dataclass
class RobotCmd:
    stamp: int = 0
    mode: np.ndarray = field(default_factory=lambda: np.zeros(N_MOTORS, dtype=np.uint8))
    q: np.ndarray = field(default_factory=_zeros)
    dq: np.ndarray = field(default_factory=_zeros)
    tau: np.ndarray = field(default_factory=_zeros)
    Kp: np.ndarray = field(default_factory=_zeros)
    Kd: np.ndarray = field(default_factory=_zeros)

    @staticmethod
    def damping(kd: float = 1.0) -> "RobotCmd":
        """What LimX's controllers send on stop: Kp = 0, Kd = 1 on every motor."""
        c = RobotCmd()
        c.Kd[:] = kd
        return c

    def copy(self) -> "RobotCmd":
        return RobotCmd(self.stamp, self.mode.copy(), self.q.copy(), self.dq.copy(),
                        self.tau.copy(), self.Kp.copy(), self.Kd.copy())


@dataclass
class RobotState:
    stamp: int = 0
    q: np.ndarray = field(default_factory=_zeros)
    dq: np.ndarray = field(default_factory=_zeros)
    tau: np.ndarray = field(default_factory=_zeros)


@dataclass
class ImuData:
    stamp: int = 0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
