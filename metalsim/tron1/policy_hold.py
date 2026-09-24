"""LimX's open RL policy + a position/heading hold, for Developer Mode through `limx_bridge`.

In Developer Mode we own the whole command path, so unlike the stock controller (which ignores
|vx| below ~0.2 m/s, docs/TRON1.md) the hold can send small, continuous corrections. The remote's
sticks still pass through the bridge's own deadzone (`BridgeConfig.deadzone`); the hold's output
does not. While a stick is off-centre the hold stays out and its anchor follows the robot.

Presents the interface `Bridge` expects of a controller: `q_stance`, `g.leg_kp/leg_kd/abad_kp/
abad_kd` (for the STAND blend), `reset()`, `step(state, imu, cmd_vel, t) -> RobotCmd`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .limx_policy import KD, KP, LimxPolicyController
from .sdk import ImuData, RobotCmd, RobotState
from .stock_hold import HoldGains, StockHold, track_from_q

# Continuous PD (no deadzone to work around here). Gentler than the sim optimum (kp 1.5 / kd 0.4)
# for the first hardware sessions, and ramped so a correction never steps the policy's command.
POLICY_HOLD = replace(HoldGains(), kp=1.0, kd=0.3, kp_psi=1.0, kd_psi=0.1, engage=0.02, release=0.005,
                      engage_psi=np.radians(1.0), release_psi=np.radians(0.3), accel_max=0.5,
                      yaw_accel_max=1.0, v_filter=0.1, v_max=0.3, w_max=0.3)


@dataclass
class _StandGains:
    leg_kp: float = KP
    leg_kd: float = KD
    abad_kp: float = KP
    abad_kd: float = KD


class PolicyHoldController:
    """`policy`: "limx" (LimX's public WF policy) or a path to our trained ONNX
    (`metalsim.learn.tron1_wf`, run through `TrainedPolicyController`)."""

    def __init__(self, hold: bool = True, gains: HoldGains | None = None, policy: str = "limx"):
        if policy == "limx":
            self.policy = LimxPolicyController()
        else:
            from .trained_policy import TrainedPolicyController
            self.policy = TrainedPolicyController(None if policy == "trained" else policy)
        self.q_stance = self.policy.initial_q()          # the policy's default pose (q = 0.25 a, a = 0)
        self.g = _StandGains()
        self.hold_enabled = hold
        self.hold_gains = gains or POLICY_HOLD
        self.last_hold_cmd = (0.0, 0.0, 0.0)
        # Everything slow happens here, not in reset(): the bridge calls reset() inside the control
        # loop when BALANCE starts, and building the MuJoCo model for the track took 125 ms on the
        # Jetson (the stall watchdog dropped the robot to DAMP). First ONNX runs are slow too: warm up.
        self.track = track_from_q(self.q_stance)
        self._warm_up()
        self.reset()

    def _warm_up(self):
        st = RobotState(stamp=0, q=np.asarray(self.q_stance, float), dq=np.zeros(8), tau=np.zeros(8))
        imu = ImuData(stamp=0, acc=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
        for k in range(30):
            self.policy.step(st, imu, (0.0, 0.0, 0.0), k * 0.002)

    def reset(self):
        self.policy.reset()
        self.hold = StockHold(self.track, self.hold_gains, mode="pd")

    def step(self, st: RobotState, imu: ImuData, cmd_vel, t) -> RobotCmd:
        self.hold.update_odometry(st, imu, t)
        vel = tuple(cmd_vel)
        if self.hold_enabled:
            # The bridge has already applied the stick deadzone and scaling; a non-zero command
            # means the operator is driving, which is what StockHold's `sticks` test looks for.
            driving = any(abs(v) > 0.0 for v in vel)
            out = self.hold.step(t, sticks=(1.0,) if driving else (0.0,))
            if out is not None:
                vel = out
        self.last_hold_cmd = vel
        return self.policy.step(st, imu, vel, t)
