"""Our own TRON1 wheel-foot policy (trained by `metalsim.learn.tron1_wf`) on the SDK interface.

Rebuilds exactly the training observation from `RobotState` / `ImuData`: gyro * 0.25, projected
gravity (from the IMU quaternion), leg angles (default pose 0), all joint velocities * 0.05, last
actions; 10 frames oldest first, plus the command scaled (2, 2, 0.25) -> 283 inputs, one ONNX MLP
-> 8 actions at 50 Hz. Legs: q = 0.25 a with Kp 42 / Kd 2.5, the action clipped so the PD torque
stays within 80 N m (training's limit); wheels: velocity 0.5 a through Kd 0.8, within 40 N m.
Same `step(state, imu, cmd_vel, t)` interface as `LimxPolicyController`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .sdk import N_MOTORS, ImuData, RobotCmd, RobotState
from .sim import quat_to_mat

KP, KD, WHEEL_KD, LEG_TAU, WHEEL_TAU = 42.0, 2.5, 0.8, 80.0, 40.0
SCALE_POS, SCALE_VEL = 0.25, 0.5
CMD_SCALE = np.array([2.0, 2.0, 0.25])
LEGS = [0, 1, 2, 4, 5, 6]
DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "assets/tron1/metalsim_policy/policy.onnx"


class TrainedPolicyController:
    def __init__(self, path: str | Path | None = None, decimation: int = 10):
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        self.path = Path(path) if path else DEFAULT_POLICY
        self.sess = ort.InferenceSession(str(self.path), sess_options=o, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.decimation = decimation
        self.reset()

    def initial_q(self):
        return np.zeros(N_MOTORS)

    def reset(self):
        self.count = 0
        self.actions = np.zeros(8)
        self.history = None

    def _obs(self, st: RobotState, imu: ImuData):
        R = quat_to_mat(imu.quat)
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        q, dq = np.asarray(st.q, float), np.asarray(st.dq, float)
        return np.concatenate([np.asarray(imu.gyro, float) * 0.25, gravity, q[LEGS], dq * 0.05, self.actions])

    def step(self, st: RobotState, imu: ImuData, cmd_vel, t) -> RobotCmd:
        if self.count % self.decimation == 0:
            obs = self._obs(st, imu)
            self.history = np.tile(obs, 10) if self.history is None else np.concatenate([self.history[28:], obs])
            x = np.concatenate([np.clip(self.history, -100, 100), CMD_SCALE * np.asarray(cmd_vel, float)]).astype(np.float32)
            self.actions = np.clip(self.sess.run(None, {self.in_name: x[None]})[0].ravel(), -100, 100).astype(float)
        self.count += 1
        q, dq = np.asarray(st.q, float), np.asarray(st.dq, float)
        cmd = RobotCmd(stamp=int(t * 1e9))
        for i in range(N_MOTORS):
            if i in (3, 7):
                v = np.clip(SCALE_VEL * self.actions[i], dq[i] - WHEEL_TAU / WHEEL_KD, dq[i] + WHEEL_TAU / WHEEL_KD)
                cmd.dq[i], cmd.Kd[i] = v, WHEEL_KD
            else:
                target = np.clip(SCALE_POS * self.actions[i], q[i] + (KD * dq[i] - LEG_TAU) / KP, q[i] + (KD * dq[i] + LEG_TAU) / KP)
                cmd.q[i], cmd.Kp[i], cmd.Kd[i] = target, KP, KD
        return cmd
