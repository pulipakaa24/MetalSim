"""LimX's open-source RL controller for WF_TRON1A, as a baseline in `Tron1Sim`.

A faithful port of `tron1-rl-deploy-python/controllers/WheelfootController.py` (commit 035e4c9,
2026-09-01) with the Isaac Gym policy (`assets/tron1/limx_policy/WF_TRON1A/policy/isaacgym`, the
default the deploy script loads). Observations (28): gyro x 0.25, projected gravity, leg joint
angles (6), all joint velocities x 0.05 (8), last actions (8); history of 10 frames into
`encoder.onnx` (-> 3), then `policy.onnx` on [encoder 3, obs 28, scaled commands 3] -> 8 actions.
Policy at 50 Hz (every 10th call of a 500 Hz loop); legs `q = 0.25 a` with Kp 42 / Kd 2.5 and the
80 N m torque-bound action clip; wheels velocity `0.8 a` via Kd 0.8 with the 40 N m bound.

Differences from the deploy script, on purpose: velocity commands are scaled as in training (see
CMD_SCALE), and the 1 s stand-up blend is skipped because the sim starts the robot standing in the
policy's default pose.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .sdk import N_MOTORS, ImuData, RobotCmd, RobotState
from .sim import quat_to_mat

POLICY_DIR = Path(__file__).resolve().parents[2] / "assets/tron1/limx_policy/WF_TRON1A/policy"
JOINT_POS_IDX = [0, 1, 2, 4, 5, 6]
KP, KD, WHEEL_KD, WHEEL_TAU, USER_TAU, SCALE = 42.0, 2.5, 0.8, 40.0, 80.0, 0.25
# Training feeds commands as (vx, vy, wz) * (obs_scales.lin_vel 2.0, 2.0, ang_vel 0.25)
# (tron1-rl-isaacgym base_task.py commands_scale). The deploy script instead sends joystick x 0.5
# x user_cmd_scales (1.5, 1.0, 0.5), which does not map to the same velocities; we use training's.
CMD_SCALE = np.array([2.0, 2.0, 0.25])


def _session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


class LimxPolicyController:
    def __init__(self, variant="isaacgym", decimation=10):
        self.policy = _session(POLICY_DIR / variant / "policy.onnx")
        self.encoder = _session(POLICY_DIR / variant / "encoder.onnx")
        if variant != "isaacgym":
            raise NotImplementedError("the isaaclab export uses a different joint order; not ported")
        self.decimation = decimation
        self.reset()

    def initial_q(self):
        return np.zeros(N_MOTORS)

    def reset(self):
        self.count = 0
        self.actions = np.zeros(8)
        self.last_actions = np.zeros(8)
        self.history = None

    def _obs(self, st: RobotState, imu: ImuData):
        R = quat_to_mat(imu.quat)                       # body -> world
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        q, dq = np.asarray(st.q), np.asarray(st.dq)
        return np.concatenate([np.asarray(imu.gyro) * 0.25, gravity, q[JOINT_POS_IDX], dq * 0.05, self.last_actions])

    def step(self, st: RobotState, imu: ImuData, cmd_vel, t) -> RobotCmd:
        if self.count % self.decimation == 0:
            obs = self._obs(st, imu)
            if self.history is None:
                self.history = np.tile(obs, 10)
            self.history = np.concatenate([self.history[28:], obs])
            enc = self.encoder.run(None, {self.encoder.get_inputs()[0].name: self.history.astype(np.float32)})[0].ravel()
            commands = CMD_SCALE * np.asarray(cmd_vel, dtype=float)
            x = np.concatenate([enc, np.clip(obs, -100, 100), commands]).astype(np.float32)
            self.actions = np.clip(self.policy.run(None, {self.policy.get_inputs()[0].name: x})[0].ravel(), -100, 100)
        self.count += 1

        q, dq = np.asarray(st.q), np.asarray(st.dq)
        cmd = RobotCmd(stamp=int(t * 1e9))
        for i in range(N_MOTORS):
            if (i + 1) % 4 != 0:
                lo = (q[i] + (KD * dq[i] - USER_TAU) / KP) / SCALE
                hi = (q[i] + (KD * dq[i] + USER_TAU) / KP) / SCALE
                self.actions[i] = np.clip(self.actions[i], lo, hi)
                cmd.q[i], cmd.Kp[i], cmd.Kd[i] = self.actions[i] * SCALE, KP, KD
                self.last_actions[i] = self.actions[i]
            else:
                lo = (dq[i] - WHEEL_TAU / WHEEL_KD) / WHEEL_KD
                hi = (dq[i] + WHEEL_TAU / WHEEL_KD) / WHEEL_KD
                self.last_actions[i] = self.actions[i]
                self.actions[i] = np.clip(self.actions[i], lo, hi)
                cmd.dq[i], cmd.Kd[i] = self.actions[i] * WHEEL_KD, WHEEL_KD
        return cmd
