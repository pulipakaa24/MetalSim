"""so101Sim's SO101LiftEnv (unchanged, imported) on a servo model of metalsim/render/film/so101.py, plus Isaac Lab's lift
task smoothness penalties. Used by train_smooth.py; nothing in so101Sim is modified.

Penalties, copied from Isaac Lab's lift task (IsaacLab3 source/isaaclab_tasks/isaaclab_tasks/contrib/lift/
lift_env_cfg.py, RewardsCfg ``action_rate`` and ``joint_vel``; the functions in source/isaaclab/isaaclab/envs/mdp/
rewards.py):

    action_rate_l2 = sum_i (a_t,i - a_{t-1},i)^2        over all action dimensions (6 here: five arm joints, gripper)
    joint_vel_l2   = sum_j qdot_j^2                      over all robot joints (the six SO-101 joints), after the step
    reward         = so101Sim's reward - w_rate * action_rate_l2 - w_vel * joint_vel_l2

``a_{t-1}`` at the first step of an episode: Isaac Lab zeroes ``prev_action`` at reset, and its Franka action is an
offset from the default joint positions (``use_default_offset``), so zero is the action that holds the start pose.
so101Sim's action is an absolute target in the ctrl range, so here ``a_{t-1}`` at reset is the action whose target is
the home pose (``qpos0``). The penalised action is the one the env receives (SB3 clips to [-1, 1] before ``step``).

``raw=True`` (variant 2) penalises the policy's raw output instead, as Isaac Lab does: its ActionManager stores the
action as the policy emits it (isaaclab/managers/action_manager.py ``process_action``: ``_prev_action[:] = _action;
_action[:] = action``) and rsl_rl's wrapper clips nothing by default (isaaclab_rl/rsl_rl/rl_cfg.py ``clip_actions:
float | None = None``; the lift task's Franka rsl_rl config does not set it). The env's action space becomes
[-RAW_BOUND, RAW_BOUND] so SB3 passes the Gaussian sample through unclipped; so101Sim's ``step`` still clips it to
[-1, 1] for the servo targets (so101_lift.py ``step``), so the arm sees exactly what it saw before. The control-style
statistics (``ep_act_rate``) stay on the clipped action in both modes.

The weights are set by ``set_weights`` (train_smooth.py's curriculum, as Isaac Lab's ``modify_reward_weight``).
Episode statistics for the training log go into ``info`` (read by Monitor at the episode's end): mean task reward and
penalty per step, mean |a_t - a_{t-1}|, the share of control steps each joint's servo force is at its limit (sampled
at the last physics step of each control step, ``actuator_force``) and the largest arm joint speed.
"""
import mujoco
import numpy as np
from gymnasium import spaces
from so101_lift import ARM_JOINTS, SO101LiftEnv

from metalsim.render.film.so101 import apply_servo

INFO_KEYS = ("is_success", "ep_task_r", "ep_pen", "ep_rate_l2", "ep_vel_l2", "ep_act_rate", "ep_sat_arm",
             "ep_sat_grip", "ep_peak_v", "lifted_end", "ep_raw_l2")
RAW_BOUND = 1000.0


class SmoothLiftEnv(SO101LiftEnv):
    def __init__(self, servo="7v4", w_rate=0.0, w_vel=0.0, raw=False, **kw):
        super().__init__(**kw)
        self.raw = bool(raw)
        if self.raw:
            self.action_space = spaces.Box(-RAW_BOUND, RAW_BOUND, (self._nu,), np.float32)
        self.servo = apply_servo(self.model, servo)
        self.w = [float(w_rate), float(w_vel)]
        m = self.model
        self._act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS])
        self._flim = m.actuator_forcerange[self._act, 1].copy()
        q0 = m.qpos0[self._arm_qadr]
        self._a_home = np.clip(2.0 * (q0 - self._ctrl_lo) / (self._ctrl_hi - self._ctrl_lo) - 1.0, -1.0, 1.0)
        self._prev = self._a_home.copy()
        self._prev_raw = self._a_home.copy()

    def set_weights(self, w_rate, w_vel):
        self.w = [float(w_rate), float(w_vel)]
        return self.w

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev = self._a_home.copy()
        self._prev_raw = self._a_home.copy()
        self._n, self._task, self._pen, self._rate, self._vel, self._da = 0, 0.0, 0.0, 0.0, 0.0, 0.0
        self._raw_l2 = 0.0
        self._sat = np.zeros(len(ARM_JOINTS))
        self._peak = 0.0
        return obs, info

    def step(self, action):
        u = np.asarray(action, dtype=np.float64)
        a = np.clip(u, -1.0, 1.0)
        obs, r, term, trunc, info = super().step(a)
        d = self.data
        raw_l2 = float(np.sum((u - self._prev_raw) ** 2))
        rate = raw_l2 if self.raw else float(np.sum((a - self._prev) ** 2))
        self._prev_raw = u
        self._raw_l2 += raw_l2
        qd = d.qvel[self._arm_vadr]
        vel = float(np.sum(qd ** 2))
        pen = self.w[0] * rate + self.w[1] * vel
        self._n += 1
        self._task += r
        self._pen += pen
        self._rate += rate
        self._vel += vel
        self._da += float(np.abs(a - self._prev).mean())
        self._sat += np.abs(d.actuator_force[self._act]) >= self._flim * (1 - 1e-4)
        self._peak = max(self._peak, float(np.abs(qd[:5]).max()))
        self._prev = a
        n = self._n
        info.update(ep_task_r=self._task / n, ep_pen=self._pen / n, ep_rate_l2=self._rate / n, ep_vel_l2=self._vel / n,
                    ep_act_rate=self._da / n, ep_sat_arm=float(self._sat[:5].mean() / n),
                    ep_sat_grip=float(self._sat[5] / n), ep_peak_v=self._peak, lifted_end=float(info["height"] > 0.10),
                    ep_raw_l2=self._raw_l2 / n)
        return obs, r - pen, term, trunc, info
