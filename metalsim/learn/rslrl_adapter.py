"""rsl_rl (3.x) VecEnv adapter for the G1 velocity task: lets the reference learner (the actual
``rsl_rl`` library, as Isaac Lab runs it) train on MetalSim physics, for a learner differential
against ``metalsim.learn.ppo_warp``.

One ``step(actions)`` runs the same kernel sequence as ``g1_preflight`` / ``benchmark_step``
(apply action, physics substeps, reward/termination/reset, observation), captured once as a Warp
graph and replayed each step:

    torch actions -> action buffer | signal torch -> Warp | [bump step counter, apply_action,
    physics, reward_done_reset, obs] | signal Warp -> torch | clone obs / reward / done

The observation returned is computed *after* the finished envs were reset, the ordering PPOWarp
and Isaac Lab's ``ManagerBasedRLEnv.step`` use (reward/termination -> reset -> command update ->
observation). The step counter is free-running (never rewound), so observation noise, command
resampling and reset randomization draw fresh random numbers every step, as Isaac's torch RNG does;
the task's one-row reward buffer is indexed modulo its length.

``extras["time_outs"]`` marks envs that ended by the 20 s limit (done and not a fall: the task's
termination term, ``terms[:, 7]``, is negative exactly when the torso touched or the state blew up),
so rsl_rl bootstraps them like Isaac's ``RslRlVecEnvWrapper``. ``episode_length_buf`` is a view of the
task's per-env step counter, so ``runner.learn(init_at_random_ep_len=True)`` (Isaac's train.py)
staggers the time-outs as in Isaac.

Per-term episode sums (the eight terms the reward kernel exposes, x dt, divided by the 20 s episode
like Isaac's RewardManager) are accumulated on the device and read once per iteration with
``pop_term_stats()``; they are not passed to rsl_rl, so its console output stays per-term free.
"""
from __future__ import annotations

import torch
import warp as wp
from tensordict import TensorDict

from metalsim.interop import torch_bridge as tb
from metalsim.learn.g1_velocity import CONTROL_DT, EPISODE_S, G1VelocityTask
from metalsim.learn.warp_policy import RolloutBuffers, bump

TERM_NAMES = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "feet_slide", "joint_deviation_all",
              "flat_orientation_l2", "action_rate_l2", "termination_penalty"]


class G1RslRlVecEnv:
    """``rsl_rl.env.VecEnv`` over ``G1VelocityTask`` (duck-typed; rsl_rl only reads the attributes)."""

    def __init__(self, task: G1VelocityTask, capture: bool = True):
        self.task = task
        self.num_envs, self.num_actions = task.n, task.act_dim
        self.max_episode_length = task.max_t
        self.device = "mps"
        self.cfg = {"task": "G1 velocity flat (MetalSim)", "physics_dt": task.physics_dt, "engine": task.engine}
        dev = task.device
        n = task.n

        class _Pol:                        # the kernels read the step counter from ``pol.step_idx``
            step_idx = wp.zeros(1, dtype=int, device=dev)
        self.pol = _Pol()
        self.bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim, device=dev)
        self.bufs.rew = wp.zeros((1, n), dtype=float, device=dev)
        self.bufs.done = wp.zeros((1, n), dtype=float, device=dev)
        self.action = wp.zeros((n, task.act_dim), dtype=float, device=dev)
        wp.synchronize_device(dev)
        self.t_action = tb.mps_tensor(self.action)
        self.t_obs = tb.mps_tensor(task.obs)
        self.t_rew = tb.mps_tensor(self.bufs.rew)
        self.t_done = tb.mps_tensor(self.bufs.done)
        self.t_terms = tb.mps_tensor(task.terms)
        self._t_len = tb.mps_tensor(task.t)          # int32 per-env step counter (time-out clock)
        # per-term bookkeeping (device side, read once per iteration)
        self.ep_terms = torch.zeros(n, 8, device="mps")
        self.done_terms = torch.zeros(8, device="mps")
        self.done_count = torch.zeros((), device="mps")
        self.done_falls = torch.zeros((), device="mps")
        self.nonfinite_obs = torch.zeros((), device="mps")
        # initial state: reset every env, then the first observation
        task.reset_all()
        task.episode_stats()
        with wp.ScopedDevice(dev):
            wp.launch(bump, dim=1, inputs=[self.pol.step_idx], device=dev)
            task.launch_obs(self.pol.step_idx)
        task.sim.synchronize()
        self._obs = self.t_obs.clone()
        self.graph = None
        if capture:
            with wp.ScopedCapture(device=dev) as cap:
                self._body()
            self.graph = cap.graph

    # -- VecEnv contract ---------------------------------------------------------------------------

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._t_len

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # rsl_rl's init_at_random_ep_len assigns a new tensor; write it into the task's clock instead
        self._t_len.copy_(value.to(self._t_len.dtype))
        v = self.task.sim.event.next_value(); tb.signal_event(self.task.sim.event, v); self.task.sim.wait(self.task.sim.event, v)

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._obs}, batch_size=[self.num_envs], device=self.device)

    def reset(self):
        return self.get_observations(), {}

    def _body(self):
        task = self.task
        wp.launch(bump, dim=1, inputs=[self.pol.step_idx], device=task.device)   # free-running: fresh RNG each step
        task.launch_apply_action(self.action)
        task.sim.launch_step()
        task.launch_reward_done_reset(self.pol, self.bufs)
        task.launch_obs(self.pol.step_idx)

    def step(self, actions: torch.Tensor):
        task = self.task
        self.t_action.copy_(actions)
        v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            with wp.ScopedDevice(task.device):
                self._body()
        task.sim.after(task.sim._signal())
        obs = self.t_obs.clone()
        bad = ~torch.isfinite(obs)
        self.nonfinite_obs += bad.any(-1).sum()
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        self._obs = obs
        rew = self.t_rew[0].clone()
        done = self.t_done[0] > 0.5
        terms = self.t_terms.clone()
        fell = terms[:, 7] < 0.0
        time_outs = done & ~fell
        # per-term episode sums in Isaac's units (weight * term * dt summed over the episode / 20 s)
        self.ep_terms += terms * CONTROL_DT
        dm = done.float().unsqueeze(-1)
        self.done_terms += (self.ep_terms * dm).sum(0)
        self.done_count += done.sum()
        self.done_falls += (done & fell).sum()
        self.ep_terms *= (1.0 - dm)
        extras = {"time_outs": time_outs.float()}
        return self.get_observations(), rew, done.long(), extras

    # -- diagnostics -------------------------------------------------------------------------------

    def pop_term_stats(self) -> dict:
        """Per-term Episode_Reward means over the episodes finished since the last call (synchronizes)."""
        c = float(self.done_count.item())
        out = {"episodes": int(c), "falls": int(self.done_falls.item()), "nonfinite_obs_rows": int(self.nonfinite_obs.item())}
        if c > 0:
            per = (self.done_terms / c / EPISODE_S).tolist()
            out["Episode_Reward"] = {k: round(v, 4) for k, v in zip(TERM_NAMES, per)}
        self.done_terms.zero_(); self.done_count.zero_(); self.done_falls.zero_(); self.nonfinite_obs.zero_()
        return out

    def close(self):
        pass
