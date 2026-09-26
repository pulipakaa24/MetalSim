"""Rollout policy in Warp kernels (WS7): a Gaussian MLP policy whose forward pass, action sampling
and log-probability run as Warp kernels on the simulator's queue, writing actions straight into
the simulator's control buffer. A whole rollout then runs with no PyTorch launches per step; torch
receives the rollout buffers (observations, actions, log-probs, values) once per update.

Weights are zero-copy views of the torch parameters (MPS tensors over Warp arrays), so the torch
optimizer updates them in place and the next rollout uses them with no copy. Layout follows
Isaac Lab / rsl_rl actor-critic MLPs (ELU activations, diagonal Gaussian with a state-independent
log-std). A parity test checks logits against the torch module to float32 tolerance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import warp as wp

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm

MAX_WIDTH = 512


@wp.func
def elu(x: float) -> float:
    if x > 0.0:
        return x
    return wp.exp(x) - 1.0


@wp.kernel
def mlp_layer(x: wp.array2d(dtype=float), W: wp.array2d(dtype=float), b: wp.array(dtype=float),
              act: int, y: wp.array2d(dtype=float)):
    """y[e, j] = act(sum_i x[e, i] W[j, i] + b[j]); one thread per (env, output)."""
    e, j = wp.tid()
    n_in = W.shape[1]          # may be < x.shape[1]: an actor reads the leading columns of a wider row
    s = b[j]
    for i in range(n_in):
        s += x[e, i] * W[j, i]
    if act == 1:
        s = elu(s)
    y[e, j] = s


@wp.kernel
def mlp_layer4(x: wp.array2d(dtype=float), W: wp.array2d(dtype=float), b: wp.array(dtype=float),
               act: int, y: wp.array2d(dtype=float)):
    """mlp_layer with one thread per (group of 4 outputs, env), envs fastest: a SIMD group reads 4 weight rows once
    and 32 x rows, each x element feeds 4 accumulators. Every output keeps mlp_layer's sequential dot product over
    its inputs, so the results are bitwise the same (scripts/diagnostics/warp_policy_cost.py, tests/test_warp_policy.py);
    measured 1.8x (256-128-128) / 2.0x (512-256-128) faster at 4096 envs on the M4 Max (2026-09-26)."""
    g, e = wp.tid()
    n_in = W.shape[1]
    n_out = W.shape[0]
    j0 = g * 4
    s0 = b[j0]; s1 = float(0.0); s2 = float(0.0); s3 = float(0.0)
    if j0 + 1 < n_out:
        s1 = b[j0 + 1]
    if j0 + 2 < n_out:
        s2 = b[j0 + 2]
    if j0 + 3 < n_out:
        s3 = b[j0 + 3]
    for i in range(n_in):
        xi = x[e, i]
        s0 += xi * W[j0, i]
        if j0 + 1 < n_out:
            s1 += xi * W[j0 + 1, i]
        if j0 + 2 < n_out:
            s2 += xi * W[j0 + 2, i]
        if j0 + 3 < n_out:
            s3 += xi * W[j0 + 3, i]
    if act == 1:
        s0 = elu(s0); s1 = elu(s1); s2 = elu(s2); s3 = elu(s3)
    y[e, j0] = s0
    if j0 + 1 < n_out:
        y[e, j0 + 1] = s1
    if j0 + 2 < n_out:
        y[e, j0 + 2] = s2
    if j0 + 3 < n_out:
        y[e, j0 + 3] = s3


def launch_layer(x, W, b, act, y, n, device):
    """One MLP layer y = act(x W^T + b) for n envs (mlp_layer4 mapping; METALSIM_MLP_MAPPING=A for the original)."""
    if _MLP_MAPPING == "A":
        wp.launch(mlp_layer, dim=(n, W.shape[0]), inputs=[x, W, b, act, y], device=device)
    else:
        wp.launch(mlp_layer4, dim=((W.shape[0] + 3) // 4, n), inputs=[x, W, b, act, y], device=device)


_MLP_MAPPING = __import__("os").environ.get("METALSIM_MLP_MAPPING", "D")


@wp.kernel
def sample_gaussian(mean: wp.array2d(dtype=float), log_std: wp.array(dtype=float), seed: int, step_idx: wp.array(dtype=int),
                    action: wp.array2d(dtype=float), logp: wp.array(dtype=float), ctrl_lo: wp.array(dtype=float),
                    ctrl_hi: wp.array(dtype=float), ctrl: wp.array2d(dtype=float)):
    """Per env: sample a ~ N(mean, exp(log_std)), write the summed log-prob, and map to ctrl range.
    The RNG stream advances with a device-side step counter so the kernel can live in a replayed graph."""
    e = wp.tid()
    n = mean.shape[1]
    rng = wp.rand_init(seed, step_idx[0] * 1000003 + e)
    lp = float(0.0)
    for k in range(n):
        s = wp.exp(log_std[k])
        z = wp.randn(rng)
        a = mean[e, k] + s * z
        action[e, k] = a
        lp += -0.5 * z * z - log_std[k] - 0.9189385332046727
        ac = wp.clamp(a, -1.0, 1.0)
        ctrl[e, k] = ctrl_lo[k] + 0.5 * (ac + 1.0) * (ctrl_hi[k] - ctrl_lo[k])
    logp[e] = lp


@wp.kernel
def store_step(step_idx: wp.array(dtype=int), obs: wp.array2d(dtype=float), action: wp.array2d(dtype=float),
               logp: wp.array(dtype=float), value: wp.array2d(dtype=float),
               buf_obs: wp.array3d(dtype=float), buf_act: wp.array3d(dtype=float), buf_logp: wp.array2d(dtype=float),
               buf_val: wp.array2d(dtype=float)):
    """Copies this step's transition into rollout buffers at row step_idx (one thread per env)."""
    e = wp.tid()
    t = step_idx[0]
    for i in range(obs.shape[1]):
        buf_obs[t, e, i] = obs[e, i]
    for k in range(action.shape[1]):
        buf_act[t, e, k] = action[e, k]
    buf_logp[t, e] = logp[e]
    buf_val[t, e] = value[e, 0]


@wp.kernel
def bump(step_idx: wp.array(dtype=int)):
    if wp.tid() == 0:
        step_idx[0] = step_idx[0] + 1


@wp.kernel
def zero_int(a: wp.array(dtype=int)):
    if wp.tid() == 0:
        a[0] = 0


class WarpMLPPolicy:
    """Gaussian MLP actor + value head evaluated in Warp. ``net`` is a torch module with attributes
    ``actor`` (nn.Sequential of Linear/ELU), ``critic`` (same) and ``log_std`` (Parameter)."""

    def __init__(self, net: nn.Module, num_envs: int, obs_dim: int, act_dim: int, ctrl_lo, ctrl_hi, device="metal:0", seed=0):
        self.net, self.n, self.obs_dim, self.act_dim, self.device = net, num_envs, obs_dim, act_dim, device
        self.seed = seed
        self.step_idx = wp.zeros(1, dtype=int, device=device)   # device-side step counter (graph-replay safe)
        # Free-running step counter that keys the random streams (action noise here; observation noise,
        # commands and resets in tasks that take it). ``step_idx`` is rewound every rollout and indexes the
        # rollout buffers; keying the RNG on it replayed the same noise sequence in every rollout.
        self.rng_step = wp.zeros(1, dtype=int, device=device)
        self.actor_layers = self._bind_layers(net.actor)
        self.critic_layers = self._bind_layers(net.critic)
        self.log_std = self._bind_param(net.log_std)
        # activations: preallocated per layer
        def bufs(layers):
            out = []
            for W, b, act in layers:
                out.append(wp.zeros((num_envs, W.shape[0]), dtype=float, device=device))
            return out
        self.actor_act = bufs(self.actor_layers)
        self.critic_act = bufs(self.critic_layers)
        self.action = wp.zeros((num_envs, act_dim), dtype=float, device=device)
        self.logp = wp.zeros(num_envs, dtype=float, device=device)
        self.ctrl_lo = wp.array(np.asarray(ctrl_lo, np.float32), dtype=float, device=device)
        self.ctrl_hi = wp.array(np.asarray(ctrl_hi, np.float32), dtype=float, device=device)
        wp.synchronize_device(device)
        self.t_action = tb.mps_tensor(self.action)
        self.t_logp = tb.mps_tensor(self.logp)
        self.t_value = tb.mps_tensor(self.critic_act[-1])
        self.t_mean = tb.mps_tensor(self.actor_act[-1])

    # -- parameter binding: torch params re-homed onto Warp memory (zero-copy both ways) ------------

    def _bind_param(self, p: nn.Parameter) -> wp.array:
        a = wp.array(p.detach().cpu().numpy().astype(np.float32), dtype=float, device=self.device)
        wp.synchronize_device(self.device)
        t = tb.mps_tensor(a)
        p.data = t                       # the parameter now lives in the Warp array's Metal buffer
        return a

    def _bind_layers(self, seq: nn.Sequential):
        layers = []
        mods = list(seq)
        for i, m in enumerate(mods):
            if isinstance(m, nn.Linear):
                act = 1 if i + 1 < len(mods) and isinstance(mods[i + 1], (nn.ELU,)) else 0
                W = self._bind_param(m.weight)
                b = self._bind_param(m.bias)
                layers.append((W, b, act))
        return layers

    # -- rollout step ------------------------------------------------------------------------------

    def act(self, obs: wp.array, ctrl: wp.array) -> None:
        """obs: (N, obs_dim) Warp array (e.g. a view of simulator state). Writes ctrl (N, act_dim)
        and the action/logp/value buffers. All launches go to the Warp queue; nothing syncs."""
        x = obs
        for (W, b, act), y in zip(self.actor_layers, self.actor_act):
            launch_layer(x, W, b, act, y, self.n, self.device)
            x = y
        xc = obs
        for (W, b, act), y in zip(self.critic_layers, self.critic_act):
            launch_layer(xc, W, b, act, y, self.n, self.device)
            xc = y
        wp.launch(sample_gaussian, dim=self.n,
                  inputs=[self.actor_act[-1], self.log_std, self.seed, self.rng_step, self.action, self.logp,
                          self.ctrl_lo, self.ctrl_hi, ctrl], device=self.device)

    def store(self, obs: wp.array, bufs: "RolloutBuffers") -> None:
        """Append this step's transition to ``bufs`` at the device-side step index, then advance it."""
        wp.launch(store_step, dim=self.n, inputs=[self.step_idx, obs, self.action, self.logp, self.critic_act[-1],
                                                    bufs.obs, bufs.act, bufs.logp, bufs.value], device=self.device)
        wp.launch(bump, dim=1, inputs=[self.step_idx], device=self.device)
        wp.launch(bump, dim=1, inputs=[self.rng_step], device=self.device)

    def rewind(self) -> None:
        """Reset the buffer row index (start of a rollout; ``rng_step`` keeps running). Warp queue, no sync."""
        wp.launch(zero_int, dim=1, inputs=[self.step_idx], device=self.device)


class RolloutBuffers:
    """Rollout storage in Warp memory with zero-copy torch views for the update."""

    def __init__(self, T: int, n: int, obs_dim: int, act_dim: int, device="metal:0"):
        self.obs = wp.zeros((T, n, obs_dim), dtype=float, device=device)
        self.act = wp.zeros((T, n, act_dim), dtype=float, device=device)
        self.logp = wp.zeros((T, n), dtype=float, device=device)
        self.value = wp.zeros((T, n), dtype=float, device=device)
        wp.synchronize_device(device)
        self.t_obs, self.t_act = tb.mps_tensor(self.obs), tb.mps_tensor(self.act)
        self.t_logp, self.t_value = tb.mps_tensor(self.logp), tb.mps_tensor(self.value)


class ActorCriticMLP(nn.Module):
    """rsl_rl-style actor-critic MLP (ELU)."""

    def __init__(self, obs_dim, act_dim, hidden=(512, 256, 128), actor_in=None):
        """Asymmetric when ``actor_in`` < ``obs_dim``: the actor reads the first ``actor_in`` columns of
        each observation row (what the real robot has), the critic the whole row (plus privileged
        simulator state)."""
        super().__init__()
        self.actor_in = actor_in or obs_dim
        def mlp(out, d_in):
            layers, d = [], d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            layers.append(nn.Linear(d, out))
            return nn.Sequential(*layers)
        self.actor = mlp(act_dim, self.actor_in)
        self.critic = mlp(1, obs_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def actor_mean(self, obs):
        return self.actor(obs[..., :self.actor_in])
