"""Isaac Lab 3.0-EA's ``IsaacContrib-Franka-Pour`` task (Franka grasps a cup of granular MPM media and pours it into a
second cup) on Metal: physics in ``franka_pour_scene`` (Newton 1.5.2 ``SolverCoupledProxy``: MuJoCo Warp + implicit
MPM, one captured graph per policy step), the MDP here in PyTorch on MPS over zero-copy views of the Newton state.

Every term is a transcription of Isaac's source (``upstream/IsaacLab3`` at ``ae37b028``; task files relative to
``source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/``), with Isaac Lab's own math utilities vendored verbatim
(``metalsim/learn/isaac_math.py`` = ``source/isaaclab/isaaclab/utils/math.py``). ``tests/test_franka_pour_terms.py``
runs Isaac's original functions on this environment's live state and asserts equality term by term.

* control 30 Hz (sim dt 1/120, decimation 4), episodes 12 s = 360 steps (``pour_env_cfg.py:597-600``)
* actions (``ActionsCfg``, :373-393): 7 arm deltas, ``EMARelativeJointPositionAction`` (scale 0.03, EMA alpha 0.2;
  target = filtered delta + current joint position every physics step) and one gripper command
  ``CurriculumGripperPositionAction`` (a < 0: close 0.021 m, else 0.04 m, filtered with alpha 1 - 0.8^(1/3);
  initial target 0.024 m); the rsl_rl wrapper clips actions to [-1, 1]
* observations (``ResetDatasetObservationsCfg``, :396-464): policy (62) + media (22) for the actor (84), + privileged
  (2) for the critic (86); no noise, no history
* rewards (:467-474): +5 on terminal particle-transfer success, -1 on a non-timeout failure, -1e-4 action L2,
  -1e-4 action rate; every term x dt as Isaac's reward manager does
* terminations (:477-519) in Isaac's order: non-finite state, extreme rigid state, source/receiver overlap, lost grasp
  (monitor only), spill > 10 %, particle out of bounds, success (>= 70 % of particles in the receiver), the
  curriculum's local-progress context (never terminates), time out (neutral)
* resets (``EventsCfg``, ``pour_env.py:595-686``): rows of Isaac's published 20,000-row reset dataset
  (``Contrib/MPM/Pour/reset_dataset.pt``), chosen by the adaptive replay curriculum (``reset_sampler.py``,
  ``mdp/reset_dataset.py``; 50 % shuffled cyclic replay, the rest weighted sqrt(p (1 - p)) + 1e-4 over each row's last
  50 local-progress outcomes); particles re-placed from env 0's initial lattice in the row's source-cup frame, MPM
  solver state cleared per world.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import torch
import warp as wp

from metalsim.learn import franka_pour_scene as S
from metalsim.learn import isaac_math as mu

STEP_DT = S.SIM_DT * S.DECIMATION
MAX_EPISODE_LENGTH = math.ceil(S.EPISODE_LENGTH_S / STEP_DT)      # manager_based_rl_env.py:104 -> 360
GRASPING_CATEGORY = 1                                              # pour_env.py:60
_GRIPPER_POSITION_TOLERANCE = 1.0e-6                               # mdp/actions.py:27, terminations.py:27
RESET_DATASET_STATE_NAMES = ("arm_joint_position", "arm_joint_velocity", "finger_joint_position",
                             "finger_joint_velocity", "finger_joint_target", "source_root_pose", "source_root_velocity",
                             "target_root_pose", "target_root_velocity", "category", "objective", "reset_region",
                             "difficulty", "particle_layout_id")


@dataclass
class SamplerConfig:
    """``ResetDatasetSamplerCfg`` as the task configures it (pour_env_cfg.py:583-590)."""

    monitored_history_len: int = 50
    target_success_rate: float = 0.50
    kappa: float = 1.0
    epsilon: float = 1.0e-4
    uniform_fraction: float = 0.50


@dataclass
class PourTaskConfig:
    """``FrankaPourResetDatasetEnvCfg`` fields (pour_env_cfg.py:548-591) and the run settings."""

    num_envs: int = 64
    seed: int = 0
    torch_device: str = "mps"
    tcp_offset_pos: tuple = S.TCP_OFFSET_POS
    tcp_offset_rot: tuple = (0.0, 0.0, 0.0, 1.0)
    source_fill_level: float = S.MEDIA_FILL_LEVEL
    pour_target_frac: float = 0.70
    particle_count_margin: float = 0.003
    spill_table_height: float = 0.0
    max_spill_fraction: float = 0.10
    success_dwell_time_s: float = 0.15
    success_min_lift_height: float = 0.05
    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_cup_linear_velocity: float = 10.0
    state_bound_max_cup_angular_velocity: float = 50.0
    particle_workspace_lower_bound: tuple = (-1.0, -1.0, -0.5)
    particle_workspace_upper_bound: tuple = (1.5, 1.0, 1.5)
    reset_dataset_path: str = S.RESET_DATASET
    reset_dataset_sampling_mode: str = "adaptive"
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    curriculum_freeze: bool = False
    # TerminationsCfg parameters
    overlap_clearance: float = 0.001
    lost_grasp_dwell_time_s: float = 0.05
    lost_grasp_max_tcp_distance: float = 0.018
    lost_grasp_max_gripper_width_error: float = 0.006
    lost_grasp_max_gripper_command: float = 0.027
    progress_minimum: float = 0.08
    progress_minimum_episode_steps: int = 3
    potential_params: dict = field(default_factory=lambda: dict(
        approach_position_std=0.20, approach_orientation_std=0.75, approach_open_hand_fraction=0.15,
        grasp_target_height=0.10, grasp_reach_std=0.025, grasp_preload_position=0.024, grasp_fraction=0.40,
        source_mouth_height=0.119, target_rim_height=0.074, target_clearance_height=0.15, alignment_std=0.20,
        tilt_alignment_radius=0.15, target_tilt=math.radians(140.0)))
    # actions (ActionsCfg)
    arm_scale: float = 0.03
    arm_alpha: float = 0.20
    gripper_alpha: float = 1.0 - (1.0 - 0.2) ** (1.0 / 3.0)
    gripper_close: float = 0.021
    gripper_neutral: float = 0.04
    gripper_default: float = 0.024
    gripper_contact_min_deflection: float = 0.001
    clip_actions: float = 1.0                     # rsl_rl runner cfg clip_actions (RslRlVecEnvWrapper)
    physics: S.PourPhysicsConfig = field(default_factory=S.PourPhysicsConfig)


# ------------------------------------------------------------------------------------------------ reset dataset
def _write_bytes(sink, value: bytes):
    sink.write(len(value).to_bytes(8, byteorder="big", signed=False))
    sink.write(value)


def _write_value(sink, value):
    """``reset_dataset_io._write_value`` (canonical encoding for the artifact's content digest)."""
    if value is None:
        sink.write(b"none")
    elif isinstance(value, bool):
        sink.write(b"bool")
        sink.write(b"1" if value else b"0")
    elif isinstance(value, int):
        sink.write(b"int")
        _write_bytes(sink, str(value).encode("ascii"))
    elif isinstance(value, float):
        sink.write(b"float")
        _write_bytes(sink, value.hex().encode("ascii"))
    elif isinstance(value, str):
        sink.write(b"str")
        _write_bytes(sink, value.encode("utf-8"))
    elif isinstance(value, bytes):
        sink.write(b"bytes")
        _write_bytes(sink, value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        sink.write(b"tensor")
        _write_bytes(sink, str(tensor.dtype).encode("ascii"))
        _write_value(sink, tuple(tensor.shape))
        _write_bytes(sink, tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        sink.write(b"mapping")
        items = []
        for key, item in value.items():
            kb = BytesIO()
            _write_value(kb, key)
            items.append((kb.getvalue(), item))
        items.sort(key=lambda p: p[0])
        _write_value(sink, len(items))
        for kb, item in items:
            _write_bytes(sink, kb)
            _write_value(sink, item)
    elif isinstance(value, (tuple, list)):
        sink.write(b"tuple" if isinstance(value, tuple) else b"list")
        _write_value(sink, len(value))
        for item in value:
            _write_value(sink, item)
    else:
        raise TypeError(f"unsupported reset-dataset value {type(value).__name__}")


class _Hash:
    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, b):
        self.digest.update(b)


def reset_dataset_content_digest(payload: dict) -> str:
    """``reset_dataset_io.reset_dataset_content_digest``: SHA-256 of the payload without its digest field."""
    h = _Hash()
    _write_value(h, {k: v for k, v in payload.items() if k != "content_sha256"})
    return h.digest.hexdigest()


def load_reset_dataset(path: str, device) -> dict:
    """Load Isaac's reset artifact and check what ``reset_dataset_validate_runtime`` checks that matters here: format,
    schema 12, its own content digest, the task contract fields this port shares, tensor shapes and finiteness."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "franka_pour_reset_dataset" or payload.get("schema_version") != 12:
        raise ValueError("not a Franka Pour reset dataset (schema 12)")
    if payload["content_sha256"] != reset_dataset_content_digest(payload):
        raise ValueError("reset dataset content digest does not match its payload")
    tc = payload["metadata"]["task_contract"]
    expect = {"robot_physics_payload_sha256": S.FRANKA_PHYSICS_PAYLOAD_SHA256, "tcp_body_name": S.TCP_BODY,
              "cup_grasp_height": S.CUP_GRASP_HEIGHT, "gripper_open_pos": S.GRIPPER_OPEN_POSITION,
              "gripper_grasp_reset_target": 0.021, "gripper_contact_min_deflection": 0.001,
              "collider_margin": S.COLLIDER_MARGIN, "mpm_collider_margin": S.MPM_COLLIDER_MARGIN}
    for k, v in expect.items():
        if (abs(float(tc[k]) - float(v)) > 1e-9) if isinstance(v, float) else tc[k] != v:
            raise ValueError(f"reset dataset task contract {k}={tc[k]!r} does not match {v!r}")
    states = payload["states"]
    n = int(states["category"].shape[0])
    for name in RESET_DATASET_STATE_NAMES:
        t = states[name]
        if t.shape[0] != n or (t.is_floating_point() and not bool(torch.isfinite(t).all())):
            raise ValueError(f"reset dataset states.{name} malformed")
    return {name: states[name].to(device) for name in RESET_DATASET_STATE_NAMES} | {
        "_content_sha256": payload["content_sha256"], "_rows": n}


# ------------------------------------------------------------------------------------------------ reset sampler
def ring_append_bool_count_rate(data, stream_ids, values, pointer, size, true_count, rate) -> None:
    """``reset_sampler._ring_append_bool_count_rate`` (verbatim)."""
    if stream_ids.numel() == 0:
        return
    capacity = data.shape[1]
    unique_ids, inverse, counts = torch.unique(stream_ids, return_inverse=True, return_counts=True)
    if unique_ids.numel() == stream_ids.numel():
        columns = pointer[stream_ids].long()
        overwritten = torch.where(size[stream_ids] == capacity, data[stream_ids, columns].to(dtype=true_count.dtype),
                                  torch.zeros_like(true_count[stream_ids]))
        new_true_counts = true_count[stream_ids] - overwritten + values.to(dtype=true_count.dtype)
        data[stream_ids, columns] = values
        pointer[stream_ids] = ((columns + 1) % capacity).to(dtype=pointer.dtype)
        size[stream_ids] = (size[stream_ids] + 1).clamp(max=capacity)
        true_count[stream_ids] = new_true_counts
        rate[stream_ids] = new_true_counts.to(rate.dtype) / size[stream_ids].clamp(min=1)
        return
    order = torch.argsort(inverse, stable=True)
    sorted_ids = stream_ids[order]
    sorted_values = values[order]
    group_starts = counts.cumsum(0) - counts
    local_rank = torch.arange(stream_ids.numel(), device=data.device) - torch.repeat_interleave(group_starts, counts)
    inverse_sorted = inverse[order]
    counts_sorted = counts[inverse_sorted]
    true_added = torch.zeros(unique_ids.shape, device=data.device, dtype=true_count.dtype)
    true_added.scatter_add_(0, inverse, values.to(dtype=true_count.dtype))
    keep_start = (counts - capacity).clamp(min=0)
    keep = local_rank >= torch.repeat_interleave(keep_start, counts)
    true_kept = torch.zeros_like(true_added)
    true_kept.scatter_add_(0, inverse_sorted[keep], sorted_values[keep].to(dtype=true_count.dtype))
    overwrite_start = capacity - size[sorted_ids].long()
    overwrite_mask = (counts_sorted < capacity) & (local_rank >= overwrite_start)
    overwritten = torch.zeros_like(true_added)
    overwrite_ids = sorted_ids[overwrite_mask]
    overwrite_columns = (pointer[overwrite_ids].long() + local_rank[overwrite_mask]) % capacity
    overwritten.scatter_add_(0, inverse_sorted[overwrite_mask],
                             data[overwrite_ids, overwrite_columns].to(dtype=true_count.dtype))
    kept_ids = sorted_ids[keep]
    kept_columns = (pointer[kept_ids].long() + local_rank[keep]) % capacity
    data[kept_ids, kept_columns] = sorted_values[keep]
    replace = counts >= capacity
    new_true_counts = torch.where(replace, true_kept, true_count[unique_ids] - overwritten + true_added)
    new_size = (size[unique_ids].long() + counts).clamp(max=capacity)
    pointer[unique_ids] = ((pointer[unique_ids].long() + counts) % capacity).to(dtype=pointer.dtype)
    size[unique_ids] = new_size.to(dtype=size.dtype)
    true_count[unique_ids] = new_true_counts
    rate[unique_ids] = new_true_counts.to(rate.dtype) / new_size.clamp(min=1).to(rate.dtype)


class ResetDatasetSampler:
    """``reset_sampler._ResetDatasetSampler`` (verbatim logic; Isaac's RNG is torch's, so is this one)."""

    def __init__(self, row_count: int, device, cfg: SamplerConfig, generator: torch.Generator | None = None):
        self.cfg = cfg
        self.row_count = row_count
        self.generator = generator
        self._alpha_minus_one = cfg.kappa * cfg.target_success_rate
        self._beta_minus_one = cfg.kappa * (1.0 - cfg.target_success_rate)
        self._success_rates = torch.zeros(row_count, dtype=torch.float32, device=device)
        self._success_history = torch.zeros((row_count, cfg.monitored_history_len), dtype=torch.bool, device=device)
        self._history_pointers = torch.zeros(row_count, dtype=torch.int32, device=device)
        self._history_sizes = torch.zeros_like(self._history_pointers)
        self._history_success_counts = torch.zeros_like(self._history_pointers)
        self._uniform_order = self._randperm(row_count)
        self._uniform_cursor = 0
        self._uniform_credit = 0.0

    def _randperm(self, n):
        # torch.randperm with an MPS generator is not available; draw on the CPU generator, then move
        return torch.randperm(n, generator=self.generator).to(self._success_rates.device)

    def _probabilities(self) -> torch.Tensor:
        weights = self._success_rates.pow(self._alpha_minus_one)
        weights *= (1.0 - self._success_rates).pow(self._beta_minus_one)
        weights.add_(self.cfg.epsilon)
        return weights / weights.sum()

    def sample_with_uniform_replay(self, count: int) -> torch.Tensor:
        device = self._success_rates.device
        rows = torch.empty(count, device=device, dtype=torch.long)
        if count == 0:
            return rows
        uniform_count = self._uniform_assignment_count(count)
        assignment_order = self._randperm(count)
        uniform_positions = assignment_order[:uniform_count]
        adaptive_positions = assignment_order[uniform_count:]
        if uniform_count > 0:
            rows[uniform_positions] = self._take_uniform_rows(uniform_count)
        if adaptive_positions.numel() > 0:
            p = self._probabilities().cpu()
            rows[adaptive_positions] = torch.multinomial(p, adaptive_positions.numel(), replacement=True,
                                                         generator=self.generator).to(device)
        return rows

    def record(self, rows: torch.Tensor, success: torch.Tensor) -> None:
        ring_append_bool_count_rate(self._success_history, rows, success, self._history_pointers, self._history_sizes,
                                    self._history_success_counts, self._success_rates)

    def metrics(self) -> dict[str, float]:
        replay_fraction = float(self.cfg.uniform_fraction)
        probabilities = self._probabilities()
        probabilities.mul_(1.0 - replay_fraction).add_(replay_fraction / self.row_count)
        top_count = max(1, math.ceil(0.01 * self.row_count))
        values = torch.stack((probabilities.square().sum().reciprocal() / self.row_count,
                              torch.topk(probabilities, top_count, sorted=False).values.sum(),
                              probabilities.new_tensor(replay_fraction)))
        names = ("sampler/effective_pool_fraction", "sampler/top_1_percent_mass", "sampler/uniform_replay_fraction")
        return dict(zip(names, values.tolist(), strict=True))

    def _uniform_assignment_count(self, unforced_count: int) -> int:
        replay_credit = self._uniform_credit + self.cfg.uniform_fraction * unforced_count
        uniform_count = math.floor(replay_credit + 1.0e-12)
        self._uniform_credit = replay_credit - uniform_count
        return min(unforced_count, uniform_count)

    def _take_uniform_rows(self, count: int) -> torch.Tensor:
        remaining = count
        chunks = []
        while remaining > 0:
            available = self.row_count - self._uniform_cursor
            take = min(remaining, available)
            chunks.append(self._uniform_order[self._uniform_cursor:self._uniform_cursor + take])
            self._uniform_cursor += take
            remaining -= take
            if self._uniform_cursor == self.row_count:
                self._uniform_order = self._randperm(self.row_count)
                self._uniform_cursor = 0
        return torch.cat(chunks)


# ------------------------------------------------------------------------------------------------ geometry.py
def box_center_from_base_pose(pose, half_extents):
    offset = torch.zeros_like(pose[:, :3])
    offset[:, 2] = half_extents[2]
    return pose[:, :3] + mu.quat_apply(pose[:, 3:7], offset)


def points_inside_box(points, pose, lower, upper, margin: float = 0.0):
    relative = points - pose[:, None, :3]
    quaternion = pose[:, None, 3:7].expand(-1, points.shape[1], -1)
    local = mu.quat_apply_inverse(quaternion, relative)
    return ((local >= lower - margin) & (local <= upper + margin)).all(dim=-1)


def oriented_boxes_overlap(center_a, quaternion_a, half_a, center_b, quaternion_b, half_b, clearance: float = 0.0):
    rotation_a = mu.matrix_from_quat(quaternion_a)
    rotation_b = mu.matrix_from_quat(quaternion_b)
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    absolute = relative.abs() + 1.0e-6
    translation = (rotation_a.transpose(-1, -2) @ (center_b - center_a).unsqueeze(-1)).squeeze(-1)
    a = half_a + 0.5 * clearance
    b = half_b + 0.5 * clearance
    separated = torch.zeros(center_a.shape[0], device=center_a.device, dtype=torch.bool)
    for axis in range(3):
        separated |= translation[:, axis].abs() > a[axis] + (absolute[:, axis, :] * b).sum(-1)
        projection = (translation * relative[:, :, axis]).sum(-1).abs()
        separated |= projection > (absolute[:, :, axis] * a).sum(-1) + b[axis]
    for axis_a in range(3):
        for axis_b in range(3):
            other_a, last_a = (axis_a + 1) % 3, (axis_a + 2) % 3
            other_b, last_b = (axis_b + 1) % 3, (axis_b + 2) % 3
            projection = (translation[:, last_a] * relative[:, other_a, axis_b]
                          - translation[:, other_a] * relative[:, last_a, axis_b]).abs()
            radius_a = a[other_a] * absolute[:, last_a, axis_b] + a[last_a] * absolute[:, other_a, axis_b]
            radius_b = b[other_b] * absolute[:, axis_a, last_b] + b[last_b] * absolute[:, axis_a, other_b]
            separated |= projection > radius_a + radius_b
    return ~separated


# ------------------------------------------------------------------------------------------------ action terms
class ArmAction:
    """``EMARelativeJointPositionAction`` (mdp/actions.py:30-53) on ``RelativeJointPositionAction`` (scale 0.03,
    zero offset): processed = alpha * raw * scale + (1 - alpha) * previous processed; applied as target = processed +
    current joint position in every physics step (``apply_pour_actions``)."""

    action_dim = 7

    def __init__(self, n, device, scale, alpha):
        self.scale, self._alpha = float(scale), float(alpha)
        self._raw_actions = torch.zeros(n, 7, device=device)
        self._processed_actions = torch.zeros(n, 7, device=device)
        self._previous_delta = torch.zeros_like(self._processed_actions)

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self.scale          # JointAction: raw * scale + offset (0)
        self._processed_actions.lerp_(self._previous_delta, 1.0 - self._alpha)
        self._previous_delta.copy_(self._processed_actions)

    def reset(self, env_ids):
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._previous_delta[env_ids] = 0.0

    @property
    def processed_actions(self):
        return self._processed_actions


class GripperAction:
    """``CurriculumGripperPositionAction`` (mdp/actions.py:56-143) on ``BinaryJointPositionAction``."""

    action_dim = 1

    def __init__(self, n, device, env, alpha, close, neutral, default, contact_min_deflection):
        self._env = env
        self._alpha = float(alpha)
        self._close_command = torch.full((n, 2), float(close), device=device)
        self._open_command = torch.full((n, 2), float(neutral), device=device)
        self._neutral_position = float(neutral)
        self._contact_min_deflection = float(contact_min_deflection)
        self._raw_actions = torch.zeros(n, 1, device=device)
        self._processed_actions = torch.full((n, 2), float(default), device=device)

    @property
    def processed_actions(self):
        return self._processed_actions

    @property
    def commanded_position(self):
        return self._processed_actions[:, :1]

    def _joint_position(self):
        return self._env.finger_joint_pos()

    @property
    def contact_deflection(self):
        joint_position = self._joint_position()
        finite = torch.isfinite(joint_position) & torch.isfinite(self._processed_actions)
        return torch.where(finite, torch.clamp(joint_position - self._processed_actions, min=0.0),
                           torch.zeros_like(joint_position))

    @property
    def bilateral_contact(self):
        joint_position = self._joint_position()
        deflection = self.contact_deflection
        finite = torch.isfinite(joint_position).all(dim=-1) & torch.isfinite(self._processed_actions).all(dim=-1)
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return finite & command_valid & (deflection.amin(dim=-1) >= self._contact_min_deflection)

    @property
    def contact_quality(self):
        deflection = self.contact_deflection
        deflection_quality = torch.clamp(deflection.amin(dim=-1) / self._contact_min_deflection, 0.0, 1.0)
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return deflection_quality * command_valid.float()

    def set_reset_position(self, position, env_ids):
        expanded = position.to(dtype=self._processed_actions.dtype).expand(-1, 2)
        self._processed_actions[env_ids] = expanded

    def process_actions(self, actions):
        previous_target = self._processed_actions
        self._raw_actions[:] = actions
        binary_mask = actions < 0                                  # binary_joint_actions.py:132-139
        new = torch.where(binary_mask, self._close_command, self._open_command)
        previous_target.lerp_(new, self._alpha)
        self._processed_actions = previous_target

    def reset(self, env_ids):
        self._raw_actions[env_ids] = 0.0                           # BinaryJointAction.reset


# ------------------------------------------------------------------------------------------------ the environment
class FrankaPourEnv:
    """Batched Franka-Pour environment with Isaac Lab's step semantics (``ManagerBasedRLEnv.step``) and the task's
    terms. ``step(actions)`` returns ``(obs, reward, terminated, time_outs, extras)`` with ``obs`` a dict of the
    ``policy`` / ``media`` / ``privileged`` groups."""

    termination_terms = ("failure", "extreme_rigid_state", "source_receiver_overlap", "lost_grasp", "spill",
                         "particle_out_of_bounds", "success", "learning_progress_context", "time_out")
    reward_terms = ("success", "action_magnitude", "action_rate", "failure")

    def __init__(self, cfg: PourTaskConfig | None = None):
        from metalsim.interop import torch_bridge as tb

        self.cfg = cfg = cfg or PourTaskConfig()
        cfg.physics.num_envs = cfg.num_envs
        self.num_envs = n = cfg.num_envs
        self.device = dev = torch.device(cfg.torch_device)
        self.generator = torch.Generator().manual_seed(cfg.seed)      # host RNG for the sampler (Isaac: torch's)
        torch.manual_seed(cfg.seed)
        self.physics = ph = S.PourPhysics(cfg.physics)
        info = self.info = ph.info
        wp.synchronize_device(ph.device)

        def view(a):
            try:
                return tb.mps_tensor(a)
            except Exception:  # noqa: BLE001  (no compiler for the native bridge: the pure-Python DLPack path)
                return tb.mps_tensor(a, via="dlpack")

        P, bpw = info.particles_per_world, info.bodies_per_world
        self._jq = view(ph.s0.joint_q).view(n, info.joint_q_per_world)
        self._jqd = view(ph.s0.joint_qd).view(n, info.joint_dof_per_world)
        self._bq = view(ph.s0.body_q).view(n, bpw, 7)
        self._bqd = view(ph.s0.body_qd).view(n, bpw, 6)
        self._pq = view(ph.s0.particle_q).view(n, P, 3)
        self._pqd = view(ph.s0.particle_qd).view(n, P, 3)
        self._pq1 = view(ph.s1.particle_q).view(n, P, 3)
        self._pqd1 = view(ph.s1.particle_qd).view(n, P, 3)
        self._target = view(ph.control.joint_target_q).view(n, info.joint_dof_per_world)
        self._arm_delta = view(ph.arm_delta)
        self._gripper_target = view(ph.gripper_target)
        self._q_idx = torch.tensor(info.arm_q + info.finger_q, device=dev)
        self._dof_idx = torch.tensor(info.arm_dofs + info.finger_dofs, device=dev)
        self._arm_q_idx, self._finger_q_idx = self._q_idx[:7], self._q_idx[7:]
        self._arm_dof_idx, self._finger_dof_idx = self._dof_idx[:7], self._dof_idx[7:]
        self._finger_joint_ids = torch.tensor([7, 8], device=dev)    # robot joint order: joint1..7, finger1, finger2
        self._tcp_body_idx = info.hand_body
        self.env_origins = torch.as_tensor(info.origins, device=dev, dtype=torch.float32)
        lo = ph.model.joint_limit_lower.numpy().reshape(n, -1)
        hi = ph.model.joint_limit_upper.numpy().reshape(n, -1)
        dofs = info.arm_dofs + info.finger_dofs
        self._joint_pos_limits_t = torch.as_tensor(np.stack([lo[:, dofs], hi[:, dofs]], -1), device=dev)
        bc = ph.model.body_com.numpy().reshape(n, bpw, 3)
        self._source_com_b = torch.as_tensor(bc[:, info.source_body], device=dev)
        self.default_joint_pos = torch.tensor(S.ARM_HOME + (S.GRIPPER_OPEN_POSITION,) * 2, device=dev).repeat(n, 1)

        # FrankaPourEnv._prepare_newton_extras / _setup_after_physics (pour_env.py:121-139, 451-494)
        self._source_inner_lo_t = torch.as_tensor(S.SOURCE_CUP_GEOMETRY.inner_bounds[0], device=dev)
        self._source_inner_hi_t = torch.as_tensor(S.SOURCE_CUP_GEOMETRY.inner_bounds[1], device=dev)
        self._target_inner_lo_t = torch.as_tensor(S.TARGET_CUP_GEOMETRY.inner_bounds[0], device=dev)
        self._target_inner_hi_t = torch.as_tensor(S.TARGET_CUP_GEOMETRY.inner_bounds[1], device=dev)
        self._tcp_offset_pos = torch.tensor(cfg.tcp_offset_pos, device=dev).repeat(n, 1)
        self._tcp_offset_quat = torch.tensor(cfg.tcp_offset_rot, device=dev).repeat(n, 1)
        grasp_q = torch.as_tensor(S.CUP_GRASP_TCP_QUAT_C, device=dev, dtype=torch.float32)
        self._desired_grasp_tcp_quat_c = mu.quat_unique(grasp_q).repeat(n, 1)
        self._gripper_open_width = 2.0 * S.GRIPPER_OPEN_POSITION
        self._gripper_grasp_width = 2.0 * S.SOURCE_CUP_GEOMETRY.outer_half_extents[1]
        self._cup_reset_height = float(S.SOURCE_CUP_POSITION[2])
        self._source_outer_half_t = torch.tensor(S.SOURCE_CUP_GEOMETRY.outer_half_extents, device=dev)
        self._target_outer_half_t = torch.tensor(S.TARGET_CUP_GEOMETRY.outer_half_extents, device=dev)
        self._particle_workspace_lower_t = torch.as_tensor(cfg.particle_workspace_lower_bound, device=dev,
                                                           dtype=torch.float32)
        self._particle_workspace_upper_t = torch.as_tensor(cfg.particle_workspace_upper_bound, device=dev,
                                                           dtype=torch.float32)
        self._num_particles = P
        self._reset_dataset_states = load_reset_dataset(cfg.reset_dataset_path, dev)
        self.reset_dataset_row_id = torch.full((n,), -1, device=dev, dtype=torch.long)
        self.pour_target_frac = torch.full((n,), float(cfg.pour_target_frac), device=dev)
        self.episode_succeeded = torch.zeros(n, device=dev, dtype=torch.bool)
        self._success_dwell_count = torch.zeros(n, device=dev, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(n, device=dev, dtype=torch.long)
        self._lifted_grasp_seen = torch.zeros(n, device=dev, dtype=torch.bool)
        self._particle_region_cache = None
        self._particle_region_cache_step = -1
        self._capture_reset_particle_layout()

        # managers
        self.step_dt = STEP_DT
        self.max_episode_length = MAX_EPISODE_LENGTH
        self.max_episode_length_s = S.EPISODE_LENGTH_S
        self.episode_length_buf = torch.zeros(n, device=dev, dtype=torch.long)
        self.common_step_counter = 0
        self.arm_action = ArmAction(n, dev, cfg.arm_scale, cfg.arm_alpha)
        self.gripper_action = GripperAction(n, dev, self, cfg.gripper_alpha, cfg.gripper_close, cfg.gripper_neutral,
                                            cfg.gripper_default, cfg.gripper_contact_min_deflection)
        self.action = torch.zeros(n, 8, device=dev)
        self.prev_action = torch.zeros(n, 8, device=dev)
        self._term_dones = {k: torch.zeros(n, device=dev, dtype=torch.bool) for k in self.termination_terms}
        self._last_episode_dones = torch.zeros(n, len(self.termination_terms), device=dev, dtype=torch.bool)
        self.terminated = torch.zeros(n, device=dev, dtype=torch.bool)
        self.time_outs = torch.zeros(n, device=dev, dtype=torch.bool)
        self._episode_sums = {k: torch.zeros(n, device=dev) for k in self.reward_terms}
        self._step_reward = torch.zeros(n, len(self.reward_terms), device=dev)
        self.progress = LearningProgress(self)
        self.curriculum = ResetDatasetCurriculum(self)
        self.extras = {"log": {}}
        self.obs_buf = None

    # -- state accessors (Isaac's asset data, from Newton's state_0) ---------------------------------------------
    def joint_pos(self):
        return self._jq[:, self._q_idx]

    def joint_vel(self):
        return self._jqd[:, self._dof_idx]

    def finger_joint_pos(self):
        return self._jq[:, self._finger_q_idx]

    def finger_joint_vel(self):
        return self._jqd[:, self._finger_dof_idx]

    def hand_pose_w(self):
        return self._bq[:, self._tcp_body_idx]

    def robot_root_pose_w(self):
        return self._bq[:, self.info.robot_bodies[0]]

    def source_pose_w(self):
        return self._bq[:, self.info.source_body]

    def target_pose_w(self):
        return self._bq[:, self.info.target_body]

    def source_vel_w(self):
        """``root_link_vel_w`` (rigid_object_data.py:305-329, kernels.py:39-61): Newton's ``body_qd`` is the COM
        twist (v_com, omega) in world; the link-origin velocity is v_com + omega x R (-com_b)."""
        qd = self._bqd[:, self.info.source_body]
        q = self.source_pose_w()[:, 3:7]
        lin = qd[:, :3] + torch.linalg.cross(qd[:, 3:], mu.quat_apply(q, -self._source_com_b), dim=-1)
        return torch.cat((lin, qd[:, 3:]), dim=-1)

    def particle_vel_w(self):
        return self._pqd

    # -- pour_env.py:518-593 --------------------------------------------------------------------------------------
    def _pose_w_to_e(self, pose_w):
        pos = torch.nan_to_num(pose_w[:, :3], nan=0.0, posinf=0.0, neginf=0.0) - self.env_origins
        raw_quat = pose_w[:, 3:7]
        quat = torch.nan_to_num(raw_quat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        ident = torch.zeros_like(raw_quat)
        ident[:, 3] = 1.0
        valid = torch.isfinite(raw_quat).all(dim=-1, keepdim=True) & (norm > 1.0e-6)
        quat = torch.where(valid, quat / torch.clamp(norm, min=1.0e-6), ident)
        return torch.cat((pos, quat), dim=-1)

    def cup_pose_e(self):
        return self._pose_w_to_e(self.source_pose_w())

    def target_pose_e(self):
        return self._pose_w_to_e(self.target_pose_w())

    def tcp_pose_e(self):
        body_pose_w = self.hand_pose_w()
        root_pose_w = self.robot_root_pose_w()
        pos, quat = mu.subtract_frame_transforms(root_pose_w[:, :3], root_pose_w[:, 3:7], body_pose_w[:, :3],
                                                 body_pose_w[:, 3:7])
        pos, quat = mu.combine_frame_transforms(pos, quat, self._tcp_offset_pos, self._tcp_offset_quat)
        return torch.cat((torch.nan_to_num(pos), torch.nan_to_num(quat)), dim=-1)

    def cup_grasp_point_e(self):
        pose = self.cup_pose_e()
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = S.CUP_GRASP_HEIGHT
        return pose[:, :3] + mu.quat_apply(pose[:, 3:7], offset)

    def gripper_width(self):
        finger_pos = self.finger_joint_pos()
        width = finger_pos.sum(dim=-1)
        valid = torch.isfinite(finger_pos).all(dim=-1) & torch.isfinite(width)
        return torch.where(valid, width, torch.full_like(width, self._gripper_open_width))

    def particle_pos_e(self):
        return self._pq - self.env_origins[:, None, :]

    def _particle_region_masks(self):
        step = self.common_step_counter
        if self._particle_region_cache is not None and self._particle_region_cache_step == step:
            return self._particle_region_cache
        points = self.particle_pos_e()
        margin = float(self.cfg.particle_count_margin)
        source = points_inside_box(points, self.cup_pose_e(), self._source_inner_lo_t, self._source_inner_hi_t, margin)
        target_region = points_inside_box(points, self.target_pose_e(), self._target_inner_lo_t,
                                          self._target_inner_hi_t, margin)
        target = target_region & ~source
        spill_margin = max(margin, S.MPM_COLLIDER_MARGIN)
        spill_height = float(self.cfg.spill_table_height) + spill_margin
        spilled = ~(source | target) & (points[..., 2] <= spill_height)
        self._particle_region_cache = (source, target, spilled)
        self._particle_region_cache_step = step
        return source, target, spilled

    def _invalidate_caches(self):
        self._particle_region_cache = None
        self._particle_region_cache_step = -1

    def _capture_reset_particle_layout(self):
        """pour_env.py:429-449: env 0's emitted lattice in its source-cup frame, reused for every reset row."""
        particle_pos = self._pq[0]
        particle_vel = self._pqd[0]
        source_pose_w = self.source_pose_w()[0]
        source_quat = source_pose_w[None, 3:7].expand(self._num_particles, -1)
        local_position = mu.quat_apply_inverse(source_quat, particle_pos - source_pose_w[None, :3])
        local_velocity = mu.quat_apply_inverse(source_quat, particle_vel)
        lo, hi = self._source_inner_lo_t, self._source_inner_hi_t
        if not bool(((local_position >= lo) & (local_position <= hi)).all()):
            raise RuntimeError("Franka Pour generated particles outside the source-cup cavity.")
        self._reset_particle_local_position = local_position.clone()
        self._reset_particle_local_velocity = local_velocity.clone()

    # -- observations (mdp/observations.py) ----------------------------------------------------------------------
    def observations(self) -> dict[str, torch.Tensor]:
        def canonical(pose):
            pose = torch.nan_to_num(pose)
            return torch.cat((pose[:, :3], mu.quat_unique(pose[:, 3:7])), dim=-1)

        jp = self.joint_pos()
        jv = self.joint_vel()
        cup_pose = self.cup_pose_e()
        tcp_pose = self.tcp_pose_e()
        target_pose = self.target_pose_e()
        tcp_c = mu.quat_apply_inverse(cup_pose[:, 3:7], tcp_pose[:, :3] - cup_pose[:, :3])
        desired_c = torch.zeros_like(tcp_c)
        desired_c[:, 2] = S.CUP_GRASP_HEIGHT
        policy = [
            (jp[:, :7] - self.default_joint_pos[:, :7]) * 0.3,                       # arm_q, joint_pos_rel x 0.3
            (jv[:, :7] - 0.0) * 0.05,                                                 # arm_qd, joint_vel_rel x 0.05
            torch.clamp(1.0 - self.episode_length_buf.float() / max(int(self.max_episode_length), 1), 0.0,
                        1.0).unsqueeze(-1),                                           # time_remaining
            torch.nan_to_num(self.pour_target_frac).clamp(0.0, 1.0).unsqueeze(-1),    # pour_target_fraction
            canonical(tcp_pose), canonical(cup_pose), canonical(target_pose),
            torch.nan_to_num(desired_c - tcp_c) * 10.0,                               # tcp_to_grasp_position_c
            nearest_grasp_to_tcp_quat(self),                                          # grasp_to_tcp_quat
            torch.nan_to_num(mu.quat_apply_inverse(cup_pose[:, 3:7], target_pose[:, :3] - cup_pose[:, :3])) * 5.0,
            torch.nan_to_num(self.finger_joint_pos()) * 25.0,
            torch.nan_to_num(self.finger_joint_vel()) * 5.0,
            torch.nan_to_num(self.gripper_action.commanded_position) * 25.0,
            torch.nan_to_num(self.gripper_action.contact_deflection) * 250.0,
            self.action * 0.2,                                                        # last_action x 0.2
        ]
        vel = torch.nan_to_num(self.source_vel_w(), nan=0.0, posinf=0.0, neginf=0.0)
        cup_velocity = torch.cat((vel[:, :3] / 0.5, vel[:, 3:] * (0.13 / 0.5)), dim=-1).clamp(-2.0, 2.0)
        media = [cup_velocity, particle_fractions_obs(self), particle_source_state_obs(self),
                 particle_transfer_obs(self)]
        dwell_steps = max(1, math.ceil(float(self.cfg.success_dwell_time_s) / max(float(self.step_dt), 1.0e-6)))
        lost_steps = max(1, math.ceil(float(self.cfg.lost_grasp_dwell_time_s) / max(float(self.step_dt), 1.0e-6)))
        privileged = [torch.clamp(self._success_dwell_count.float() / dwell_steps, 0.0, 1.0).unsqueeze(-1),
                      torch.clamp(self._lost_grasp_dwell_count.float() / lost_steps, 0.0, 1.0).unsqueeze(-1)]
        return {"policy": torch.cat(policy, -1), "media": torch.cat(media, -1),
                "privileged": torch.cat(privileged, -1)}

    # -- terminations (mdp/terminations.py) ----------------------------------------------------------------------
    def _compute_terminations(self):
        n, dev = self.num_envs, self.device
        self.terminated = torch.zeros(n, device=dev, dtype=torch.bool)
        self.time_outs = torch.zeros(n, device=dev, dtype=torch.bool)
        for name in self.termination_terms:
            value = getattr(self, f"_term_{name}")()
            self._term_dones[name] = value
            if name == "time_out":
                self.time_outs |= value
            else:
                self.terminated |= value
        dones = self.terminated | self.time_outs
        rows = dones.nonzero().flatten()
        if rows.numel():
            self._last_episode_dones[rows] = torch.stack([self._term_dones[k] for k in self.termination_terms],
                                                         -1)[rows]
        return dones

    def _term_failure(self):
        cup_velocity = self.source_vel_w()
        finite = (torch.isfinite(self.joint_pos()).all(dim=-1) & torch.isfinite(self.joint_vel()).all(dim=-1)
                  & torch.isfinite(self.hand_pose_w()).all(dim=-1) & torch.isfinite(self.source_pose_w()).all(dim=-1)
                  & torch.isfinite(cup_velocity).all(dim=-1) & torch.isfinite(self._pq).all(dim=(1, 2)))
        return ~finite

    def _term_extreme_rigid_state(self):
        cfg = self.cfg
        joint_position, joint_velocity = self.joint_pos(), self.joint_vel()
        joint_lower = self._joint_pos_limits_t[..., 0] - float(cfg.state_bound_joint_position_margin)
        joint_upper = self._joint_pos_limits_t[..., 1] + float(cfg.state_bound_joint_position_margin)
        joint_ok = ((joint_position >= joint_lower) & (joint_position <= joint_upper)).all(dim=-1)
        joint_ok &= (joint_velocity.abs() <= float(cfg.state_bound_max_joint_velocity)).all(dim=-1)
        hand_pose = self.hand_pose_w()
        tcp_position, tcp_quaternion = mu.combine_frame_transforms(hand_pose[:, :3], hand_pose[:, 3:7],
                                                                   self._tcp_offset_pos, self._tcp_offset_quat)
        tcp_pose = torch.cat((tcp_position, tcp_quaternion), dim=-1)
        cup_pose = self.source_pose_w()
        lower, upper = self._particle_workspace_lower_t, self._particle_workspace_upper_t
        tcp_position_e = tcp_pose[:, :3] - self.env_origins
        cup_position_e = cup_pose[:, :3] - self.env_origins
        pose_ok = (((tcp_position_e >= lower) & (tcp_position_e <= upper)).all(dim=-1)
                   & ((cup_position_e >= lower) & (cup_position_e <= upper)).all(dim=-1)
                   & torch.isfinite(tcp_pose).all(dim=-1) & torch.isfinite(cup_pose).all(dim=-1)
                   & ((torch.linalg.vector_norm(tcp_pose[:, 3:7], dim=-1) - 1.0).abs() <= 0.1)
                   & ((torch.linalg.vector_norm(cup_pose[:, 3:7], dim=-1) - 1.0).abs() <= 0.1))
        cup_velocity = self.source_vel_w()
        velocity_ok = torch.linalg.vector_norm(cup_velocity[:, :3], dim=-1) <= float(
            cfg.state_bound_max_cup_linear_velocity)
        velocity_ok &= torch.linalg.vector_norm(cup_velocity[:, 3:], dim=-1) <= float(
            cfg.state_bound_max_cup_angular_velocity)
        return ~(joint_ok & pose_ok & velocity_ok)

    def _term_source_receiver_overlap(self):
        source_pose, target_pose = self.cup_pose_e(), self.target_pose_e()
        return oriented_boxes_overlap(box_center_from_base_pose(source_pose, self._source_outer_half_t),
                                      source_pose[:, 3:7], self._source_outer_half_t,
                                      box_center_from_base_pose(target_pose, self._target_outer_half_t),
                                      target_pose[:, 3:7], self._target_outer_half_t, self.cfg.overlap_clearance)

    def _term_lost_grasp(self):
        cfg = self.cfg
        cup_lift = self.cup_pose_e()[:, 2] - self._cup_reset_height
        tcp_distance = torch.linalg.vector_norm(self.tcp_pose_e()[:, :3] - self.cup_grasp_point_e(), dim=-1)
        gripper_width_error = torch.abs(self.gripper_width() - self._gripper_grasp_width)
        g = self.gripper_action
        reached_grasp = tcp_distance <= float(cfg.lost_grasp_max_tcp_distance)
        bilateral_grasp = (reached_grasp & (gripper_width_error <= float(cfg.lost_grasp_max_gripper_width_error))
                           & (g.commanded_position[:, 0] <= float(cfg.lost_grasp_max_gripper_command)
                              + _GRIPPER_POSITION_TOLERANCE)
                           & g.bilateral_contact)
        lifted_grasp = bilateral_grasp & (cup_lift >= max(float(cfg.success_min_lift_height), 1.0e-6))
        self._lifted_grasp_seen |= lifted_grasp
        lost = self._lifted_grasp_seen & ~bilateral_grasp
        dwell_steps = max(1, math.ceil(float(cfg.lost_grasp_dwell_time_s) / max(float(self.step_dt), 1.0e-6)))
        self._lost_grasp_dwell_count[:] = torch.where(lost, torch.clamp(self._lost_grasp_dwell_count + 1,
                                                                        max=dwell_steps), 0)
        return torch.zeros_like(lost)                              # terminate=False (pour_env_cfg.py:491)

    def _term_spill(self):
        spilled = self._particle_region_masks()[2].sum(dim=1) / max(self._num_particles, 1)
        return spilled > float(self.cfg.max_spill_fraction)

    def _term_particle_out_of_bounds(self):
        particles = self.particle_pos_e()
        return ~((particles >= self._particle_workspace_lower_t)
                 & (particles <= self._particle_workspace_upper_t)).all(dim=(1, 2))

    def _term_success(self):
        target_fraction = self._particle_region_masks()[1].sum(dim=1) / max(self._num_particles, 1)
        success = (target_fraction >= self.pour_target_frac) & ~self.terminated
        self._success_dwell_count[:] = success.to(dtype=self._success_dwell_count.dtype)
        self.episode_succeeded |= success
        return success

    def _term_learning_progress_context(self):
        return self.progress(self)

    def _term_time_out(self):
        deadline = self.episode_length_buf >= self.max_episode_length
        return deadline & ~self.episode_succeeded

    # -- rewards (mdp/rewards.py, reward_manager.py:146-152) ----------------------------------------------------
    def _compute_rewards(self):
        dt = self.step_dt
        success = self._term_dones["success"]
        values = {
            "success": 5.0 * success.float() / max(dt, 1.0e-6),
            "action_magnitude": -1.0e-4 * torch.sum(torch.square(self.action), dim=1),
            "action_rate": -1.0e-4 * torch.sum(torch.square(self.action - self.prev_action), dim=1),
            "failure": -1.0 * (self.terminated & ~success).float() / max(dt, 1.0e-6),
        }
        reward = torch.zeros(self.num_envs, device=self.device)
        for k, name in enumerate(self.reward_terms):
            v = values[name] * dt
            reward += v
            self._episode_sums[name] += v
            self._step_reward[:, k] = v / dt
        return reward

    # -- reset (manager_based_rl_env.py:406-451) ----------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        log = {}
        log.update(self.curriculum(self, env_ids))
        self._reset_pour_scene(env_ids)
        # managers in Isaac's order: observation (none stateful), action, reward, curriculum, termination
        self.action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.arm_action.reset(env_ids)
        self.gripper_action.reset(env_ids)
        for name in self.reward_terms:
            log["Episode_Reward/" + name] = float(torch.mean(self._episode_sums[name][env_ids])) / \
                self.max_episode_length_s
            self._episode_sums[name][env_ids] = 0.0
        for k, name in enumerate(self.termination_terms):
            log["Episode_Termination/" + name] = float(self._last_episode_dones[:, k].float().mean())
        self.progress.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.extras["log"] = log

    def _reset_pour_scene(self, env_ids: torch.Tensor):
        """``FrankaPourEnv._reset_from_dataset`` (pour_env.py:595-674)."""
        rows = self.reset_dataset_row_id[env_ids]
        st = self._reset_dataset_states
        arm_q, arm_qd = st["arm_joint_position"][rows], st["arm_joint_velocity"][rows]
        finger_q, finger_qd = st["finger_joint_position"][rows], st["finger_joint_velocity"][rows]
        finger_target = st["finger_joint_target"][rows]
        ids = env_ids[:, None]
        self._jq[ids, self._arm_q_idx[None]] = arm_q
        self._jqd[ids, self._arm_dof_idx[None]] = arm_qd
        self._target[ids, self._arm_dof_idx[None]] = arm_q
        self._jq[ids, self._finger_q_idx[None]] = finger_q
        self._jqd[ids, self._finger_dof_idx[None]] = finger_qd
        self._target[ids, self._finger_dof_idx[None]] = finger_target
        self.gripper_action.set_reset_position(finger_target[:, :1], env_ids)
        info = self.info
        source_pose = st["source_root_pose"][rows].clone()
        source_pose[:, :3] += self.env_origins[env_ids]
        target_pose = st["target_root_pose"][rows].clone()
        target_pose[:, :3] += self.env_origins[env_ids]
        sq = torch.arange(7, device=self.device)
        sd = torch.arange(6, device=self.device)
        self._jq[ids, (info.source_free_q + sq)[None]] = source_pose
        self._jqd[ids, (info.source_free_dof + sd)[None]] = st["source_root_velocity"][rows]
        self._jq[ids, (info.target_free_q + sq)[None]] = target_pose
        self._jqd[ids, (info.target_free_dof + sd)[None]] = st["target_root_velocity"][rows]
        particle_count = self._num_particles
        local_position = self._reset_particle_local_position[None].expand(len(env_ids), -1, -1)
        local_velocity = self._reset_particle_local_velocity[None].expand(len(env_ids), -1, -1)
        source_quat = source_pose[:, None, 3:7].expand(-1, particle_count, -1)
        particle_position = mu.quat_apply(source_quat, local_position) + source_pose[:, None, :3]
        particle_velocity = mu.quat_apply(source_quat, local_velocity)
        for pq, pqd in ((self._pq, self._pqd), (self._pq1, self._pqd1)):   # both state buffers (mpm_object.py:398)
            pq[env_ids] = particle_position
            pqd[env_ids] = particle_velocity
        torch.mps.synchronize() if self.device.type == "mps" else None
        self.physics.finish_reset(env_ids.cpu().numpy())
        wp.synchronize_device(self.physics.device)
        self._invalidate_caches()
        self.episode_succeeded[env_ids] = False
        self._success_dwell_count[env_ids] = 0
        self._lost_grasp_dwell_count[env_ids] = 0
        self._lifted_grasp_seen[env_ids] = st["category"][rows] == GRASPING_CATEGORY

    def reset_all(self):
        """``env.reset()``: every env from a curriculum row, then the first observation."""
        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(ids)
        self.obs_buf = self.observations()
        return self.obs_buf

    # -- step (manager_based_rl_env.py:197-308) -----------------------------------------------------------------
    def step(self, action: torch.Tensor):
        self.process_action(action)
        self.simulate()
        return self.post_physics()

    def process_action(self, action: torch.Tensor):
        """``ActionManager.process_action`` after the wrapper's clip; hands the processed terms to the physics."""
        action = torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)   # RslRlVecEnvWrapper
        self.prev_action[:] = self.action
        self.action[:] = action.to(self.device)
        self.arm_action.process_actions(self.action[:, :7])
        self.gripper_action.process_actions(self.action[:, 7:8])
        self._arm_delta.copy_(self.arm_action.processed_actions)
        self._gripper_target.copy_(self.gripper_action.processed_actions[:, 0])

    def simulate(self):
        """The decimation loop (4 x [apply_action, sim.step]) as graph replays on the Warp queue."""
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.physics.policy_step()
        wp.synchronize_device(self.physics.device)

    def post_physics(self):
        """Counters, terminations, rewards, resets of done envs, observations (manager_based_rl_env.py:248-308)."""
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._invalidate_caches()
        dones = self._compute_terminations()
        reward = self._compute_rewards()
        terminated, time_outs = self.terminated.clone(), self.time_outs.clone()
        reset_ids = dones.nonzero().flatten()
        if reset_ids.numel():
            self._reset_idx(reset_ids)
        self.obs_buf = self.observations()
        self.extras["time_outs"] = time_outs
        return self.obs_buf, reward, terminated, time_outs, self.extras


# ------------------------------------------------------------------------------------------------ observation helpers
def nearest_grasp_to_tcp_quat(env) -> torch.Tensor:
    """``_nearest_grasp_to_tcp_quat`` (mdp/observations.py:46-87)."""
    cup_quat = env.cup_pose_e()[:, 3:7]
    tcp_quat = env.tcp_pose_e()[:, 3:7]
    base_grasp = env._desired_grasp_tcp_quat_c
    side_angles = torch.arange(4, device=cup_quat.device, dtype=cup_quat.dtype) * (0.5 * math.pi)
    side_axes = torch.zeros((4, 3), device=cup_quat.device, dtype=cup_quat.dtype)
    side_axes[:, 2] = 1.0
    side_rotations = mu.quat_from_angle_axis(side_angles, side_axes)
    side_rotations = side_rotations.unsqueeze(0).expand(cup_quat.shape[0], -1, -1)
    base_grasps = base_grasp.unsqueeze(1).expand_as(side_rotations)
    cup_quaternions = cup_quat.unsqueeze(1).expand_as(side_rotations)
    tcp_quaternions = tcp_quat.unsqueeze(1).expand_as(side_rotations)
    desired = mu.quat_mul(cup_quaternions, mu.quat_mul(side_rotations, base_grasps))
    error_quaternions = mu.quat_unique(mu.quat_mul(mu.quat_conjugate(desired), tcp_quaternions))
    error_magnitudes = torch.linalg.vector_norm(mu.axis_angle_from_quat(error_quaternions), dim=-1)
    error_magnitudes = torch.nan_to_num(error_magnitudes, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
    nearest_side = torch.argmin(error_magnitudes, dim=1)
    nearest_error = torch.gather(error_quaternions, dim=1, index=nearest_side[:, None, None].expand(-1, 1, 4)).squeeze(1)
    return torch.nan_to_num(mu.quat_unique(nearest_error))


def particle_fractions_obs(env):
    scale = max(env._num_particles, 1)
    return torch.stack([mask.sum(dim=1).float() / scale for mask in env._particle_region_masks()], dim=-1)


def _masked_particle_moments(env, mask):
    weight = mask.unsqueeze(-1).to(dtype=torch.float32)
    count = weight.sum(dim=1)
    denominator = count.clamp_min(1.0)
    centroid = (env.particle_pos_e() * weight).sum(dim=1) / denominator
    velocity = (env.particle_vel_w() * weight).sum(dim=1) / denominator
    return centroid, velocity, count


def particle_source_state_obs(env):
    source = env._particle_region_masks()[0]
    centroid, velocity, count = _masked_particle_moments(env, source)
    cup_pose = env.cup_pose_e()
    centroid_offset = centroid - cup_pose[:, :3]
    cup_velocity = env.source_vel_w()
    rigid_point_velocity = cup_velocity[:, :3] + torch.linalg.cross(cup_velocity[:, 3:], centroid_offset, dim=-1)
    centroid_c = mu.quat_apply_inverse(cup_pose[:, 3:7], centroid_offset)
    velocity_c = mu.quat_apply_inverse(cup_pose[:, 3:7], velocity - rigid_point_velocity)
    present = count > 0.0
    summary = torch.cat((centroid_c / 0.12, velocity_c / 2.0), dim=-1)
    summary = torch.where(present.expand_as(summary), summary, torch.zeros_like(summary))
    return torch.nan_to_num(summary).clamp_(-2.0, 2.0)


def particle_transfer_obs(env):
    source, target, spilled = env._particle_region_masks()
    airborne = ~(source | target | spilled)
    centroid, velocity, count = _masked_particle_moments(env, airborne)
    receiver_pose = env.target_pose_e()
    centroid_relative = mu.quat_apply_inverse(receiver_pose[:, 3:7], centroid - receiver_pose[:, :3])
    velocity = mu.quat_apply_inverse(receiver_pose[:, 3:7], velocity)
    centroid_relative = torch.where(count > 0.0, centroid_relative, torch.zeros_like(centroid))
    velocity = torch.where(count > 0.0, velocity, torch.zeros_like(velocity))
    fraction = count / max(env._num_particles, 1)
    summary = torch.cat((centroid_relative / 0.30, velocity / 2.0, fraction), dim=-1)
    return torch.nan_to_num(summary).clamp_(-2.0, 2.0)


# ------------------------------------------------------------------------------------------------ curriculum
def _open_quality(env):
    travel = max(env._gripper_open_width - env._gripper_grasp_width, 1.0e-4)
    return torch.clamp((env.gripper_width() - env._gripper_grasp_width) / travel, 0.0, 1.0)


def tcp_cup_grasp_pose_tanh(env, position_std=0.20, orientation_std=0.75, open_hand_fraction=0.35):
    """mdp/rewards.py:43-73."""
    distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.cup_grasp_point_e(), dim=-1)
    position_quality = 1.0 - torch.tanh(distance / float(position_std))
    error_quat = nearest_grasp_to_tcp_quat(env)
    orientation_error = torch.linalg.vector_norm(mu.axis_angle_from_quat(error_quat), dim=-1)
    orientation_quality = 1.0 - torch.tanh(orientation_error / float(orientation_std))
    pose_quality = position_quality * (0.5 + 0.5 * orientation_quality)
    potential = pose_quality + float(open_hand_fraction) * (1.0 - pose_quality) * _open_quality(env)
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _grasp_width_quality(env, preload_position):
    width = env.gripper_width()
    open_width, grasp_width = env._gripper_open_width, env._gripper_grasp_width
    travel = max(open_width - grasp_width, 1.0e-4)
    close_progress = torch.clamp((open_width - width) / travel, 0.0, 1.0)
    open_position = 0.5 * open_width
    preload_travel = max(open_position - float(preload_position), 1.0e-4)
    command = env.gripper_action.commanded_position[:, 0]
    commanded_preload = torch.clamp((open_position - command) / preload_travel, 0.0, 1.0)
    return torch.minimum(close_progress, commanded_preload) * env.gripper_action.contact_quality


def _grasp_lift_potential(env, target_height, grasp_reach_std, grasp_preload_position, grasp_fraction):
    distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.cup_grasp_point_e(), dim=-1)
    proximity = torch.clamp(1.0 - distance / float(grasp_reach_std), 0.0, 1.0)
    proximity = proximity.square() * (3.0 - 2.0 * proximity)
    held = proximity * _grasp_width_quality(env, grasp_preload_position)
    height = torch.clamp((env.cup_pose_e()[:, 2] - env._cup_reset_height) / float(target_height), 0.0, 1.0)
    potential = held * (float(grasp_fraction) + (1.0 - float(grasp_fraction)) * height)
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def reset_dataset_physical_potential(env, approach_position_std, approach_orientation_std,
                                     approach_open_hand_fraction, grasp_target_height, grasp_reach_std,
                                     grasp_preload_position, grasp_fraction, source_mouth_height, target_rim_height,
                                     target_clearance_height, alignment_std, tilt_alignment_radius, target_tilt):
    """mdp/rewards.py:114-170."""
    approach = tcp_cup_grasp_pose_tanh(env, position_std=approach_position_std,
                                       orientation_std=approach_orientation_std,
                                       open_hand_fraction=approach_open_hand_fraction)
    grasp_lift = _grasp_lift_potential(env, target_height=grasp_target_height, grasp_reach_std=grasp_reach_std,
                                       grasp_preload_position=grasp_preload_position, grasp_fraction=grasp_fraction)
    source_pose, target_pose = env.cup_pose_e(), env.target_pose_e()
    local_mouth = torch.zeros_like(source_pose[:, :3])
    local_mouth[:, 2] = float(source_mouth_height)
    mouth_position = source_pose[:, :3] + mu.quat_apply(source_pose[:, 3:7], local_mouth)
    target_rim_position = target_pose[:, :3].clone()
    target_rim_position[:, 2] += float(target_rim_height)
    target_offset_xy = target_rim_position[:, :2] - mouth_position[:, :2]
    target_distance_xy = torch.linalg.vector_norm(target_offset_xy, dim=-1)
    clearance = mouth_position[:, 2] - target_rim_position[:, 2]
    clearance_quality = torch.clamp(clearance / float(target_clearance_height), 0.0, 1.0)
    alignment_quality = 1.0 - torch.tanh(target_distance_xy / float(alignment_std))
    alignment = grasp_lift * clearance_quality * alignment_quality
    local_open_axis = torch.zeros_like(source_pose[:, :3])
    local_open_axis[:, 2] = 1.0
    open_axis = mu.quat_apply(source_pose[:, 3:7], local_open_axis)
    tilt_angle = torch.acos(torch.clamp(open_axis[:, 2], -1.0, 1.0))
    tilt_quality = torch.clamp(tilt_angle / float(target_tilt), 0.0, 1.0)
    proximity = torch.clamp(1.0 - target_distance_xy / float(tilt_alignment_radius), 0.0, 1.0)
    tilt_alignment = proximity.square() * (3.0 - 2.0 * proximity)
    tilt = grasp_lift * clearance_quality * tilt_alignment * tilt_quality
    potential = 0.20 * approach + 0.30 * grasp_lift + 0.25 * alignment + 0.25 * tilt
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class LearningProgress:
    """``PourResetLearningProgress`` (mdp/rewards.py:173-262): latches local physical progress for the curriculum;
    never terminates and never rewards."""

    _milestones = (0.20, 0.32, 0.50, 0.75, 0.95, 1.00)

    def __init__(self, env: FrankaPourEnv):
        n, dev = env.num_envs, env.device
        self._env = env
        self._target_potential = torch.ones(n, device=dev)
        self._requires_terminal_success = torch.zeros(n, device=dev, dtype=torch.bool)
        self.is_success = torch.zeros(n, device=dev, dtype=torch.bool)
        self.new_success = torch.zeros_like(self.is_success)
        self.ever_success = torch.zeros_like(self.is_success)
        self._no_termination = torch.zeros_like(self.is_success)

    def reset(self, env_ids):
        env = self._env
        initial = reset_dataset_physical_potential(env, **env.cfg.potential_params)[env_ids]
        rows = env.reset_dataset_row_id[env_ids]
        row_progress = env._reset_dataset_states["difficulty"][rows]
        milestones = initial.new_tensor(self._milestones)
        next_index = torch.searchsorted(milestones, row_progress + 1.0e-6)
        next_index.clamp_(max=milestones.numel() - 1)
        target = torch.maximum(initial + float(env.cfg.progress_minimum), milestones[next_index]).clamp_(max=1.0)
        self._target_potential[env_ids] = target
        self._requires_terminal_success[env_ids] = target >= 1.0 - 1.0e-6
        self.is_success[env_ids] = False
        self.new_success[env_ids] = False
        self.ever_success[env_ids] = False

    def __call__(self, env):
        current = reset_dataset_physical_potential(env, **env.cfg.potential_params)
        reached_potential = ((current >= self._target_potential) & ~self._requires_terminal_success
                             & (env.episode_length_buf >= int(env.cfg.progress_minimum_episode_steps)))
        unsafe_terminal = env.terminated & ~env.episode_succeeded
        reached_potential &= ~unsafe_terminal
        reached = reached_potential | env.episode_succeeded
        self.is_success.copy_(reached)
        self.new_success.copy_(reached & ~self.ever_success)
        self.ever_success |= reached
        return self._no_termination


class ResetDatasetCurriculum:
    """``PourResetDatasetCurriculum`` (mdp/reset_dataset.py:22-129)."""

    def __init__(self, env: FrankaPourEnv):
        st = env._reset_dataset_states
        self._row_count = int(st["category"].numel())
        self._device = st["category"].device
        self._sampler = ResetDatasetSampler(self._row_count, self._device, env.cfg.sampler, env.generator)
        self._frozen_rows = torch.arange(self._row_count, device=self._device, dtype=torch.long)
        self._metrics_cache = self._sampler.metrics()
        self._since_metrics = 0
        self._refresh_interval = max(1, env.num_envs)

    def __call__(self, env: FrankaPourEnv, env_ids=None):
        ids = env_ids
        completed = (env.episode_length_buf[ids] > 0) & (env.reset_dataset_row_id[ids] >= 0)
        completed_ids = ids[completed]
        if completed_ids.numel() > 0 and not env.cfg.curriculum_freeze:
            self._sampler.record(env.reset_dataset_row_id[completed_ids], env.progress.ever_success[completed_ids])
        if env.cfg.curriculum_freeze:
            slots = torch.randint(self._frozen_rows.numel(), (ids.numel(),), generator=env.generator).to(self._device)
            rows = self._frozen_rows[slots]
        elif env.cfg.reset_dataset_sampling_mode == "uniform":
            rows = torch.randint(self._row_count, (ids.numel(),), generator=env.generator).to(self._device)
        else:
            rows = self._sampler.sample_with_uniform_replay(ids.numel())
        env.reset_dataset_row_id[ids] = rows
        env.pour_target_frac[ids] = float(env.cfg.pour_target_frac)
        if not env.cfg.curriculum_freeze:
            self._since_metrics += int(ids.numel())
            if self._since_metrics >= self._refresh_interval:
                self._metrics_cache = self._sampler.metrics()
                self._since_metrics = 0
        return dict(self._metrics_cache)
