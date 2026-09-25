"""Isaac Lab event terms (``isaaclab.envs.mdp.events``) on a ``BatchSim``.

Each term is a function ``term(ctx, env_ids, **params)`` with Isaac Lab's parameter names and
semantics, writing per-world MuJoCo Warp tensors (``sim.t`` data, ``sim.tm`` model fields) on the MPS
device; ``EventManager`` runs them in Isaac's modes (``startup``, ``reset``, ``interval`` with per-env
timers). Ordering is the caller's, as for any torch write into the sim: ``sim.after(v)`` before the
term, and ``sim.wait`` on a torch-signalled event (or a synchronize) before the next physics launch.

Mapping from PhysX to MuJoCo (the differences are in the physics, not in the sampling):

* material: PhysX static/dynamic friction and restitution per shape; MuJoCo has one Coulomb
  coefficient per geom (``geom_friction[..., 0]``), set to the sampled *static* friction (as
  ``metalsim.learn.g1_velocity`` does); dynamic friction and restitution are sampled and kept in
  ``ctx.material`` but have no MuJoCo counterpart.
* mass/inertia/COM: ``body_mass``, ``body_inertia``, ``body_ipos``; ``sim.recompute_constants()`` runs
  after them.
* actuator gains: position actuators (``gainprm[0] = kp``, ``biasprm[1] = -kp``, ``biasprm[2] = -kd``).
* joint friction/armature/limits: ``dof_frictionloss``, ``dof_armature``, ``jnt_range``.
* external wrench: Isaac applies the force/torque in the body frame until the next reset; here a
  substep hook rotates the stored body-frame wrench into ``xfrc_applied`` (world frame, at the COM).
* root velocities: Isaac's root velocity is the COM velocity in the world frame; MuJoCo's free joint
  stores the body-origin linear velocity (world) and the angular velocity in the body frame. The
  terms convert.

The fields a term writes must be per-world: build the sim with
``BatchSimOptions(per_world_fields=REQUIRED_FIELDS[...])``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp

REQUIRED_FIELDS = {
    "randomize_rigid_body_material": ("geom_friction",),
    "randomize_rigid_body_mass": ("body_mass", "body_inertia"),
    "randomize_rigid_body_com": ("body_ipos",),
    "randomize_actuator_gains": ("actuator_gainprm", "actuator_biasprm"),
    "randomize_joint_parameters": ("dof_frictionloss", "dof_armature", "jnt_range"),
}


# -- math -------------------------------------------------------------------------------------------------

def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_from_euler_xyz(roll, pitch, yaw):
    """Isaac Lab's convention (extrinsic XYZ = q_z * q_y * q_x), quaternion wxyz."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                        cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp], -1)


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def _sample(lo, hi, shape, gen, dev, distribution="uniform"):
    lo = torch.as_tensor(lo, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(hi, dtype=torch.float32, device=dev)
    if distribution == "uniform":
        return lo + (hi - lo) * torch.rand(shape, generator=gen, device=dev)
    if distribution == "log_uniform":
        return torch.exp(torch.log(lo) + (torch.log(hi) - torch.log(lo)) * torch.rand(shape, generator=gen, device=dev))
    if distribution == "gaussian":   # Isaac: params are (mean, std)
        return lo + hi * torch.randn(shape, generator=gen, device=dev)
    raise ValueError(distribution)


def _by_op(default, params, operation, distribution, gen):
    """Isaac Lab ``_randomize_prop_by_op``: ``add`` / ``scale`` / ``abs`` on the default value."""
    s = _sample(params[0], params[1], default.shape, gen, default.device, distribution)
    if operation == "add":
        return default + s
    if operation == "scale":
        return default * s
    if operation == "abs":
        return s
    raise ValueError(operation)


# -- context --------------------------------------------------------------------------------------------

@dataclass
class AssetCfg:
    """``SceneEntityCfg`` equivalent: regexes over MuJoCo body / joint names (None = all)."""
    body_names: str | list | None = None
    joint_names: str | list | None = None


@dataclass
class EventContext:
    sim: object
    seed: int = 0
    root_body: int | None = None      # the articulation root (free-joint body); default: first free joint
    default_qpos: torch.Tensor | None = None
    env_origins: torch.Tensor | None = None
    soft_joint_pos_limit_factor: float = 1.0
    defaults: dict = field(default_factory=dict)
    material: dict = field(default_factory=dict)
    ext_wrench: torch.Tensor | None = None    # (N, nbody, 6) body-frame force, torque
    _hooked: bool = False

    def __post_init__(self):
        m = self.m
        self.dev = torch.device("mps")
        self.gen = torch.Generator(device="mps").manual_seed(self.seed)
        if self.root_body is None:
            free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
            self.root_body = int(m.jnt_bodyid[free[0]]) if free else None
        if self.default_qpos is None:
            self.default_qpos = torch.as_tensor(m.qpos0.astype(np.float32), device=self.dev)
        self.root_qadr = self.root_dadr = None
        if self.root_body is not None:
            j = int(m.body_jntadr[self.root_body])
            self.root_qadr, self.root_dadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        # 1-dof joints (hinge/slide): the articulation's "joints" in Isaac's sense
        self.joints = [j for j in range(m.njnt) if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]

    @property
    def m(self) -> mujoco.MjModel:
        return self.sim.mj_model

    @property
    def n(self) -> int:
        return self.sim.n

    def field(self, name: str) -> torch.Tensor:
        t = getattr(self.sim.tm, name)
        if t.stride(0) == 0 or t.shape[0] != self.n:
            raise ValueError(f"model field {name!r} is shared across worlds: add it to BatchSimOptions.per_world_fields")
        return t

    def default(self, name: str) -> torch.Tensor:
        """The field's value before any randomization (captured on first use)."""
        if name not in self.defaults:
            self.defaults[name] = self.field(name).clone()
        return self.defaults[name]

    def bodies(self, cfg: AssetCfg | None) -> list:
        return _select(self.m, mujoco.mjtObj.mjOBJ_BODY, range(1, self.m.nbody), None if cfg is None else cfg.body_names)

    def joint_ids(self, cfg: AssetCfg | None) -> list:
        return _select(self.m, mujoco.mjtObj.mjOBJ_JOINT, self.joints, None if cfg is None else cfg.joint_names)

    def env_index(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.n, device=self.dev)
        env_ids = torch.as_tensor(env_ids, device=self.dev)
        return torch.nonzero(env_ids).flatten() if env_ids.dtype == torch.bool else env_ids.long()


def _select(m, objtype, ids, names):
    ids = list(ids)
    if names is None:
        return ids
    import re
    pats = [names] if isinstance(names, str) else list(names)
    return [i for i in ids if any(re.fullmatch(p, mujoco.mj_id2name(m, objtype, i) or "") for p in pats)]


# -- startup-type terms (model randomization) --------------------------------------------------------

def randomize_rigid_body_material(ctx: EventContext, env_ids, asset_cfg: AssetCfg | None = None,
                                  static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0),
                                  restitution_range=(0.0, 0.0), num_buckets: int = 1, make_consistent: bool = False):
    """Isaac: ``num_buckets`` materials sampled uniformly, each shape of each env assigned a random
    bucket. MuJoCo: the geom's sliding friction := the bucket's static friction."""
    envs = ctx.env_index(env_ids)
    m = ctx.m
    bodies = set(ctx.bodies(asset_cfg))
    geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies]
    if not geoms:
        return
    lo = torch.tensor([static_friction_range[0], dynamic_friction_range[0], restitution_range[0]], device=ctx.dev)
    hi = torch.tensor([static_friction_range[1], dynamic_friction_range[1], restitution_range[1]], device=ctx.dev)
    buckets = lo + (hi - lo) * torch.rand(num_buckets, 3, generator=ctx.gen, device=ctx.dev)
    if make_consistent:
        buckets[:, 1] = torch.minimum(buckets[:, 0], buckets[:, 1])
    ids = torch.randint(0, num_buckets, (len(envs), len(geoms)), generator=ctx.gen, device=ctx.dev)
    mats = buckets[ids]                                                     # (E, G, 3)
    fr = ctx.field("geom_friction")
    gi = torch.as_tensor(geoms, dtype=torch.long, device=ctx.dev)
    fr[envs[:, None], gi[None, :], 0] = mats[..., 0]
    if "values" not in ctx.material:
        ctx.material["values"] = torch.zeros(ctx.n, m.ngeom, 3, device=ctx.dev)
    ctx.material["values"][envs[:, None], gi[None, :]] = mats


def randomize_rigid_body_mass(ctx: EventContext, env_ids, asset_cfg: AssetCfg | None = None,
                              mass_distribution_params=(1.0, 1.0), operation="scale", distribution="uniform",
                              recompute_inertia: bool = True, min_mass: float = 1e-6):
    """Isaac: per (env, body) sample on the default mass by ``operation``; inertia scaled by the mass
    ratio when ``recompute_inertia``. Call ``sim.recompute_constants()`` after."""
    envs = ctx.env_index(env_ids)
    b = torch.as_tensor(ctx.bodies(asset_cfg), dtype=torch.long, device=ctx.dev)
    m0 = ctx.default("body_mass")[envs[:, None], b[None, :]]
    new = torch.clamp(_by_op(m0, mass_distribution_params, operation, distribution, ctx.gen), min=min_mass)
    ctx.field("body_mass")[envs[:, None], b[None, :]] = new
    if recompute_inertia:
        i0 = ctx.default("body_inertia")[envs[:, None], b[None, :]]
        ctx.field("body_inertia")[envs[:, None], b[None, :]] = i0 * (new / m0).unsqueeze(-1)


def randomize_rigid_body_com(ctx: EventContext, env_ids, com_range: dict, asset_cfg: AssetCfg | None = None):
    """Isaac: one uniform offset per env (x, y, z ranges) added to the COM of the selected bodies."""
    envs = ctx.env_index(env_ids)
    b = torch.as_tensor(ctx.bodies(asset_cfg), dtype=torch.long, device=ctx.dev)
    r = [com_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")]
    off = _sample([a for a, _ in r], [c for _, c in r], (len(envs), 3), ctx.gen, ctx.dev)
    ctx.field("body_ipos")[envs[:, None], b[None, :]] = ctx.default("body_ipos")[envs[:, None], b[None, :]] + off[:, None, :]


def randomize_actuator_gains(ctx: EventContext, env_ids, asset_cfg: AssetCfg | None = None,
                             stiffness_distribution_params=None, damping_distribution_params=None,
                             operation="abs", distribution="uniform"):
    """Isaac: per (env, joint) stiffness / damping of the joint drives. MuJoCo: the position
    actuators on the selected joints."""
    envs = ctx.env_index(env_ids)
    m = ctx.m
    joints = set(ctx.joint_ids(asset_cfg))
    acts = [a for a in range(m.nu) if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and int(m.actuator_trnid[a, 0]) in joints]
    if not acts:
        return
    a = torch.as_tensor(acts, dtype=torch.long, device=ctx.dev)
    g, bprm = ctx.field("actuator_gainprm"), ctx.field("actuator_biasprm")
    if stiffness_distribution_params is not None:
        kp = _by_op(ctx.default("actuator_gainprm")[envs[:, None], a[None, :], 0], stiffness_distribution_params, operation, distribution, ctx.gen)
        g[envs[:, None], a[None, :], 0] = kp
        bprm[envs[:, None], a[None, :], 1] = -kp
    if damping_distribution_params is not None:
        kd = _by_op(-ctx.default("actuator_biasprm")[envs[:, None], a[None, :], 2], damping_distribution_params, operation, distribution, ctx.gen)
        bprm[envs[:, None], a[None, :], 2] = -kd


def randomize_joint_parameters(ctx: EventContext, env_ids, asset_cfg: AssetCfg | None = None,
                               friction_distribution_params=None, armature_distribution_params=None,
                               lower_limit_distribution_params=None, upper_limit_distribution_params=None,
                               operation="abs", distribution="uniform"):
    envs = ctx.env_index(env_ids)
    m = ctx.m
    j = ctx.joint_ids(asset_cfg)
    d = torch.as_tensor([int(m.jnt_dofadr[k]) for k in j], dtype=torch.long, device=ctx.dev)
    jt = torch.as_tensor(j, dtype=torch.long, device=ctx.dev)
    for name, params in (("dof_frictionloss", friction_distribution_params), ("dof_armature", armature_distribution_params)):
        if params is not None:
            ctx.field(name)[envs[:, None], d[None, :]] = _by_op(ctx.default(name)[envs[:, None], d[None, :]], params, operation, distribution, ctx.gen)
    for k, params in ((0, lower_limit_distribution_params), (1, upper_limit_distribution_params)):
        if params is not None:
            ctx.field("jnt_range")[envs[:, None], jt[None, :], k] = _by_op(ctx.default("jnt_range")[envs[:, None], jt[None, :], k], params, operation, distribution, ctx.gen)


def randomize_physics_scene_gravity(ctx: EventContext, env_ids, gravity_distribution_params, operation="add",
                                    distribution="uniform"):
    """Isaac: gravity is a scene property, one sample for all envs (``env_ids`` ignored)."""
    from metalsim.interop import torch_bridge as tb
    g = tb.mps_tensor(ctx.sim.m.opt.gravity)       # (1, 3)
    if "gravity" not in ctx.defaults:
        ctx.defaults["gravity"] = g.clone()
    lo = torch.as_tensor(gravity_distribution_params[0], dtype=torch.float32, device=ctx.dev)
    hi = torch.as_tensor(gravity_distribution_params[1], dtype=torch.float32, device=ctx.dev)
    g0 = ctx.defaults["gravity"][0]
    s = _sample(lo, hi, (3,), ctx.gen, ctx.dev, distribution) if distribution != "gaussian" else lo + hi * torch.randn(3, generator=ctx.gen, device=ctx.dev)
    g[0] = g0 + s if operation == "add" else (g0 * s if operation == "scale" else s)


# -- reset / interval-type terms (state) ------------------------------------------------------------

def reset_root_state_uniform(ctx: EventContext, env_ids, pose_range: dict, velocity_range: dict, asset_cfg=None):
    """Isaac: default root pose + env origin + uniform (x, y, z) offset, orientation = default *
    euler_xyz(roll, pitch, yaw); root velocity (world COM, lin + ang) = default + uniform."""
    envs = ctx.env_index(env_ids)
    q, v = ctx.sim.t.qpos, ctx.sim.t.qvel
    qa, da = ctx.root_qadr, ctx.root_dadr
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    pr = [pose_range.get(k, (0.0, 0.0)) for k in keys]
    vr = [velocity_range.get(k, (0.0, 0.0)) for k in keys]
    rp = _sample([a for a, _ in pr], [b for _, b in pr], (len(envs), 6), ctx.gen, ctx.dev)
    rv = _sample([a for a, _ in vr], [b for _, b in vr], (len(envs), 6), ctx.gen, ctx.dev)
    d0 = ctx.default_qpos
    pos = d0[qa:qa + 3] + rp[:, 0:3]
    if ctx.env_origins is not None:
        pos = pos + ctx.env_origins[envs]
    quat = quat_mul(d0[qa + 3:qa + 7].expand(len(envs), 4), quat_from_euler_xyz(rp[:, 3], rp[:, 4], rp[:, 5]))
    q[envs, qa:qa + 3] = pos
    q[envs, qa + 3:qa + 7] = quat
    _set_root_com_velocity(ctx, envs, quat_to_mat(quat), rv[:, 0:3], rv[:, 3:6])


def _set_root_com_velocity(ctx, envs, R, lin_com_w, ang_w):
    """Write a world-frame COM velocity into MuJoCo's free-joint qvel (origin linear velocity in the
    world frame, angular velocity in the body frame)."""
    v = ctx.sim.t.qvel
    da = ctx.root_dadr
    ipos = ctx.sim.tm.body_ipos
    ipos = ipos[envs, ctx.root_body] if ipos.shape[0] == ctx.n and ipos.stride(0) != 0 else ipos[0, ctx.root_body].expand(len(envs), 3)
    r_com = torch.einsum("nij,nj->ni", R, ipos)        # origin -> COM, world frame
    v[envs, da:da + 3] = lin_com_w - torch.cross(ang_w, r_com, dim=-1)
    v[envs, da + 3:da + 6] = torch.einsum("nji,nj->ni", R, ang_w)


def push_by_setting_velocity(ctx: EventContext, env_ids, velocity_range: dict, asset_cfg=None):
    """Isaac: root COM velocity (world) += uniform sample over (x, y, z, roll, pitch, yaw)."""
    envs = ctx.env_index(env_ids)
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    vr = [velocity_range.get(k, (0.0, 0.0)) for k in keys]
    dv = _sample([a for a, _ in vr], [b for _, b in vr], (len(envs), 6), ctx.gen, ctx.dev)
    v = ctx.sim.t.qvel
    da, b = ctx.root_dadr, ctx.root_body
    R = ctx.sim.t.xmat[envs, b].reshape(-1, 3, 3)
    w_w = torch.einsum("nij,nj->ni", R, v[envs, da + 3:da + 6])
    r_com = ctx.sim.t.xipos[envs, b] - ctx.sim.t.xpos[envs, b]
    lin_com = v[envs, da:da + 3] + torch.cross(w_w, r_com, dim=-1)
    _set_root_com_velocity(ctx, envs, R, lin_com + dv[:, 0:3], w_w + dv[:, 3:6])


def _joint_limits(ctx, joints, envs):
    m = ctx.m
    jr = ctx.sim.tm.jnt_range
    jt = torch.as_tensor(joints, dtype=torch.long, device=ctx.dev)
    lim = jr[envs[:, None], jt[None, :]] if jr.shape[0] == ctx.n else jr[0, jt].expand(len(envs), -1, 2)
    limited = torch.as_tensor([bool(m.jnt_limited[j]) for j in joints], device=ctx.dev)
    mid, half = lim.mean(-1), 0.5 * (lim[..., 1] - lim[..., 0]) * ctx.soft_joint_pos_limit_factor
    lo = torch.where(limited, mid - half, torch.full_like(mid, -math.inf))
    hi = torch.where(limited, mid + half, torch.full_like(mid, math.inf))
    return lo, hi


def _reset_joints(ctx, env_ids, asset_cfg, position_range, velocity_range, scale: bool):
    envs = ctx.env_index(env_ids)
    m = ctx.m
    j = ctx.joint_ids(asset_cfg)
    qa = torch.as_tensor([int(m.jnt_qposadr[k]) for k in j], dtype=torch.long, device=ctx.dev)
    da = torch.as_tensor([int(m.jnt_dofadr[k]) for k in j], dtype=torch.long, device=ctx.dev)
    shape = (len(envs), len(j))
    p0 = ctx.default_qpos[qa].expand(shape)
    v0 = torch.zeros(shape, device=ctx.dev)
    sp = _sample(position_range[0], position_range[1], shape, ctx.gen, ctx.dev)
    sv = _sample(velocity_range[0], velocity_range[1], shape, ctx.gen, ctx.dev)
    pos, vel = (p0 * sp, v0 * sv) if scale else (p0 + sp, v0 + sv)
    lo, hi = _joint_limits(ctx, j, envs)
    pos = torch.maximum(torch.minimum(pos, hi), lo)
    ctx.sim.t.qpos[envs[:, None], qa[None, :]] = pos
    ctx.sim.t.qvel[envs[:, None], da[None, :]] = vel


def reset_joints_by_scale(ctx: EventContext, env_ids, position_range, velocity_range, asset_cfg: AssetCfg | None = None):
    """Isaac: default joint pos/vel times a uniform scale, clamped to the soft position limits."""
    _reset_joints(ctx, env_ids, asset_cfg, position_range, velocity_range, scale=True)


def reset_joints_by_offset(ctx: EventContext, env_ids, position_range, velocity_range, asset_cfg: AssetCfg | None = None):
    """Isaac: default joint pos/vel plus a uniform offset, clamped to the soft position limits."""
    _reset_joints(ctx, env_ids, asset_cfg, position_range, velocity_range, scale=False)


@wp.kernel
def _rotate_wrench(xmat: wp.array2d(dtype=wp.mat33), wrench: wp.array3d(dtype=float), xfrc: wp.array2d(dtype=wp.spatial_vector)):
    e, b = wp.tid()
    R = xmat[e, b]
    f = R * wp.vec3(wrench[e, b, 0], wrench[e, b, 1], wrench[e, b, 2])
    t = R * wp.vec3(wrench[e, b, 3], wrench[e, b, 4], wrench[e, b, 5])
    xfrc[e, b] = wp.spatial_vector(f[0], f[1], f[2], t[0], t[1], t[2])


def apply_external_force_torque(ctx: EventContext, env_ids, force_range, torque_range, asset_cfg: AssetCfg | None = None):
    """Isaac: per (env, body) uniform force and torque in the body frame, applied every physics step
    until the next time the term runs. The first call installs a substep hook that rotates the stored wrench into ``xfrc_applied``."""
    envs = ctx.env_index(env_ids)
    b = torch.as_tensor(ctx.bodies(asset_cfg), dtype=torch.long, device=ctx.dev)
    if ctx.ext_wrench is None:
        from metalsim.interop import torch_bridge as tb
        ctx._wrench_wp = wp.zeros((ctx.n, ctx.m.nbody, 6), dtype=float, device=ctx.sim.device)
        wp.synchronize_device(ctx.sim.device)     # the zero-fill runs on Warp's queue: finish it before torch writes
        ctx.ext_wrench = tb.mps_tensor(ctx._wrench_wp)
    shape = (len(envs), len(b), 3)
    ctx.ext_wrench[envs] = 0.0
    ctx.ext_wrench[envs[:, None], b[None, :], 0:3] = _sample(force_range[0], force_range[1], shape, ctx.gen, ctx.dev)
    ctx.ext_wrench[envs[:, None], b[None, :], 3:6] = _sample(torque_range[0], torque_range[1], shape, ctx.gen, ctx.dev)
    if not ctx._hooked:
        sim = ctx.sim

        def hook():
            wp.launch(_rotate_wrench, dim=(ctx.n, ctx.m.nbody), inputs=[sim.d.xmat, ctx._wrench_wp, sim.d.xfrc_applied])
        sim.add_substep_hook(hook)
        ctx._hooked = True
    # the wrench for the current pose (the hook updates it after every substep)
    R = ctx.sim.t.xmat[envs].reshape(len(envs), -1, 3, 3)
    w = ctx.ext_wrench[envs]
    ctx.sim.t.xfrc_applied[envs, :, 0:3] = torch.einsum("nbij,nbj->nbi", R, w[..., 0:3])
    ctx.sim.t.xfrc_applied[envs, :, 3:6] = torch.einsum("nbij,nbj->nbi", R, w[..., 3:6])


def reset_scene_to_default(ctx: EventContext, env_ids):
    envs = ctx.env_index(env_ids)
    mask = torch.zeros(ctx.n, dtype=torch.bool, device=ctx.dev)
    mask[envs] = True
    torch.mps.synchronize()
    ctx.sim.reset(mask)
    ctx.sim.synchronize()


# -- visual terms (renderer tensors) -------------------------------------------------------------------

def randomize_visual_color(ctx_or_renderer, env_ids, colors, slots=None, gen: torch.Generator | None = None):
    """Isaac Lab ``randomize_visual_color``: per env one colour for the selected meshes, sampled from
    ``{"r": (lo, hi), "g": ..., "b": ...}`` ranges or chosen from a list of RGB triples. Writes the
    renderer's per-env per-slot colour table (``t_colors``)."""
    r = ctx_or_renderer
    dev = r.t_colors.device
    gen = gen or torch.Generator(device="mps").manual_seed(0)
    envs = torch.arange(r.n, device=dev) if env_ids is None else torch.as_tensor(env_ids, device=dev).long()
    if isinstance(colors, dict):
        lo = [colors[k][0] for k in "rgb"]; hi = [colors[k][1] for k in "rgb"]
        c = _sample(lo, hi, (len(envs), 3), gen, dev)
    else:
        table = torch.as_tensor(colors, dtype=torch.float32, device=dev)
        c = table[torch.randint(0, len(table), (len(envs),), generator=gen, device=dev)]
    s = torch.arange(r.G, device=dev) if slots is None else torch.as_tensor(list(slots), device=dev)
    r.t_colors[envs[:, None], s[None, :], 0:3] = c[:, None, :]


# -- manager --------------------------------------------------------------------------------------------

@dataclass
class EventTerm:
    """``EventTermCfg``: func, mode ("startup" | "reset" | "interval"), params, interval_range_s,
    is_global_time, min_step_count_between_reset."""
    func: object
    mode: str
    params: dict = field(default_factory=dict)
    interval_range_s: tuple | None = None
    is_global_time: bool = False
    min_step_count_between_reset: int = 0


class EventManager:
    """Isaac Lab's ``EventManager`` semantics for ``startup``, ``reset`` and ``interval`` terms."""

    def __init__(self, ctx: EventContext, terms: dict, step_dt: float):
        self.ctx, self.terms, self.dt = ctx, terms, step_dt
        self.time_left, self.last_reset = {}, {}
        for name, t in terms.items():
            if t.mode == "interval":
                lo, hi = t.interval_range_s
                shape = (1,) if t.is_global_time else (ctx.n,)
                self.time_left[name] = _sample(lo, hi, shape, ctx.gen, ctx.dev)
            if t.mode == "reset":
                self.last_reset[name] = torch.zeros(ctx.n, dtype=torch.long, device=ctx.dev)
        self.step_count = 0

    def apply(self, mode: str, env_ids=None, global_step: int | None = None):
        """Returns True if anything wrote model fields that need ``recompute_constants``."""
        dirty = False
        for name, t in self.terms.items():
            if t.mode != mode:
                continue
            if mode == "interval":
                tl = self.time_left[name]
                tl -= self.dt
                fire = tl <= 1e-6
                if not bool(fire.any()):
                    continue
                lo, hi = t.interval_range_s
                tl[fire] = _sample(lo, hi, (int(fire.sum()),), self.ctx.gen, self.ctx.dev)
                ids = None if t.is_global_time else torch.nonzero(fire).flatten()
                t.func(self.ctx, ids, **t.params)
                continue
            ids = env_ids
            if mode == "reset" and t.min_step_count_between_reset > 0 and global_step is not None:
                envs = self.ctx.env_index(env_ids)
                ok = (global_step - self.last_reset[name][envs]) >= t.min_step_count_between_reset
                ok |= self.last_reset[name][envs] == 0
                ids = envs[ok]
                self.last_reset[name][ids] = global_step
                if len(ids) == 0:
                    continue
            t.func(self.ctx, ids, **t.params)
            dirty |= t.func in (randomize_rigid_body_mass, randomize_rigid_body_com)
        return dirty
