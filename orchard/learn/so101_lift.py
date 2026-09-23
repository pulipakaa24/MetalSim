"""SO-101 lift task, fully on the GPU: MuJoCo Warp physics, tier-0 rendering, torch (MPS) reward,
termination, reset and domain randomization. The Phase 2 acceptance task.

Replicates mjbatch-metal's ``BatchPixelVecEnv`` (examples/so101_lift_vecenv.py): same reward
(reach + lift + hold bonus), termination (box out of bounds or time limit), success (box held
above LIFT_Z for SUCCESS_HOLD steps), visual DR (per-env link colors, camera jitter, light
direction, background) and reset distribution (box radius 0.15–0.26 m, random yaw, arm joint
noise). Physics DR (box mass / friction) is not yet applied per world (MuJoCo Warp model fields
are shared across worlds unless expanded); logged as a gap.

Per step, the only host activity is Python driving submissions; no tensor leaves the GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from orchard.interop import torch_bridge as tb
from orchard.interop import warp_metal as wm
from orchard.physics.batch import BatchSim, BatchSimOptions
from orchard.render.tier0 import Tier0Renderer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.environ.get("SCENE_XML", os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml"))
_TCP_LOCAL = torch.tensor([-0.00739, -0.01061, 0.01278])
LIFT_Z = 0.10
SUCCESS_HOLD = 10
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@dataclass
class LiftConfig:
    num_envs: int = 64
    ctrl_dt: float = 0.02
    max_episode_steps: int = 300
    tile: int = 128
    decimate_faces: int = 3000
    seed: int = 0
    reward_mode: str = "vanilla"     # or "grasped" (grasp-conditioned lift bonus; needs contact sensing)
    render: bool = True
    backgrounds: bool = True
    njmax: int = 512                 # constraint rows per world; see BatchSimOptions.njmax
    nconmax: int | None = None
    warn_overflow: bool = False
    physics_dr: bool = True          # per-world box mass (0.7-1.4x) and friction (0.7-1.3x), as the baseline


class SO101LiftEnv:
    """Batched env with a Gym-like vectorized API over MPS tensors.

    ``obs`` is a dict: ``image`` (N,H,W,3) uint8, ``qpos`` (N,6) float32, both zero-copy views of
    simulator memory (valid until the next ``step``/``reset``; clone if kept).
    """

    def __init__(self, cfg: LiftConfig | None = None):
        self.cfg = cfg or LiftConfig()
        self.n = self.cfg.num_envs
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        m = self.model
        substeps = max(1, int(round(self.cfg.ctrl_dt / m.opt.timestep)))
        self.sim = BatchSim(m, self.n, options=BatchSimOptions(
            substeps=substeps, njmax=self.cfg.njmax, nconmax=self.cfg.nconmax, warn_overflow=self.cfg.warn_overflow,
            per_world_fields=("body_mass", "geom_friction") if self.cfg.physics_dr else ()))
        self.rend = None
        if self.cfg.render:
            self.rend = Tier0Renderer(m, self.n, width=self.cfg.tile, height=self.cfg.tile, camera="base_cam",
                                      include_planes=not self.cfg.backgrounds, backgrounds=self.cfg.backgrounds,
                                      outputs=("rgb",), decimate_faces=self.cfg.decimate_faces)
        self.dev = torch.device("mps")
        self.gen = torch.Generator(device="mps").manual_seed(self.cfg.seed)
        self.learner_event = wm.SharedEvent("metal:0", "orchard.learner")

        rngs = m.actuator_ctrlrange.copy()
        limited = m.actuator_ctrllimited.astype(bool).reshape(-1)
        self.ctrl_lo = torch.as_tensor(np.where(limited, rngs[:, 0], -np.pi), dtype=torch.float32, device=self.dev)
        self.ctrl_hi = torch.as_tensor(np.where(limited, rngs[:, 1], np.pi), dtype=torch.float32, device=self.dev)
        self.grip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box")
        self.box_qadr = int(m.jnt_qposadr[box_jnt])
        self.arm_qadr = torch.as_tensor([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
                                         for j in ARM_JOINTS], device=self.dev)
        self.home = torch.as_tensor(m.qpos0, dtype=torch.float32, device=self.dev)
        self.tcp_local = _TCP_LOCAL.to(self.dev)
        if self.rend is not None:
            self.box_slot = self.rend.tables.geoms.index(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box"))
        self.box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.mass0 = float(m.body_mass[self.box_body])
        self.fric0 = torch.as_tensor(m.geom_friction[self.box_geom], dtype=torch.float32, device=self.dev)
        # episode state on the GPU
        z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=self.dev)
        self.t = z(self.n, dtype=torch.int32)
        self.held = z(self.n, dtype=torch.int32)
        self.succ = z(self.n, dtype=torch.bool)
        self.ep_rew = z(self.n)
        self.ep_len = z(self.n, dtype=torch.int32)
        self._last_done = z(self.n, dtype=torch.bool)
        self.obs_space = {"image": (self.cfg.tile, self.cfg.tile, 3), "qpos": (6,)}
        self.act_dim = m.nu
        self.sim.synchronize()
        if self.rend is not None and self.cfg.backgrounds:
            self._init_backgrounds()

    # -- domain randomization --------------------------------------------------------------------

    def _init_backgrounds(self):
        rng = np.random.default_rng(self.cfg.seed)
        T = self.cfg.tile
        imgs = []
        for e in range(self.rend.bg_layers):
            kind = rng.integers(0, 3)
            if kind == 0:
                im = np.full((T, T, 3), rng.integers(0, 256, 3), np.uint8)
            elif kind == 1:
                a = rng.integers(0, 256, 3).astype(np.float32); b = rng.integers(0, 256, 3).astype(np.float32)
                t = np.linspace(0, 1, T)[:, None, None]
                im = np.broadcast_to((a * (1 - t) + b * t).astype(np.uint8), (T, T, 3)).copy()
            else:
                small = rng.integers(0, 256, (8, 8, 3)).astype(np.uint8)
                idx = np.linspace(0, 7, T).astype(int)
                im = small[np.ix_(idx, idx)]
            imgs.append(im)
        self.rend.set_backgrounds(range(self.rend.bg_layers), imgs)

    def _randomize(self, mask: torch.Tensor):
        """GPU-side reset randomization for the worlds in ``mask`` (after sim.reset, before forward)."""
        n = self.n
        q = self.sim.t.qpos
        u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.dev)
        r = u(0.15, 0.26, n); th = u(-0.7, 0.7, n); yaw = u(-np.pi, np.pi, n)
        box = torch.stack([r * torch.cos(th), r * torch.sin(th), torch.full((n,), 0.03, device=self.dev)], 1)
        quat = torch.stack([torch.cos(yaw / 2), torch.zeros(n, device=self.dev), torch.zeros(n, device=self.dev), torch.sin(yaw / 2)], 1)
        arm = self.home[self.arm_qadr[:5]] + u(-0.08, 0.08, n, 5)
        mf = mask.unsqueeze(1).float()
        a = self.box_qadr
        q[:, a:a + 3] = mf * box + (1 - mf) * q[:, a:a + 3]
        q[:, a + 3:a + 7] = mf * quat + (1 - mf) * q[:, a + 3:a + 7]
        q[:, self.arm_qadr[:5]] = mf * arm + (1 - mf) * q[:, self.arm_qadr[:5]]
        if self.cfg.physics_dr:   # per-world model fields (physics DR), then mass-derived constants
            bm = self.sim.tm.body_mass; gf = self.sim.tm.geom_friction
            bm[:, self.box_body] = mask.float() * self.mass0 * u(0.7, 1.4, n) + (1 - mask.float()) * bm[:, self.box_body]
            gf[:, self.box_geom] = mf * self.fric0 * u(0.7, 1.3, n, 1) + (1 - mf) * gf[:, self.box_geom]
        if self.rend is not None:
            c = self.rend.t_colors
            base = u(0.1, 0.95, n, 1, 3)
            cols = torch.clamp(base + u(-0.12, 0.12, n, self.rend.G, 3), 0, 1)
            cols[:, self.box_slot] = torch.clamp(torch.tensor([0.1, 0.8, 0.15], device=self.dev) + u(-0.1, 0.1, n, 3), 0, 1)
            c[:, :, :3] = mask.view(n, 1, 1).float() * cols + (1 - mask.view(n, 1, 1).float()) * c[:, :, :3]
            cs = self.rend.t_cam_spec
            pos_delta = u(-0.015, 0.015, n, 3)
            rot = torch.cat([torch.ones(n, 1, device=self.dev), u(-0.01, 0.01, n, 3)], 1)
            rot = rot / rot.norm(dim=1, keepdim=True)
            light = torch.tensor([0.3, 0.3, -0.9], device=self.dev) + u(-0.15, 0.15, n, 3)
            cs[:, 4:7] = mf * pos_delta + (1 - mf) * cs[:, 4:7]
            cs[:, 8:12] = mf * rot + (1 - mf) * cs[:, 8:12]
            cs[:, 12:15] = mf * light + (1 - mf) * cs[:, 12:15]
            cs[:, 15] = 1.0                       # the DR light is on (adds to the model lights, casts the shadow)
        # bookkeeping
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.held = torch.where(mask, torch.zeros_like(self.held), self.held)
        self.succ = torch.where(mask, torch.zeros_like(self.succ), self.succ)
        self.ep_rew = torch.where(mask, torch.zeros_like(self.ep_rew), self.ep_rew)
        self.ep_len = torch.where(mask, torch.zeros_like(self.ep_len), self.ep_len)

    # -- ordering helpers -----------------------------------------------------------------------

    def _learner_done(self) -> None:
        """Commit torch's writes to sim/renderer memory and order the sim after them."""
        v = self.learner_event.next_value()
        tb.signal_event(self.learner_event, v)
        self.sim.wait(self.learner_event, v)
        if self.rend is not None:
            self.rend.ctx_wait = (self.learner_event, v)

    def _render(self, after_value: int):
        if self.rend is None:
            return None
        # renderer waits for the sim (poses) and for torch's DR writes
        cb_wait = getattr(self.rend, "ctx_wait", None)
        vr = self.rend.render(self.sim, after_value)
        self.rend.wait_sim_after_render(self.sim, vr)
        self.rend.after(vr)
        return vr

    def _obs(self):
        img = self.rend.out.rgb if self.rend is not None else None
        return {"image": img, "qpos": self.sim.t.qpos[:, self.arm_qadr]}

    # -- API ------------------------------------------------------------------------------------------

    def reset(self):
        mask = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        v = self.sim.reset(mask)
        self.sim.after(v)
        self._randomize(mask)
        self._learner_done()
        if self.cfg.physics_dr:
            self.sim.recompute_constants()
        v = self.sim.forward()
        self._render(v)
        self.sim.after(v)
        return self._obs()

    def step(self, actions: torch.Tensor):
        """actions: (N, nu) in [-1, 1] on MPS. Returns obs, reward, done, info (all MPS tensors)."""
        a = torch.clamp(actions, -1.0, 1.0)
        self.sim.t.ctrl[:] = self.ctrl_lo + 0.5 * (a + 1.0) * (self.ctrl_hi - self.ctrl_lo)
        self._learner_done()
        vs = self.sim.step()
        self.sim.after(vs)
        # reward and termination on MPS from aliased state
        box = self.sim.t.xpos[:, self.box_body]                                   # (N,3)
        site = self.sim.t.site_xpos[:, self.grip_site]
        smat = self.sim.t.site_xmat[:, self.grip_site]                            # (N,3,3)
        grip = site + torch.einsum("nij,j->ni", smat, self.tcp_local)
        dist = torch.linalg.norm(box - grip, dim=1)
        height = box[:, 2]
        lifted = height > LIFT_Z
        r_reach = 1.0 - torch.tanh(10.0 * dist)
        r_lift = 5.0 * torch.clamp((height - 0.035) / (LIFT_Z - 0.035), 0.0, 1.0)
        reward = r_reach + r_lift + 3.0 * lifted.float()
        self.t += 1
        self.held = torch.where(lifted, self.held + 1, torch.zeros_like(self.held))
        self.succ |= self.held >= SUCCESS_HOLD
        oob = torch.linalg.norm(box[:, :2], dim=1) > 0.45
        trunc = self.t >= self.cfg.max_episode_steps
        done = oob | trunc
        self.ep_rew += reward
        self.ep_len += 1
        info = {"episode_reward": self.ep_rew.clone(), "episode_length": self.ep_len.clone(),
                "is_success": self.succ.clone(), "truncated": trunc & ~oob, "done": done}
        # auto-reset finished worlds, then render the post-reset observation (Isaac Lab convention)
        vres = self.sim.reset(done)
        self.sim.after(vres)
        self._randomize(done)
        self._learner_done()
        if self.cfg.physics_dr:
            self.sim.recompute_constants()
        vf = self.sim.forward()
        self._render(vf)
        self.sim.after(vf)
        self._last_done = done
        return self._obs(), reward, done, info

    def synchronize(self):
        self.sim.synchronize()
        if self.rend is not None:
            self.rend.ctx.synchronize()
        torch.mps.synchronize()
