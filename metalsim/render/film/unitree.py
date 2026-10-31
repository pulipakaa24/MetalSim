# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# MetalSim code, Apache-2.0 (see LICENSE). The observation, gravity projection, gait clock and PD torque are ported from
# unitree_rl_gym's deploy/deploy_mujoco/deploy_mujoco.py (commit 276801e4): Copyright (c) 2016-2023 HangZhou YuShu
# TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD-3-Clause (licenses/BSD-3-Clause-Unitree.txt). See THIRD_PARTY_NOTICES.md.
"""Unitree's own G1 walking policy (``unitree_rl_gym``, BSD-3-Clause) in the filming tool: its MuJoCo sim2sim
playback (``deploy/deploy_mujoco/deploy_mujoco.py`` with ``configs/g1.yaml``) run on MetalSim's MuJoCo Warp.

Unitree's loop, per 2 ms physics step (``deploy_mujoco.py``, the ``while viewer.is_running()`` body):

    tau = (target_q - q) * kp + (0 - dq) * kd ; d.ctrl = tau ; mj_step ; counter += 1
    every 10th step: obs from the state after the step -> policy -> target_q = action * 0.25 + default_angles

``UnitreeG1`` runs that loop on ``BatchSim`` (n worlds of Unitree's ``scene.xml``, timestep 0.002 s): the PD torque
is a Warp kernel launched before the first physics step of a policy step and, through ``BatchSim.add_substep_hook``,
after every physics step (inside the step graph), so each of the 10 steps of a 20 ms policy step is driven by the
torque of the state it starts from, as Unitree computes it from ``d.qpos`` / ``d.qvel`` before each ``mj_step``. The
model's actuators are Unitree's ``<motor>``s (no ctrlrange; the joints' ``actuatorfrcrange`` clamps the torque, in
MuJoCo and MuJoCo Warp alike). The observation (47: angular velocity x 0.25, projected gravity, command x (2, 2, 0.25),
joint position minus default, joint velocity x 0.05, last action, sin / cos of the 0.8 s gait clock) is built on the
host in float64 and cast to float32 exactly as ``deploy_mujoco.py`` builds it, with the clock taken from the physics
step counter (``counter * simulation_dt``). The policy is Unitree's TorchScript file (``deploy/pre_train/g1/motion.pt``,
an LSTM 47 -> 64 with an ELU MLP 64 -> 32 -> 12) loaded with ``torch.jit.load``: its recurrent state lives in the
module (buffers of batch 1), so each env gets its own loaded copy.

Gait data (``record.gait_metrics``): the G1's feet in Unitree's model collide through four 5 mm spheres under each
ankle-roll link (``UNITREE_FOOT_BODIES``); a foot's point is the mean of its four sphere centres (the sole, 5 mm above
the floor in stance), its footprint their xy, and it is in contact when any of its spheres has a contact with the
floor (``unitree_contacts``). The CPU reference run (``runs/film/g1_unitree/sim2sim_check/reference.py``) measures
with the same functions.
"""
from __future__ import annotations

import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

UNITREE_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
UNITREE_FLOOR = "floor"
FALLEN_Z = 0.4      # m: a pelvis below this is down (Unitree's training terminates on pelvis contact)


def unitree_paths(src: dict) -> dict:
    """Absolute paths of the fetched repository's files (``tasks.UNITREE_RL_GYM``)."""
    root = os.path.join(ROOT, src["root"])
    return dict(root=root, policy=os.path.join(root, src["policy"]), config=os.path.join(root, src["config"]),
                scene=os.path.join(root, src["scene"]))


def load_unitree_config(path: str) -> dict:
    """``deploy_mujoco.py``'s reading of its YAML config (the same keys and dtypes)."""
    import yaml

    with open(path) as f:
        c = yaml.load(f, Loader=yaml.FullLoader)
    out = dict(c)
    for k in ("kps", "kds", "default_angles", "cmd_scale", "cmd_init"):
        out[k] = np.array(c[k], dtype=np.float32)
    return out


def load_unitree_model(scene: str, cfg: dict):
    """Unitree's scene with ``deploy_mujoco.py``'s one change: ``m.opt.timestep = simulation_dt``."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(scene)
    m.opt.timestep = cfg["simulation_dt"]
    return m


def unitree_foot_geoms(m) -> list:
    """Per foot (left, right): the ids of the colliding geoms of its ankle-roll link (Unitree's four sole spheres)."""
    import mujoco

    out = []
    for b in UNITREE_FOOT_BODIES:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        out.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and (m.geom_contype[g] or m.geom_conaffinity[g])])
    return out


def unitree_feet(m, d_fk, qpos, foot_geoms):
    """For each row of ``qpos`` (N, nq), by forward kinematics on ``d_fk``: (N, 2, 3) foot points (mean of the sole
    spheres' centres), (N, 2, K, 2) footprints (the spheres' xy) and (N, 2, 3) ankle-roll link origins (the point
    Unitree's swing-height reward reads: ``feet_pos`` = rigid-body position of the ``ankle_roll`` links)."""
    import mujoco

    qpos = np.atleast_2d(qpos)
    K = len(foot_geoms[0])
    feet, fp, ank = np.zeros((len(qpos), 2, 3)), np.zeros((len(qpos), 2, K, 2)), np.zeros((len(qpos), 2, 3))
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in UNITREE_FOOT_BODIES]
    for e in range(len(qpos)):
        d_fk.qpos[:] = qpos[e]
        mujoco.mj_kinematics(m, d_fk)
        for f, gs in enumerate(foot_geoms):
            P = d_fk.geom_xpos[gs]
            feet[e, f] = P.mean(0)
            fp[e, f] = P[:, :2]
            ank[e, f] = d_fk.xpos[bids[f]]
    return feet, fp, ank


def unitree_contacts(pairs, worlds, n: int, foot_geoms, floor: int) -> np.ndarray:
    """(n, 2) bool: env ``w``'s foot ``f`` touches the floor (a contact between the floor and one of its spheres), from
    contact geom pairs (K, 2) and their world ids (K,): MuJoCo's ``d.contact.geom`` (one world) or MuJoCo Warp's
    ``d.contact.geom`` / ``d.contact.worldid`` up to ``d.nacon``."""
    out = np.zeros((n, 2), bool)
    pairs, worlds = np.asarray(pairs).reshape(-1, 2), np.asarray(worlds).reshape(-1)
    for f, gs in enumerate(foot_geoms):
        hit = ((pairs[:, 0] == floor) & np.isin(pairs[:, 1], gs)) | ((pairs[:, 1] == floor) & np.isin(pairs[:, 0], gs))
        out[worlds[hit], f] = True
    return out


def gravity_orientation(quat):
    """``deploy_mujoco.get_gravity_orientation`` for (N, 4) wxyz quaternions."""
    qw, qx, qy, qz = (quat[:, i] for i in range(4))
    return np.stack([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)], 1)


def unitree_obs(cfg: dict, qpos, qvel, action, cmd, counter: int) -> np.ndarray:
    """(N, 47) float32: ``deploy_mujoco.py``'s observation for each env (float64 arithmetic, then the float32 store),
    with the gait clock at physics step ``counter``."""
    n, na = len(qpos), int(cfg["num_actions"])
    qj = (qpos[:, 7:] - cfg["default_angles"]) * cfg["dof_pos_scale"]
    dqj = qvel[:, 6:] * cfg["dof_vel_scale"]
    grav = gravity_orientation(qpos[:, 3:7])
    omega = qvel[:, 3:6] * cfg["ang_vel_scale"]
    period = 0.8
    count = counter * cfg["simulation_dt"]
    phase = count % period / period
    obs = np.zeros((n, int(cfg["num_obs"])), np.float32)
    obs[:, :3] = omega
    obs[:, 3:6] = grav
    obs[:, 6:9] = cmd * cfg["cmd_scale"]
    obs[:, 9:9 + na] = qj
    obs[:, 9 + na:9 + 2 * na] = dqj
    obs[:, 9 + 2 * na:9 + 3 * na] = action
    obs[:, 9 + 3 * na:9 + 3 * na + 2] = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
    return obs


_PD = None


def _pd_kernel():
    """Unitree's ``pd_control`` per env and actuator: ctrl = (target - q) kp + (0 - dq) kd (actuator i drives joint i,
    qpos 7 + i, dof 6 + i in Unitree's 12-dof model)."""
    global _PD
    if _PD is None:
        import warp as wp

        @wp.kernel(module="unique")
        def unitree_pd(target: wp.array2d(dtype=float), qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                       kp: wp.array(dtype=float), kd: wp.array(dtype=float), ctrl: wp.array2d(dtype=float)):
            w, i = wp.tid()
            ctrl[w, i] = (target[w, i] - qpos[w, 7 + i]) * kp[i] + (0.0 - qvel[w, 6 + i]) * kd[i]

        _PD = unitree_pd
    return _PD


class UnitreeG1:
    """Unitree's G1 sim2sim walk (``deploy_mujoco.py`` + ``configs/g1.yaml`` + ``deploy/pre_train/g1/motion.pt``) on
    MuJoCo Warp, ``num_envs`` copies of the same start (Unitree's: the model's ``qpos0``, pelvis at 0.793 m, joints at
    0, target the default angles until the first policy step at 20 ms). ``adapter_kw``: ``cmd`` (default: the config's
    ``cmd_init``). One frame per policy step (50 fps), the state after its 10 physics steps (the state the next
    observation reads). No episode ends (Unitree's playback has no termination or reset); an env whose pelvis drops
    below ``FALLEN_Z`` is reported ``terminated`` (down) and its first episode ends there."""

    backend = "mujoco"

    def __init__(self, task, film, checkpoint, viewer=None):
        import mujoco
        import torch
        import warp as wp

        from metalsim.physics.batch import BatchSim, BatchSimOptions
        from metalsim.render.film import tasks

        wp.config.quiet = True
        self.wp, self.torch, self.mujoco = wp, torch, mujoco
        akw = dict(getattr(task, "adapter_kw", None) or {})
        self.src = tasks.UNITREE_RL_GYM
        self.paths = unitree_paths(self.src)
        if not os.path.exists(self.paths["scene"]):
            raise FileNotFoundError(f"{self.src['root']} missing: run {self.src['fetch']}")
        self.cfg = cfg = load_unitree_config(self.paths["config"])
        self.checkpoint = checkpoint or self.paths["policy"]
        n = self.num_envs = int(film.num_envs)
        self.cmd = np.array(akw.get("cmd", cfg["cmd_init"]), dtype=np.float32)
        self.dec = int(cfg["control_decimation"])
        t0 = time.perf_counter()
        self.model = m = load_unitree_model(self.paths["scene"], cfg)
        self.sim = BatchSim(m, n, options=BatchSimOptions(substeps=self.dec))
        self.frame_dt = float(cfg["simulation_dt"]) * self.dec
        dev = self.sim.device
        na = int(cfg["num_actions"])
        if m.nu != na or m.nq != 7 + na:
            raise ValueError(f"model nu {m.nu} / nq {m.nq} do not fit {na} actions")
        self.target = wp.array(np.tile(cfg["default_angles"], (n, 1)), dtype=float, device=dev)
        self.kp = wp.array(cfg["kps"], dtype=float, device=dev)
        self.kd = wp.array(cfg["kds"], dtype=float, device=dev)
        self._pd = _pd_kernel()
        self.sim.add_substep_hook(self._launch_pd)
        self.build_s = time.perf_counter() - t0
        self.foot_geoms = unitree_foot_geoms(m)
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, UNITREE_FLOOR)
        self._fk = mujoco.MjData(m)
        self.ckpt_format = "torchscript"
        self.ckpt_iter = None
        self.ckpt_source = dict(source="Unitree unitree_rl_gym pretrained G1 policy", source_url=self.src["policy_url"],
                                published_sha256=self.src["policy_sha256"], commit=self.src["commit"],
                                licence=self.src["licence"], fetch=self.src["fetch"])
        self.overflow = None

    def _launch_pd(self):
        d = self.sim.d
        self.wp.launch(self._pd, dim=(self.num_envs, self.model.nu), device=self.sim.device,
                       inputs=[self.target, d.qpos, d.qvel, self.kp, self.kd], outputs=[d.ctrl])

    def _policy(self, obs):
        torch = self.torch
        return np.stack([p(torch.from_numpy(obs[e:e + 1])).detach().numpy().squeeze()
                         for e, p in enumerate(self.policies)]).astype(np.float32)

    def reset(self):
        n, cfg = self.num_envs, self.cfg
        self.sim.reset()
        self.sim.synchronize()
        self.policies = [self.torch.jit.load(self.checkpoint) for _ in range(n)]   # fresh LSTM state per env
        self.action = np.zeros((n, int(cfg["num_actions"])), np.float32)
        self.target.assign(np.tile(cfg["default_angles"], (n, 1)))
        self.counter = 0
        q = self._qpos()
        self.start = q[:, :3].copy()
        self._xy = q[:, :2].copy()
        self.dist = np.zeros(n)
        self.descent = np.zeros(n)
        self._alive = np.ones(n, bool)

    def _qpos(self):
        return self.sim.d.qpos.numpy().astype(np.float64).reshape(self.num_envs, -1)

    def step(self):
        wp, sim, cfg = self.wp, self.sim, self.cfg
        t0 = time.perf_counter()
        with wp.ScopedDevice(sim.device):
            self._launch_pd()               # the torque of the first physics step: current target, current state
            sim.step()                      # 10 physics steps, the PD torque relaunched after each (substep hook)
            sim.synchronize()
        self.counter += self.dec
        qpos = self._qpos()
        qvel = sim.d.qvel.numpy().astype(np.float64).reshape(self.num_envs, -1)
        nac = int(sim.d.nacon.numpy()[0])
        pairs = sim.d.contact.geom.numpy()[:nac]
        worlds = sim.d.contact.worldid.numpy()[:nac]
        obs = unitree_obs(cfg, qpos, qvel, self.action, self.cmd, self.counter)
        self.action = self._policy(obs)
        self.target.assign((self.action * cfg["action_scale"] + cfg["default_angles"]).astype(np.float32))
        sim_s = time.perf_counter() - t0
        n = self.num_envs
        down = qpos[:, 2] < FALLEN_Z
        al = self._alive
        self.dist[al] += np.linalg.norm(qpos[al, :2] - self._xy[al], axis=1)
        dz = qpos[:, 2] - self.start[:, 2]
        self.descent[al] = np.minimum(self.descent[al], dz[al])
        self._xy = qpos[:, :2].copy()
        self._alive &= ~down
        feet, footprint, _ = unitree_feet(self.model, self._fk, qpos, self.foot_geoms)
        touch = unitree_contacts(pairs, worlds, n, self.foot_geoms, self.floor)
        snap = dict(qpos=qpos, feet=feet, feet_footprint=footprint, feet_contact=touch,
                    cmd=np.tile(self.cmd.astype(np.float64), (n, 1)))
        info = dict(terminated=down.copy(), base_z=qpos[:, 2].copy())
        return snap, down, info, dict(sim_s=sim_s)

    def summary(self, ends) -> str:
        self.overflow = self.sim.check_overflow()
        rows = [f"{e}:{self.dist[e]:.2f}/{self.descent[e]:+.2f}" + (f"@{int(ends[e])}" if ends[e] >= 0 else "")
                for e in range(self.num_envs)]
        return "[record] first episodes, env:path m/drop m[@end frame]: " + " ".join(rows)

    def rank(self, first, gait=None):
        """Envs not down first, then the natural-gait preference (``adapters.natural_gait``), then the longest path.
        (Every env starts from the same state; they differ only by the GPU rollout's float noise.)"""
        from metalsim.render.film.adapters import natural_gait

        n = self.num_envs
        term = np.asarray(first.get("terminated", np.zeros(n)), bool)
        self.natural = natural_gait(gait, [e for e in range(n) if not term[e]], n)
        return [k[-1] for k in sorted((term[e], not self.natural[e], -self.dist[e], e) for e in range(n))]

    def twin_spec(self):
        """Unitree's scene (``scene.xml`` with ``g1_12dof.xml``) and its STL meshes: the simulated model itself, with
        the mesh directory made absolute so the exported twin finds the meshes."""
        spec = self.mujoco.MjSpec.from_file(self.paths["scene"])
        spec.meshdir = os.path.join(os.path.dirname(self.paths["scene"]), spec.meshdir or "")
        return spec

    def meta(self):
        from metalsim.render.film import provenance

        m, cfg = self.model, self.cfg
        keys = ("simulation_dt", "control_decimation", "kps", "kds", "default_angles", "ang_vel_scale", "dof_pos_scale",
                "dof_vel_scale", "action_scale", "cmd_scale", "num_actions", "num_obs", "cmd_init")
        out = dict(policy=provenance.relpath(self.checkpoint), policy_format=self.ckpt_format,
                   policy_source=self.ckpt_source, policy_arch="TorchScript: LSTM 47 -> 64, ELU MLP 64 -> 32 -> 12; one "
                   "loaded copy per env (recurrent state in the module)",
                   unitree_rl_gym=dict(repo=self.src["repo"], commit=self.src["commit"], licence=self.src["licence"],
                                       config=self.src["config"], scene=self.src["scene"], policy=self.src["policy"]),
                   unitree_config={k: (cfg[k].tolist() if hasattr(cfg[k], "tolist") else cfg[k]) for k in keys},
                   cmd=self.cmd.tolist(), env_build_s=self.build_s,
                   physics=("MuJoCo Warp BatchSim (metalsim.physics.batch), Unitree's scene.xml; Unitree's PD torque "
                            "every physics step (Warp kernel: before the first step of a policy step, then "
                            "BatchSim.add_substep_hook after each)"),
                   model=dict(timestep=float(m.opt.timestep), integrator=int(m.opt.integrator), cone=int(m.opt.cone),
                              solver=int(m.opt.solver), iterations=int(m.opt.iterations),
                              ls_iterations=int(m.opt.ls_iterations), nq=int(m.nq), nv=int(m.nv), nu=int(m.nu)),
                   foot_geoms=dict(bodies=list(UNITREE_FOOT_BODIES), geoms=[list(map(int, g)) for g in self.foot_geoms],
                                   point="mean of the four sole-sphere centres", contact="any sphere-floor contact"),
                   capacity_overflow=self.overflow, actions="Unitree's policy output (deterministic)",
                   done=f"none (Unitree's playback has no termination); pelvis below {FALLEN_Z} m reported as down")
        if hasattr(self, "dist"):
            out["first_episode"] = dict(start=self.start.round(4).tolist(), path_m=self.dist.round(3).tolist(),
                                        descent_m=self.descent.round(3).tolist())
        return out
