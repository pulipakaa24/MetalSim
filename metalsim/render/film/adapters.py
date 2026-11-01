"""Recorder adapters: the only per-task code of the filming tool. An adapter runs a task's own environment with its
policy and hands the recorder one frame at a time; everything else (selection, render twin, rendering, encoding,
provenance, doc) is generic (``metalsim.render.film.record`` / ``render``).

Protocol (duck-typed):

* ``backend``: "newton" (Newton ``Model`` / ``State``) or "mujoco" (MuJoCo / MuJoCo Warp ``qpos``)
* ``frame_dt`` [s] (one frame per policy step: Isaac's ``render_fps`` = 1 / step_dt), ``num_envs``
* ``reset()``; ``step() -> (snap, done, info, timing)``: advance one frame. ``snap``: "newton": ``body_q`` (N, B, 7)
  and ``particles`` (N, P, 3) or None, both in each env's own frame (env origin subtracted); "mujoco": ``qpos``
  (N, nq). The state is captured before done envs are reset (Isaac Lab's ``ManagerBasedRLEnv.step`` order), so a
  finished episode's last frame is in the film. ``done`` (N,) bool; ``info``: per-env arrays (outcome at a done;
  ``info["terminated"]``, when an adapter reports it, marks a failure termination such as a fall, as opposed to a
  time-out or a success: the recorder refuses to film such an episode unless ``FilmCfg.allow_early_end``);
  ``timing``: {"sim_s": seconds of physics + policy for this frame}.
* ``rank(first) -> list``: env order for automatic selection, given each env's first-episode info.
* ``twin_newton()`` ("newton"): a Newton model with one world whose bodies are one env's bodies in the same order, for
  the render twin (visual shapes loaded); ``twin_spec()`` ("mujoco"): the ``MjSpec`` of a render model with the task
  model's ``qpos`` layout.
* ``meta()``: provenance of the rollout (policy, training run, physics settings).
"""
from __future__ import annotations

import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class FrankaPour:
    """Isaac Lab 3.0-EA ``IsaacContrib-Franka-Pour`` (metalsim.learn.franka_pour) with a trained rsl_rl actor, as
    ``franka_pour_eval`` runs it: uniform reset rows (Isaac's play mode freezes the curriculum), the actor's mean
    action."""

    backend = "newton"

    def __init__(self, task, film, checkpoint, viewer=None):
        import torch
        import warp as wp

        wp.config.quiet = True
        from metalsim.learn import franka_pour as P
        from metalsim.learn import franka_pour_ppo as PPO

        self.P, self.PPO, self.torch, self.wp = P, PPO, torch, wp
        n = self.num_envs = int(film.num_envs)
        t0 = time.perf_counter()
        pcfg = P.S.PourPhysicsConfig(num_envs=n)
        self.env = env = P.FrankaPourEnv(P.PourTaskConfig(num_envs=n, seed=film.seed, physics=pcfg,
                                                          curriculum_freeze=True))
        self.build_s = time.perf_counter() - t0
        self.checkpoint = checkpoint
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.ckpt_iter = ckpt.get("iter", ckpt.get("iteration"))
        self.actor = PPO.Actor(84, 8, PPO.PourPPOConfig()).to(env.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.eval()
        self.frame_dt = float(P.STEP_DT)
        self.info = env.info
        self.origins = np.asarray(env.info.origins, np.float64)
        self._cat = env._reset_dataset_states["category"]
        m = env.physics.model
        r = m.particle_radius.numpy()
        self.particle_radius = float(np.median(r))
        self.particles_per_env = int(env.info.particles_per_world)

    def reset(self):
        self.torch.manual_seed(self.env.cfg.seed)
        self.obs = self.env.reset_all()

    def step(self):
        torch, env = self.torch, self.env
        cat = self._cat[env.reset_dataset_row_id].cpu().numpy().astype(np.int64)
        t0 = time.perf_counter()
        with torch.inference_mode():
            mean, _ = self.actor(self.PPO.actor_input(self.obs))
        env.process_action(mean)
        env.simulate()
        self.wp.synchronize_device(env.physics.device)
        sim_s = time.perf_counter() - t0
        n = self.num_envs
        bq = env._bq.cpu().numpy().astype(np.float64).reshape(n, -1, 7)
        bq[:, :, :3] -= self.origins[:, None, :]
        pq = env._pq.cpu().numpy().astype(np.float64).reshape(n, -1, 3) - self.origins[:, None, :]
        # the pour state the termination terms see this step (before done envs reset), as franka_pour_eval reads it
        env.common_step_counter += 1
        env._invalidate_caches()
        src, tgt, spl = env._particle_region_masks()
        frac = [(m.sum(dim=1).float() / env._num_particles).cpu().numpy() for m in (src, tgt, spl)]
        length = (env.episode_length_buf + 1).cpu().numpy()
        env.common_step_counter -= 1
        t1 = time.perf_counter()
        self.obs, _, term, tout, _ = env.post_physics()
        sim_s += time.perf_counter() - t1
        done = (term | tout).cpu().numpy()
        info = dict(success=env._term_dones["success"].cpu().numpy(), time_out=env._term_dones["time_out"].cpu().numpy(),
                    category=cat, length=length, source_frac=frac[0], target_frac=frac[1], spill_frac=frac[2])
        return dict(body_q=bq.astype(np.float32), particles=pq.astype(np.float32)), done, info, dict(sim_s=sim_s)

    def rank(self, first):
        """Successful episodes first, those starting before the grasp (category 0: approach, grasp, lift, pour) ahead
        of those starting with the cup in hand; then by env id."""
        n = self.num_envs
        succ = np.asarray(first.get("success", np.zeros(n)), bool)
        cat = np.asarray(first.get("category", np.ones(n)), np.int64)
        key = [(0 if succ[e] else 1, 0 if cat[e] == 0 else 1, e) for e in range(n)]
        return [k[2] for k in sorted(key)]

    def twin_newton(self):
        """The task's scene builder with one world on the CPU, with the USD visual meshes loaded (the physics model
        loads colliders only, as Isaac's Newton import does); bodies in the same order as each simulated world."""
        S = self.P.S
        try:
            model, info = S.build_model(1, device="cpu", visuals=True)
            self.twin_visuals = True
        except Exception as ex:  # noqa: BLE001  (render the colliders rather than nothing; reported in rec.json)
            print(f"[film] visual-mesh import failed ({type(ex).__name__}: {ex}); the twin draws the colliders", flush=True)
            model, info = S.build_model(1, device="cpu", visuals=False)
            self.twin_visuals = f"colliders ({type(ex).__name__}: {ex})"
        sim_labels = [str(x).split("/World/envs/env_0/")[-1] for x in list(self.env.physics.model.body_label)[:self.info.bodies_per_world]]
        tw_labels = [str(x).split("/World/envs/env_0/")[-1] for x in list(model.body_label)]
        if sim_labels != tw_labels:
            raise RuntimeError(f"render-twin bodies differ from the simulated world's: {sim_labels} vs {tw_labels}")
        return model

    def meta(self):
        import json
        cfg_path = self.checkpoint.replace("_model_299.pt", "_config.json")
        run_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
        return dict(policy=os.path.relpath(self.checkpoint, ROOT) if os.path.isabs(self.checkpoint) else self.checkpoint,
                    policy_iteration=self.ckpt_iter, training_config=run_cfg, rows="uniform (play mode)",
                    actions="actor mean", env_build_s=self.build_s, twin_visuals=getattr(self, "twin_visuals", None), particles_per_env=self.particles_per_env,
                    particle_radius=self.particle_radius, physics="Newton 1.5.2 SolverCoupledProxy: MuJoCo Warp arm + "
                    "implicit MPM media (metalsim.learn.franka_pour_scene)")


# task settings ``metalsim.learn.g1_velocity.train_g1`` saves in a checkpoint (the rest, e.g. physics_dt, are in its log header)
G1_SAVED_TASK_KEYS = ("reward_cfg", "scan_ordering", "terrain_collision", "scan_surface", "solver_cfg", "il3_events",
                      "contact_cfg", "base_velocity", "feet_slide_velocity")

# Isaac Lab 2.3's G1 joint order: PhysX's articulation order (breadth-first over g1_minimal.usd's kinematic tree) as
# Isaac Sim 5.1 / Isaac Lab 2.3.2 reports it (``robot.joint_names``, written by
# metalsim/parity/isaac_side/play_policy.py into runs/parity/isaac/rough/play_rough/L*/isaac_it*/traj.npz on the L4). An rsl_rl checkpoint trained in Isaac reads
# its joint blocks (pos, vel, last action) and writes its actions in this order. MetalSim's G1 model is depth-first
# (left leg, right leg, torso, arms, hands), so the adapter maps the blocks by joint name, as
# metalsim/parity/side_by_side.py and scripts/diagnostics/g1_rough_transfer.py do.
ISAAC_G1_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint", "left_four_joint", "left_one_joint",
    "right_six_joint", "right_four_joint", "right_one_joint", "left_two_joint", "right_two_joint")

# Task settings that an rsl_rl checkpoint of Isaac's own G1 tasks implies (it saves none): Isaac Lab 2.3's G1FlatEnvCfg
# and G1RoughEnvCfg (assets/isaac/g1_flat_env_cfg.py, g1_rough_env_cfg.py), i.e. their reward and command sets
# (lin_vel_y in +-0.5 / 0; ``reward_cfg`` "flat" / "rough_isaac") and Isaac's height-scan ray order ("xy", x fastest:
# GridPatternCfg ordering "xy", metalsim.learn.terrain.scan_grid). The rest are the task's defaults (elliptic contacts
# "recommended", base and feet-slide velocity at the COM, rough collision "boxes_local" with the exact height scan),
# the settings of the PARITY §1.5 transfer runs; FilmTask.adapter_kw["task"] (physics_dt) applies on top.
ISAAC_TASK_SETTINGS = {"flat": {"reward_cfg": "flat"}, "rough": {"reward_cfg": "rough_isaac", "scan_ordering": "xy"}}


def load_g1_checkpoint(path: str) -> dict:
    """A G1 policy checkpoint in either format, loaded with ``torch.load(weights_only=True)`` (Isaac's published
    checkpoints are downloaded pickles):

    * "metalsim": ``metalsim.learn.g1_velocity.train_g1``'s ``{"net": ActorCriticMLP state, "iterations", "terrain",
      task settings}``; joints in MetalSim's order;
    * "rsl_rl": rsl_rl's ``{"model_state_dict": actor.* / critic.* / std, "optimizer_state_dict", "iter", "infos"}``
      (Isaac Lab's training and its published PretrainedCheckpoints); the actor is read as
      ``metalsim.parity.side_by_side.load_policy`` reads it (the ``actor.*`` keys of an ELU MLP); joints in Isaac's
      order (the checkpoint's ``joint_names`` if it has them, else ``ISAAC_G1_JOINTS``).

    Returns dict(format, actor: the state dict of ``ActorCriticMLP.actor``, hidden, obs_dim, act_dim, iteration,
    terrain, engine, joints: the policy's joint order (None: MetalSim's), saved: the task settings the checkpoint
    saved)."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ck:
        fmt, net, it = "rsl_rl", ck["model_state_dict"], ck.get("iter")
        joints, saved = list(ck.get("joint_names") or ISAAC_G1_JOINTS), {}
    elif "net" in ck:
        fmt, net, it, joints = "metalsim", ck["net"], ck.get("iterations"), None
        saved = {k: ck[k] for k in G1_SAVED_TASK_KEYS if k in ck}
    else:
        raise ValueError(f"{path}: neither an rsl_rl checkpoint (model_state_dict) nor a MetalSim one (net)")
    sd = {k[len("actor."):]: v for k, v in net.items() if k.startswith("actor.")}
    lin = sorted(int(k.split(".")[0]) for k in sd if k.endswith(".weight"))
    hidden = tuple(int(sd[f"{i}.weight"].shape[0]) for i in lin[:-1])
    obs_dim, act_dim = int(sd[f"{lin[0]}.weight"].shape[1]), int(sd[f"{lin[-1]}.weight"].shape[0])
    # Isaac's G1 tasks: 12 + 3 x 37 policy inputs on flat ground, + 187 height-scan rays on rough terrain
    terrain = ck.get("terrain") if fmt == "metalsim" else {123: "flat", 310: "rough"}.get(obs_dim)
    return dict(format=fmt, actor=sd, hidden=hidden, obs_dim=obs_dim, act_dim=act_dim, iteration=it, terrain=terrain,
                engine=ck.get("engine", "mjwarp") if fmt == "metalsim" else "physx (Isaac Lab)", joints=joints,
                saved=saved)


def checkpoint_source(path: str | None) -> dict:
    """Where a G1 checkpoint comes from: Isaac Lab's published rsl_rl checkpoints (fetched by
    scripts/fetch_isaac_assets.sh to assets/isaac/pretrained/rsl_rl/<task id>/checkpoint.pt) map back to their URL and
    published sha256 (``tasks.ISAAC_PRETRAINED``); anything else is this repository's."""
    from metalsim.render.film import provenance, tasks

    rel = provenance.relpath(path) if path else None
    for tid, ent in tasks.ISAAC_PRETRAINED.items():
        if rel == ent["path"]:
            return dict(source="NVIDIA Isaac Lab 2.3 published pretrained checkpoint", task_id=tid,
                        source_url=ent["url"], published_sha256=ent["sha256"], iteration=ent["iteration"],
                        licence=tasks.NVIDIA_TERMS)
    return dict(source="this repository", source_url=None)


class G1Velocity:
    """Isaac Lab's G1 velocity task (metalsim.learn.g1_velocity, MuJoCo Warp) with a policy checkpoint evaluated in
    torch (the actor's mean; no Warp-bound policy, so the PPOWarp load-ordering bug of 2026-10-13 does not apply). The
    render model is the same G1 with its 43 USD visual meshes, on the task's own ground.

    Checkpoints (``load_g1_checkpoint``): MetalSim's (``train_g1``) or rsl_rl's (Isaac Lab's training and its
    published pretrained checkpoints). An rsl_rl actor reads and writes the joint blocks in Isaac's joint order
    (``ISAAC_G1_JOINTS``): its joint pos / joint vel / last action blocks are permuted from MetalSim's order by joint
    name and its actions back, exactly as metalsim/parity/side_by_side.py and scripts/diagnostics/g1_rough_transfer.py
    do; the 12-value head (base lin vel, base ang vel, projected gravity, command) and the 187 height-scan rays (Isaac's
    "xy" order, which the task is then required to use) are passed through.

    Task settings (``meta()["task_settings"]``): the ones the checkpoint saved (``G1_SAVED_TASK_KEYS``, as
    scripts/diagnostics/g1_policy_replay.py reads them; rsl_rl: ``ISAAC_TASK_SETTINGS``), then
    ``FilmTask.adapter_kw["task"]`` (settings a checkpoint does not save, e.g. ``physics_dt``, from its training log
    header). ``adapter_kw["terrain_level"]`` (rough terrain):
    every env starts on that difficulty row of its terrain column (default: the task's random initial levels 0-5).

    Done mask: the task's own termination and time-out terms (``G1VelocityTask.launch_reward_done_reset``: torso
    contact, Isaac's base_contact / illegal_contact, or a non-finite state; the 20 s time-out), read after the frame is
    captured; the task then resets those envs (curriculum included), as Isaac's ``ManagerBasedRLEnv.step``.
    ``info["terminated"]`` / ``info["time_out"]`` say which ended an episode.

    Gait (``record.gait_metrics``): each frame also carries the two foot colliders' centres (``feet``, world frame, by
    forward kinematics of the recorded ``qpos``) and the xy of their bounding-box corners (``feet_footprint``), the
    task's foot touch sensors above 1 N (``feet_contact``, the contact threshold of its reward terms) and the command
    (``cmd``); ``ground_height`` is the task's own collision-surface lookup (``metalsim.learn.terrain.collision_top``,
    the top of the MuJoCo collision geometry straight above a point, equal to ``mj_ray``; 0 on flat ground)."""

    backend = "mujoco"
    terrain = "flat"

    def __init__(self, task, film, checkpoint, viewer=None):
        import inspect
        from types import SimpleNamespace

        import mujoco
        import torch
        import warp as wp

        wp.config.quiet = True
        from metalsim.interop import torch_bridge as tb
        from metalsim.learn.g1_velocity import CONTROL_DT, G1VelocityTask
        from metalsim.learn.warp_policy import ActorCriticMLP, bump

        self.torch, self.wp, self.tb, self.bump = torch, wp, tb, bump
        akw = dict(getattr(task, "adapter_kw", None) or {})
        n = self.num_envs = int(film.num_envs)
        self.checkpoint, self.net, self.note = checkpoint, None, ""
        self.perm_obs = self.perm_act = None
        ck = load_g1_checkpoint(checkpoint) if checkpoint else None
        self.ckpt_format = ck["format"] if ck else None
        if ck is not None and ck["terrain"] not in (None, self.terrain):
            raise ValueError(f"checkpoint trained on {ck['terrain']!r} terrain, this film is {self.terrain!r}")
        if ck is not None and ck["format"] == "metalsim" and ck["engine"] != "mjwarp":
            raise ValueError(f"checkpoint trained on engine {ck['engine']!r}; the film adapter runs MuJoCo Warp")
        params = inspect.signature(G1VelocityTask.__init__).parameters
        saved = (ck["saved"] if ck["format"] == "metalsim" else ISAAC_TASK_SETTINGS[self.terrain]) if ck else {}
        self.saved_settings = {k: v for k, v in saved.items() if k in params}
        self.task_kw = dict(self.saved_settings) | dict(akw.get("task", {}))
        t0 = time.perf_counter()
        self.task = G1VelocityTask(n, terrain=self.terrain, seed=film.seed, **self.task_kw)
        self.build_s = time.perf_counter() - t0
        self.frame_dt = float(CONTROL_DT)
        self.terrain_level = akw.get("terrain_level")
        if self.terrain_level is not None and self.task.hfield is not None:
            tab = self.task.hfield["origin_table"]()                     # (levels, terrain types, 3)
            lv = np.full(n, int(self.terrain_level), np.int32)
            self.task.level.assign(lv)
            self.task.origins.assign(tab[lv, self.task.col.numpy()].astype(np.float32))
        self.ckpt_iter = ck["iteration"] if ck else None
        self.ckpt_source = checkpoint_source(checkpoint) if checkpoint else None
        if ck is not None:
            if ck["obs_dim"] == int(self.task.obs_dim) and ck["act_dim"] == int(self.task.act_dim):
                self.net = ActorCriticMLP(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to("mps")
                self.net.actor.load_state_dict({k: v.to("mps") for k, v in ck["actor"].items()})
                self.net.eval()
                if ck["joints"] is not None:          # rsl_rl: Isaac's joint order inside the three joint blocks
                    ours = [mujoco.mj_id2name(self.task.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                            for j in range(1, self.task.model.njnt)]
                    pol = list(ck["joints"])
                    if sorted(pol) != sorted(ours):
                        raise ValueError(f"checkpoint joints {pol} are not the task's {ours}")
                    # policy joint k <- our joint perm_obs[k]; our joint i <- policy output perm_act[i]
                    self.perm_obs = torch.as_tensor([ours.index(j) for j in pol], device="mps")
                    self.perm_act = torch.as_tensor([pol.index(j) for j in ours], device="mps")
                    self.policy_joints = pol
                if ck["format"] == "rsl_rl" and self.task.scanner is not None and self.task.scanner.ordering != "xy":
                    raise ValueError("an Isaac rsl_rl policy reads Isaac's height-scan ray order: scan_ordering='xy'")
            else:
                self.note = (f"checkpoint obs/act dims {ck['obs_dim']}/{ck['act_dim']} != task {self.task.obs_dim}/"
                             f"{self.task.act_dim}: zero actions")
                print("[film] " + self.note, flush=True)
        dev = self.task.device
        self.step_idx = wp.zeros(1, dtype=int, device=dev)
        self.pol = SimpleNamespace(step_idx=self.step_idx)       # reset randomization keyed by the film's step counter
        self.bufs = SimpleNamespace(rew=wp.zeros((1, n), dtype=float, device=dev),
                                    done=wp.zeros((1, n), dtype=float, device=dev),
                                    timeout=wp.zeros((1, n), dtype=float, device=dev))
        self.t_obs = tb.mps_tensor(self.task.obs)
        # gait data: the foot colliders (the G1's only foot geoms) posed on the host from each frame's qpos
        m = self.task.model
        fb = [int(self.task.foot_body[0]), int(self.task.foot_body[1])]
        self.foot_geoms = [int(next(g for g in range(m.ngeom) if m.geom_bodyid[g] == b and
                                    (m.geom_contype[g] or m.geom_conaffinity[g]))) for b in fb]
        self.foot_geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.foot_geoms]
        self._fk = mujoco.MjData(m)
        self.touch_adr = [int(self.task.touch_adr[0]), int(self.task.touch_adr[1])]

    def _feet(self, qpos):
        """For each env's ``qpos``: (N, 2, 3) world centres of the left / right foot colliders and (N, 2, 8, 2) the xy
        of their bounding boxes' corners (the footprint the foot can stand on)."""
        import mujoco

        m, d = self.task.model, self._fk
        out, fp = np.zeros((len(qpos), 2, 3)), np.zeros((len(qpos), 2, 8, 2))
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
        for e in range(len(qpos)):
            d.qpos[:] = qpos[e]
            mujoco.mj_kinematics(m, d)
            for f, g in enumerate(self.foot_geoms):
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                out[e, f] = d.geom_xpos[g]
                fp[e, f] = (d.geom_xpos[g] + (c + corners * h) @ d.geom_xmat[g].reshape(3, 3).T)[:, :2]
        return out, fp

    def ground_height(self, x, y):
        """Top of the task's collision surface straight above world points (0 on flat ground)."""
        if self.task.hfield is None:
            return np.zeros(np.shape(x))
        from metalsim.learn.terrain import collision_top

        return collision_top(self.task.hfield, np.ravel(x), np.ravel(y)).reshape(np.shape(x))

    def _qpos(self):
        return self.task.sim.d.qpos.numpy().astype(np.float64).reshape(self.num_envs, -1)

    def reset(self):
        self.task.reset_all()
        q = self._qpos()
        n = self.num_envs
        self.start = q[:, :3].copy()                      # first-episode start (after the reset's random xy / yaw)
        self.level0 = self.task.level.numpy().astype(np.int64).copy()
        self.col0 = self.task.col.numpy().astype(np.int64).copy()
        self._xy = q[:, :2].copy()
        self.dist = np.zeros(n)                           # path length of the root over the first episode [m]
        self.climb = np.zeros(n)                          # highest root height above its start [m]
        self.descent = np.zeros(n)                        # lowest root height below its start [m]
        self._alive = np.ones(n, bool)

    def step(self):
        wp, task, torch = self.wp, self.task, self.torch
        t0 = time.perf_counter()
        with wp.ScopedDevice(task.device):
            wp.launch(self.bump, dim=1, inputs=[self.step_idx])
            task.launch_obs(self.step_idx)
            v = task.sim._signal(); task.sim.after(v)
            with torch.no_grad():
                if self.net is None:
                    a = torch.zeros(self.num_envs, task.act_dim, device="mps")
                else:
                    o = self.t_obs
                    if self.perm_obs is not None:     # MetalSim's joint order -> the policy's (side_by_side.py)
                        nj, p = task.nj, self.perm_obs
                        o = torch.cat([o[:, :12], o[:, 12:12 + nj][:, p], o[:, 12 + nj:12 + 2 * nj][:, p],
                                       o[:, 12 + 2 * nj:12 + 3 * nj][:, p], o[:, 12 + 3 * nj:]], 1)
                    a = self.net.actor(o)
                    if self.perm_act is not None:
                        a = a[:, self.perm_act]
            task.action_scratch.assign(a.cpu().numpy().astype(np.float32))
            task.launch_apply_action(task.action_scratch)
            task.sim.step()
            task.sim.synchronize()
            qpos = self._qpos()                               # the frame: before the task resets done envs
            touch = task.sim.d.sensordata.numpy()[:, self.touch_adr] > 1.0
            cmd = task.cmd.numpy().copy()
            task.launch_reward_done_reset(self.pol, self.bufs)
            task.launch_timeouts(self.pol, self.bufs)
            task.sim.synchronize()
        sim_s = time.perf_counter() - t0
        done = self.bufs.done.numpy()[0] > 0.5
        time_out = self.bufs.timeout.numpy()[0] > 0.5
        terminated = done & ~time_out
        al = self._alive
        self.dist[al] += np.linalg.norm(qpos[al, :2] - self._xy[al], axis=1)
        dz = qpos[:, 2] - self.start[:, 2]
        self.climb[al] = np.maximum(self.climb[al], dz[al])
        self.descent[al] = np.minimum(self.descent[al], dz[al])
        self._xy = qpos[:, :2].copy()
        self._alive &= ~done
        info = dict(terminated=terminated, time_out=time_out, base_z=qpos[:, 2].copy(), level=self.level0, col=self.col0)
        feet, footprint = self._feet(qpos)
        snap = dict(qpos=qpos, feet=feet, feet_footprint=footprint, feet_contact=touch, cmd=cmd)
        return snap, done, info, dict(sim_s=sim_s)

    def summary(self, ends) -> str:
        """Per env over its first episode: path length, highest climb and lowest drop of the pelvis relative to its start
        (a drop beyond ~0.25 m is a robot down without torso contact, which the task does not terminate), end frame."""
        rows = [f"{e}:{self.dist[e]:.1f}/{self.climb[e]:+.2f}/{self.descent[e]:+.2f}" + (f"@{int(ends[e])}" if ends[e] >= 0 else "")
                for e in range(self.num_envs)]
        return ("[record] first episodes, env:path m/climb m/drop m[@end frame]" +
                (f" (level {self.level0.tolist()})" if self.terrain != "flat" else "") + ": " + " ".join(rows))

    def rank(self, first, gait=None):
        """Envs whose first episode did not terminate first; flat: then those whose pelvis never dropped more than
        0.25 m (not down on its knees); then, with ``gait`` (``record.gait_metrics``), a natural-gait preference:
        envs whose lower swing apex of the two feet is at least the median of these candidates and whose larger stance
        slip is at most their median (relative to the take's own envs: no absolute reference value exists); then flat:
        the longest path; rough: the most climbed, then the longest path."""
        n = self.num_envs
        term = np.asarray(first.get("terminated", np.zeros(n)), bool)
        down = (self.descent < -0.25) if self.terrain == "flat" else np.zeros(n, bool)
        self.natural = natural_gait(gait, [e for e in range(n) if not term[e] and not down[e]], n)
        if self.terrain == "flat":
            key = [(term[e], down[e], not self.natural[e], -self.dist[e], e) for e in range(n)]
        else:
            key = [(term[e], not self.natural[e], -self.climb[e], -self.dist[e], e) for e in range(n)]
        return [k[-1] for k in sorted(key)]

    def twin_spec(self):
        """The task's G1 with its visual meshes on the task's own ground: the plane, or the rough terrain's collision
        surface exactly as the task builds it (its heightfield, lowered under box-built cells, and Isaac's terrain
        boxes, merged into one static mesh ``terrain_boxes``; the task's per-world box slots hold the same boxes)."""
        import mujoco

        from metalsim.learn.g1_velocity import build_g1_model

        t = self.task
        hf = t.hfield
        if hf is None:
            m, info = build_g1_model(self.terrain, visuals=True, physics_dt=t.physics_dt, effort_limit=t.effort_limit)
            spec = info["spec"]
        else:
            vis = dict(hf, collision="hfield")              # the ground geom alone; boxes added below
            ch = hf.get("collision_hfield")
            if ch is not None:
                vis.update({k: ch[k] for k in ("nrow", "ncol", "size", "data", "zmin")})
            m, info = build_g1_model(self.terrain, vis, visuals=True, physics_dt=t.physics_dt, effort_limit=t.effort_limit)
            spec = info["spec"]
            boxes = hf.get("boxes")
            if hf.get("collision") in ("boxes", "meshes", "boxes_fine", "boxes_local") and boxes is not None and len(boxes):
                V, F, UV = terrain_box_faces(boxes)
                me = spec.add_mesh(); me.name = "terrain_boxes"
                me.uservert = V.ravel().tolist(); me.userface = F.ravel().tolist()
                g = spec.worldbody.add_geom(); g.name = "terrain_boxes"; g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.meshname = "terrain_boxes"; g.contype = 0; g.conaffinity = 0
                # A mesh without texcoords gets MuJoCo's object-linear 2D mapping in its geom frame, and MuJoCo aligns a
                # mesh's frame with its principal axes (here a tilted frame: the box tops came out smeared). Planar
                # texcoords per face instead, at the heightfield's world scale (s = 0.5 x, t = -0.5 y, times
                # texrepeat). With texuniform MuJoCo (and the film's renderer: builtin_geoms.material_uv_transform)
                # multiplies explicit texcoords by the geom's size, so they are stored divided by it.
                mc = spec.compile()
                size = mc.geom_size[mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_GEOM, "terrain_boxes")]
                me.usertexcoord = (UV / np.maximum(size[:2], 1e-9)).ravel().tolist()
            self.twin_terrain = dict(heightfield=f"{vis['nrow']} x {vis['ncol']} at {(ch or hf)['res']} m",
                                     boxes=int(len(boxes)) if boxes is not None else 0, collision=hf.get("collision"))
        if m.nq != t.model.nq:
            raise RuntimeError(f"visual G1 nq {m.nq} != task nq {t.model.nq}")
        return spec

    def meta(self):
        from metalsim.render.film import provenance

        t = self.task
        out = dict(policy=provenance.relpath(self.checkpoint) if self.checkpoint else None,
                   policy_iteration=getattr(self, "ckpt_iter", None), policy_format=self.ckpt_format,
                   policy_source=self.ckpt_source,
                   policy_joint_order=("Isaac's (ISAAC_G1_JOINTS), mapped by joint name" if self.perm_obs is not None
                                       else "MetalSim's" if self.net is not None else None),
                   foot_geoms=self.foot_geom_names, note=self.note, env_build_s=self.build_s,
                   physics="MuJoCo Warp BatchSim (metalsim.learn.g1_velocity)", terrain=self.terrain,
                   saved_task_settings=self.saved_settings, task_settings=self.task_kw,
                   task_resolved=dict(physics_dt=t.physics_dt, decimation=t.decimation, seed=t.seed,
                                      reward_cfg=t.reward_cfg, contact_cfg=t.contact_cfg,
                                      terrain_collision=t.terrain_collision if t.hfield is not None else None),
                   actions="actor mean", done="task terminations (torso contact, non-finite state) + 20 s time-out")
        if hasattr(self, "dist"):
            out["first_episode"] = dict(start=self.start.round(4).tolist(), path_m=self.dist.round(3).tolist(),
                                        climb_m=self.climb.round(3).tolist(), descent_m=self.descent.round(3).tolist())
        if t.hfield is not None:
            out.update(terrain_level=self.terrain_level, levels=getattr(self, "level0", np.zeros(0)).tolist(),
                       terrain_types=getattr(self, "col0", np.zeros(0)).tolist(),
                       twin_terrain=getattr(self, "twin_terrain", None))
        return out


def natural_gait(gait, cands, n) -> np.ndarray:
    """(n,) bool: among ``cands``, envs whose lower swing apex of the two feet is at least the candidates' median and
    whose larger stance slip is at most their median (``record.gait_metrics`` per-env values); all False without gait
    data."""
    out = np.zeros(n, bool)
    per = (gait or {}).get("per_env") or {}
    rows = {e: per.get(str(e)) for e in cands}
    rows = {e: r for e, r in rows.items() if r and r.get("apex_m") and r.get("slip_mps")
            and None not in r["apex_m"] + r["slip_mps"]}
    if len(rows) < 2:
        return out
    apex = {e: min(r["apex_m"]) for e, r in rows.items()}
    slip = {e: max(r["slip_mps"]) for e, r in rows.items()}
    ma, ms = np.median(list(apex.values())), np.median(list(slip.values()))
    for e in rows:
        out[e] = apex[e] >= ma and slip[e] <= ms
    return out


def terrain_box_faces(boxes):
    """Isaac's terrain boxes (``metalsim.learn.terrain.terrain_boxes`` rows: centre, half extents) as one mesh with 4
    vertices per face (6 outward quads per box) and planar texture coordinates per face in MuJoCo's object-linear
    convention (0.5 a, -0.5 b): tops (x, y), x faces (y, z), y faces (x, z). Returns (V (N, 3), F (M, 3), UV (N, 2))."""
    B = np.asarray(boxes, np.float64)
    c, h = B[:, None, :3], B[:, None, 3:6]
    q = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    Vs, UVs = [], []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        for sg in (-1.0, 1.0):
            # counter-clockwise seen from outside: (a, b, n) right-handed for n = +x, +z; y is (x, z, -y)
            qq = q if sg * (1 if ax != 1 else -1) > 0 else q[::-1]
            P = np.zeros((4, 3)); P[:, ax] = sg; P[:, a] = qq[:, 0]; P[:, b] = qq[:, 1]
            W = c + h * P[None]                                           # (boxes, 4, 3)
            Vs.append(W); UVs.append(np.stack([0.5 * W[..., a], -0.5 * W[..., b]], -1))
    V = np.concatenate(Vs, 1)                                            # (boxes, 24, 3): 6 faces x 4
    UV = np.concatenate(UVs, 1)
    quad = np.array([[0, 1, 2], [0, 2, 3]])
    F = (np.arange(6)[:, None, None] * 4 + quad[None]).reshape(-1, 3)   # (12, 3) per box
    F = (np.arange(len(B))[:, None, None] * 24 + F[None]).reshape(-1, 3)
    return V.reshape(-1, 3), F.astype(np.int32), UV.reshape(-1, 2)


class G1Flat(G1Velocity):
    """Isaac Lab's G1 flat-terrain velocity task (``Isaac-Velocity-Flat-G1-v0``)."""

    terrain = "flat"


class G1Rough(G1Velocity):
    """Isaac Lab's G1 rough-terrain velocity task (``Isaac-Velocity-Rough-G1-v0``): Isaac's rough terrain (10 difficulty
    rows x 20 terrain types, ``metalsim.learn.terrain.isaac_rough_terrain``) with the 187-ray height scan."""

    terrain = "rough"
