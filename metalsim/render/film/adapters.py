"""Recorder adapters: the only per-task code of the filming tool. An adapter runs a task's own environment with its
policy and hands the recorder one frame at a time; everything else (selection, render twin, rendering, encoding,
provenance, doc) is generic (``metalsim.render.film.record`` / ``render``).

Protocol (duck-typed):

* ``backend``: "newton" (Newton ``Model`` / ``State``) or "mujoco" (MuJoCo / MuJoCo Warp ``qpos``)
* ``frame_dt`` [s] (one frame per policy step: Isaac's ``render_fps`` = 1 / step_dt), ``num_envs``
* ``reset()``; ``step() -> (snap, done, info, timing)``: advance one frame. ``snap``: "newton": ``body_q`` (N, B, 7)
  and ``particles`` (N, P, 3) or None, both in each env's own frame (env origin subtracted); "mujoco": ``qpos``
  (N, nq). The state is captured before done envs are reset (Isaac Lab's ``ManagerBasedRLEnv.step`` order), so a
  finished episode's last frame is in the film. ``done`` (N,) bool; ``info``: per-env arrays (outcome at a done);
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


class G1Flat:
    """Isaac Lab's G1 flat-terrain velocity task (metalsim.learn.g1_velocity, MuJoCo Warp) with an rsl_rl-style
    ActorCriticMLP checkpoint; the render model is the same G1 with its 43 USD visual meshes."""

    backend = "mujoco"

    def __init__(self, task, film, checkpoint, viewer=None):
        import torch
        import warp as wp

        wp.config.quiet = True
        from metalsim.interop import torch_bridge as tb
        from metalsim.learn.g1_velocity import G1VelocityTask, CONTROL_DT
        from metalsim.learn.warp_policy import ActorCriticMLP, bump

        self.torch, self.wp, self.tb, self.bump = torch, wp, tb, bump
        n = self.num_envs = int(film.num_envs)
        t0 = time.perf_counter()
        self.task = G1VelocityTask(n, terrain="flat", seed=film.seed)
        self.build_s = time.perf_counter() - t0
        self.frame_dt = float(CONTROL_DT)
        self.checkpoint, self.net, self.note = checkpoint, None, ""
        if checkpoint:
            ck = torch.load(checkpoint, map_location="mps", weights_only=False)
            if int(ck["obs_dim"]) == int(self.task.obs_dim) and int(ck["act_dim"]) == int(self.task.act_dim):
                self.net = ActorCriticMLP(ck["obs_dim"], ck["act_dim"], hidden=tuple(ck["hidden"])).to("mps")
                self.net.load_state_dict(ck["net"])
                self.net.eval()
                self.ckpt_iter = ck.get("iterations")
            else:
                self.note = (f"checkpoint obs/act dims {ck['obs_dim']}/{ck['act_dim']} != task {self.task.obs_dim}/"
                             f"{self.task.act_dim}: zero actions")
                print("[film] " + self.note, flush=True)
        self.step_idx = wp.zeros(1, dtype=int, device="metal:0")
        self.t_obs = tb.mps_tensor(self.task.obs)

    def reset(self):
        self.task.reset_all()

    def step(self):
        wp, task = self.wp, self.task
        t0 = time.perf_counter()
        wp.launch(self.bump, dim=1, inputs=[self.step_idx], device="metal:0")
        task.launch_obs(self.step_idx)
        v = task.sim._signal(); task.sim.after(v)
        with self.torch.no_grad():
            a = self.net.actor(self.t_obs) if self.net is not None else self.torch.zeros(self.num_envs, task.act_dim, device="mps")
        task.action_scratch.assign(a.cpu().numpy().astype(np.float32))
        task.launch_apply_action(task.action_scratch)
        task.sim.step()
        task.sim.synchronize()
        sim_s = time.perf_counter() - t0
        qpos = task.sim.d.qpos.numpy().astype(np.float64).reshape(self.num_envs, -1)
        n = self.num_envs
        return dict(qpos=qpos), np.zeros(n, bool), dict(base_z=qpos[:, 2].copy()), dict(sim_s=sim_s)

    def rank(self, first):
        return list(range(self.num_envs))

    def twin_spec(self):
        from metalsim.learn.g1_velocity import build_g1_model

        m, info = build_g1_model("flat", visuals=True)
        if m.nq != self.task.model.nq:
            raise RuntimeError(f"visual G1 nq {m.nq} != task nq {self.task.model.nq}")
        return info["spec"]

    def meta(self):
        return dict(policy=os.path.relpath(self.checkpoint, ROOT) if self.checkpoint and os.path.isabs(self.checkpoint)
                    else self.checkpoint, policy_iteration=getattr(self, "ckpt_iter", None), note=self.note,
                    env_build_s=self.build_s, physics="MuJoCo Warp BatchSim (metalsim.learn.g1_velocity)")
