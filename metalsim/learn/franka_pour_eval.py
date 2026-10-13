"""Evaluation of a trained Franka-Pour policy.

Default (``--mode first --rows uniform``, deterministic): every env runs one episode from a reset-dataset row drawn
uniformly (Isaac's play mode freezes the curriculum: ``FrankaPourResetDatasetEnvCfg.play_mode``, ``curriculum_freeze``),
the actor's mean action at every step, until each env's first episode ends (success, a failure, or the 360-step
time-out). Reports the success rate over those episodes, the termination breakdown, and the pour metrics at each
episode's last step (fraction of particles in the receiver, spilled, still in the source cup).

Diagnostic variants, to relate this number to training's logged success:
  ``--stochastic``            sample actions from the policy's Gaussian, as during training
  ``--rows curriculum``       draw reset rows with the training curriculum, restored from the checkpoint's sampler state
  ``--mode continuous``       run ``--steps`` policy steps with resets, as a training rollout, and report training's
                              metrics: Isaac's ``Episode_Termination/success`` (mean over envs of the last finished
                              episode's flag, averaged over steps) and the success fraction of the episodes finished
Every finished episode is also written (row category, length, success) so rate-weighted statistics can be formed.

    python -m metalsim.learn.franka_pour_eval --checkpoint runs/franka_pour/logs/m414_2048_model_299.pt \
        --num-envs 2048 [--stochastic] [--rows curriculum] [--mode continuous --steps 360] --json out.json

The same ``--seed`` gives the same rows, so two physics configurations are compared on the same start states.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rigid-det", action="store_true")
    ap.add_argument("--mpm-grid-budget", default="isaac", choices=["isaac", "measured"])
    ap.add_argument("--mpm-cells-per-world", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--rows", default="uniform", choices=["uniform", "curriculum"])
    ap.add_argument("--mode", default="first", choices=["first", "continuous"])
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    from metalsim.learn import franka_pour as P
    from metalsim.learn import franka_pour_ppo as PPO

    pcfg = P.S.PourPhysicsConfig(num_envs=a.num_envs, rigid_deterministic=a.rigid_det,
                                 mpm_grid_budget=a.mpm_grid_budget, mpm_cells_per_world=a.mpm_cells_per_world)
    env = P.FrankaPourEnv(P.PourTaskConfig(num_envs=a.num_envs, seed=a.seed, physics=pcfg,
                                           curriculum_freeze=a.rows == "uniform"))
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    actor = PPO.Actor(84, 8, PPO.PourPPOConfig()).to(env.device)
    actor.load_state_dict(ckpt["actor_state_dict"])     # obs_normalization=False: the actor is the whole policy
    actor.eval()
    if a.rows == "curriculum":           # the training sampler as it was at the checkpoint
        s = env.curriculum._sampler
        for k, v in ckpt.get("sampler", {}).items():
            getattr(s, k).copy_(v.to(getattr(s, k).device))
        for k, v in ckpt.get("sampler_scalars", {}).items():
            setattr(s, k, v)
    torch.manual_seed(a.seed)
    obs = env.reset_all()
    n = a.num_envs
    cat_all = env._reset_dataset_states["category"]
    first_done = torch.zeros(n, dtype=torch.bool, device=env.device)
    out = {k: torch.zeros(n, device=env.device) for k in ("target", "spill", "source", "length")}
    terms = {k: torch.zeros(n, dtype=torch.bool, device=env.device) for k in env.termination_terms}
    episodes = []                        # (category, length, success, time_out) of every finished episode
    succ_term_steps = []
    k_succ = env.termination_terms.index("success")
    t0 = time.perf_counter()
    steps = 0
    limit = a.steps if a.mode == "continuous" else P.MAX_EPISODE_LENGTH + 2
    while steps < limit and (a.mode == "continuous" or not bool(first_done.all())):
        with torch.inference_mode():
            mean, std = actor(PPO.actor_input(obs))
            act = torch.distributions.Normal(mean, std).sample() if a.stochastic else mean
        row_cat = cat_all[env.reset_dataset_row_id].clone()
        env.process_action(act)
        env.simulate()
        # the pour state the termination terms see this step (before done envs reset)
        env.common_step_counter += 1
        env._invalidate_caches()
        src, tgt, spl = env._particle_region_masks()
        frac = [m.sum(dim=1).float() / env._num_particles for m in (src, tgt, spl)]
        length = env.episode_length_buf + 1
        env.common_step_counter -= 1
        obs, rew, term, tout, _ = env.post_physics()
        dones = term | tout
        ids = dones.nonzero().flatten()
        if ids.numel():
            episodes += list(zip(row_cat[ids].tolist(), length[ids].tolist(), env._term_dones["success"][ids].tolist(),
                                 env._term_dones["time_out"][ids].tolist()))
        succ_term_steps.append(float(env._last_episode_dones[:, k_succ].float().mean()))
        done = dones & ~first_done
        if bool(done.any()):
            out["source"][done], out["target"][done], out["spill"][done] = frac[0][done], frac[1][done], frac[2][done]
            out["length"][done] = length[done].float()
            for k in env.termination_terms:
                terms[k][done] = env._term_dones[k][done]
            first_done |= done
        steps += 1
    env.physics.check_status()            # raises on a sparse-grid capacity overflow during the rollout
    f = first_done
    ep = np.asarray(episodes, dtype=np.float64).reshape(-1, 4)
    res = {"checkpoint": a.checkpoint, "num_envs": n, "seed": a.seed, "steps": steps, "mode": a.mode, "rows": a.rows,
           "stochastic": a.stochastic, "wall_s": time.perf_counter() - t0,
           "physics": {"mpm_grid_budget": a.mpm_grid_budget, "mpm_cells_per_world": a.mpm_cells_per_world,
                       "rigid_deterministic": a.rigid_det},
           "first_episodes": int(f.sum()),
           "success_rate": float(terms["success"][f].float().mean()) if bool(f.any()) else None,
           "termination_rates": {k: float(v[f].float().mean()) for k, v in terms.items()} if bool(f.any()) else None,
           "mean_target_fraction": float(out["target"][f].mean()) if bool(f.any()) else None,
           "mean_spill_fraction": float(out["spill"][f].mean()) if bool(f.any()) else None,
           "mean_source_fraction": float(out["source"][f].mean()) if bool(f.any()) else None,
           "mean_episode_length": float(out["length"][f].mean()) if bool(f.any()) else None,
           "finished_episodes": int(len(ep)),
           "success_of_finished_episodes": float(ep[:, 2].mean()) if len(ep) else None,
           "episode_termination_success_mean": float(np.mean(succ_term_steps)),
           "grasping_fraction_of_finished": float(ep[:, 0].mean()) if len(ep) else None,
           "grid_overflow": "none (check_status passed)"}
    if len(ep):
        # each episode weighted by 1 / length: the share it has among episodes finishing per unit time
        w = 1.0 / ep[:, 1]
        res["success_rate_weighted"] = float((w * ep[:, 2]).sum() / w.sum())
        for name, c in (("grasping", 1.0), ("non_grasping", 0.0)):
            m = ep[:, 0] == c
            res[f"success_finished_{name}"] = float(ep[m, 2].mean()) if m.any() else None
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
        np.save(a.json.replace(".json", "_episodes.npy"), ep)


if __name__ == "__main__":
    main()
