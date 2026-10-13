"""Deterministic evaluation of a trained Franka-Pour policy: every env runs one episode from a reset-dataset row drawn
uniformly (Isaac's play mode freezes the curriculum: ``FrankaPourResetDatasetEnvCfg.play_mode``, ``curriculum_freeze``),
the actor's mean action at every step, until each env's first episode ends (success, a failure, or the 360-step
time-out). Reports the success rate over those episodes, the termination breakdown, and the pour metrics at each
episode's last step (fraction of particles in the receiver, spilled, still in the source cup).

    python -m metalsim.learn.franka_pour_eval --checkpoint runs/franka_pour/logs/m414_2048_model_299.pt \
        --num-envs 2048 [--mpm-grid-budget measured --mpm-cells-per-world 256] --json out.json

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
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    from metalsim.learn import franka_pour as P
    from metalsim.learn import franka_pour_ppo as PPO

    pcfg = P.S.PourPhysicsConfig(num_envs=a.num_envs, rigid_deterministic=a.rigid_det,
                                 mpm_grid_budget=a.mpm_grid_budget, mpm_cells_per_world=a.mpm_cells_per_world)
    env = P.FrankaPourEnv(P.PourTaskConfig(num_envs=a.num_envs, seed=a.seed, physics=pcfg, curriculum_freeze=True))
    actor = PPO.Actor(84, 8, PPO.PourPPOConfig()).to(env.device)
    actor.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"])
    actor.eval()
    obs = env.reset_all()
    rows = env.reset_dataset_row_id.cpu().numpy().copy()
    n = a.num_envs
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    out = {k: torch.zeros(n, device=env.device) for k in ("target", "spill", "source", "length")}
    terms = {k: torch.zeros(n, dtype=torch.bool, device=env.device) for k in env.termination_terms}
    t0 = time.perf_counter()
    steps = 0
    while not bool(finished.all()) and steps < P.MAX_EPISODE_LENGTH + 2:
        with torch.inference_mode():
            mean, _ = actor(PPO.actor_input(obs))
        env.process_action(mean)
        env.simulate()
        # the pour state the termination terms see this step (before done envs reset)
        env.common_step_counter += 1
        env._invalidate_caches()
        src, tgt, spl = env._particle_region_masks()
        frac = [m.sum(dim=1).float() / env._num_particles for m in (src, tgt, spl)]
        length = env.episode_length_buf + 1
        env.common_step_counter -= 1
        obs, rew, term, tout, _ = env.post_physics()
        done = (term | tout) & ~finished
        if bool(done.any()):
            out["source"][done], out["target"][done], out["spill"][done] = frac[0][done], frac[1][done], frac[2][done]
            out["length"][done] = length[done].float()
            for k in env.termination_terms:
                terms[k][done] = env._term_dones[k][done]
            finished |= done
        steps += 1
    env.physics.check_status()            # raises on a sparse-grid capacity overflow during the rollout
    f = finished.cpu().numpy()
    res = {"checkpoint": a.checkpoint, "num_envs": n, "seed": a.seed, "steps": steps, "episodes": int(f.sum()),
           "wall_s": time.perf_counter() - t0, "rows_sha": int(np.bitwise_xor.reduce(rows * 2654435761 % 2**31)),
           "physics": {"mpm_grid_budget": a.mpm_grid_budget, "mpm_cells_per_world": a.mpm_cells_per_world,
                       "rigid_deterministic": a.rigid_det},
           "success_rate": float(terms["success"][finished].float().mean()),
           "termination_rates": {k: float(v[finished].float().mean()) for k, v in terms.items()},
           "mean_target_fraction": float(out["target"][finished].mean()),
           "mean_spill_fraction": float(out["spill"][finished].mean()),
           "mean_source_fraction": float(out["source"][finished].mean()),
           "mean_episode_length": float(out["length"][finished].mean()),
           "grid_overflow": "none (check_status passed)"}
    st = env._reset_dataset_states
    cat = st["category"][torch.as_tensor(rows, device=env.device)].bool()
    for name, m in (("grasping_rows", cat), ("non_grasping_rows", ~cat)):
        m = m & finished
        res[f"success_rate_{name}"] = float(terms["success"][m].float().mean()) if bool(m.any()) else None
        res[f"episodes_{name}"] = int(m.sum())
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
