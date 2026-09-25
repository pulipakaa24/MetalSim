"""Per-term reward breakdown of a MetalSim G1 checkpoint, in Isaac Lab's units, to compare with the
`Episode_Reward/<term>` lines of Isaac's rsl_rl training log (runs/parity/isaac_train_g1_flat_terms.txt).

Isaac's RewardManager reports, at every episode end, sum_t(weight * term_t * dt) / max_episode_length_s
(20 s), averaged over the episodes that ended. This script rolls a checkpoint's mean action for
--steps control steps over --envs environments with the task's own reset/command logic and reports
the same quantity for the thirteen terms the reward kernel exposes (tracking lin/ang, feet air time,
feet slide, joint deviation (all groups), flat orientation, action rate, termination, lin_vel_z, ang_vel_xy,
dof torques, dof acc, dof pos limits) under the task's reward set (--reward_cfg), plus the mean
return and episode length of the episodes that ended.

    python scripts/diagnostics/g1_reward_terms.py runs/policies/g1_flat_dt25_fixed_it300.pt --envs 1024 --steps 1000
    python scripts/diagnostics/g1_reward_terms.py runs/policies/g1_flat_newton_fixed_it300.pt --engine newton --newton_it 4 --newton_dt 0.00125
"""
import argparse, json
import numpy as np, torch, warp as wp

from metalsim.learn.g1_velocity import G1VelocityTask, CONTROL_DT
from metalsim.learn.warp_policy import ActorCriticMLP, RolloutBuffers, bump, zero_int
from metalsim.interop import torch_bridge as tb

TERM_NAMES = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "feet_slide", "joint_deviation_all", "flat_orientation_l2",
              "action_rate_l2", "termination_penalty", "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2", "dof_pos_limits"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--envs", type=int, default=1024); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--physics_dt", type=float, default=0.0025); ap.add_argument("--terrain", default="flat"); ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--reward_cfg", default=None, choices=(None, "flat", "rough"), help="default: the terrain's (flat -> Isaac G1FlatEnvCfg)")
    ap.add_argument("--engine", default="mjwarp", choices=("mjwarp", "newton")); ap.add_argument("--newton_it", type=int, default=4)
    ap.add_argument("--newton_dt", type=float, default=0.00125); ap.add_argument("--newton_limit_margin", default="0.15", help="'none' = USD limits")
    a = ap.parse_args(); wp.config.quiet = True
    ck = torch.load(a.ckpt, map_location="mps", weights_only=False); sd = ck["net"]
    hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.startswith("actor.") and k.endswith("weight")), key=lambda s: int(s.split(".")[1]))[:-1])
    ekw = {}
    if a.engine == "newton":
        lm = None if a.newton_limit_margin.lower() == "none" else float(a.newton_limit_margin)
        ekw = dict(engine="newton", newton_iterations=a.newton_it, newton_dt=a.newton_dt, newton_kw={"limit_margin": lm})
    task = G1VelocityTask(a.envs, terrain=a.terrain, seed=1, physics_dt=a.physics_dt, reward_cfg=a.reward_cfg, **ekw)
    net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.load_state_dict(sd); net.eval()
    n = a.envs
    class _Pol: step_idx = wp.zeros(1, dtype=int, device=task.device)
    pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device=task.device); bufs.done = wp.zeros((1, n), dtype=float, device=task.device)
    action = wp.zeros((n, task.act_dim), dtype=float, device=task.device); t_action = tb.mps_tensor(action); t_obs = tb.mps_tensor(task.obs)
    task.reset_all(); task.episode_stats()
    wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device); task.launch_obs(pol.step_idx); task.sim.synchronize()
    ep_sum = np.zeros((n, len(TERM_NAMES))); ep_len = np.zeros(n, int); done_sums = []; done_lens = []; term_counts = {"fell": 0, "timeout": 0}
    for t in range(a.steps):
        with torch.no_grad():
            o = t_obs.clone(); act = net.actor(o)
            if a.stochastic: act = act + torch.randn_like(act) * net.log_std.exp()
            t_action.copy_(act)
        v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
        wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device)
        task.launch_apply_action(action); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
        task.sim.synchronize()
        terms = task.terms.numpy(); done = bufs.done.numpy()[0] > 0.5
        ep_sum += terms * CONTROL_DT; ep_len += 1
        if done.any():
            done_sums.append(ep_sum[done].copy()); done_lens.append(ep_len[done].copy())
            term_counts["fell"] += int((terms[done, 7] < 0).sum()); term_counts["timeout"] += int((terms[done, 7] >= 0).sum())
            ep_sum[done] = 0; ep_len[done] = 0
    if not done_sums:
        print("no episode ended; increase --steps"); return
    S = np.concatenate(done_sums); L = np.concatenate(done_lens); episode_s = 20.0
    per_term = S.mean(0) / episode_s
    ret = S.sum(1); print(json.dumps({"ckpt": a.ckpt, "iterations": ck.get("iterations"), "episodes": int(len(L)), "mean_episode_length": float(L.mean()), "mean_return": float(ret.mean()),
                                      "terminations": term_counts, "Episode_Reward": {k: round(float(v), 4) for k, v in zip(TERM_NAMES, per_term)},
                                      "note": "per-second average over the episode like Isaac's RewardManager; joint_deviation_all = hip+arms+fingers+torso", "reward_cfg": task.reward_cfg,
                                      "engine": task.engine, "physics_dt": task.physics_dt}, indent=1))


if __name__ == "__main__":
    main()
