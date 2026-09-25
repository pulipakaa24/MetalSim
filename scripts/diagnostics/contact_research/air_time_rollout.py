"""feet_air_time / feet_slide of a checkpoint on the MuJoCo Warp G1 flat task, with the gait statistics behind
them, to decide whether the air-time gap to Isaac is a contact-model or a policy quantity.

Same rollout and accounting as scripts/diagnostics/g1_reward_terms.py (task's own commands/resets, per-term
episode sums / 20 s over the episodes that ended), plus, from the task's ContactSensor (Isaac's semantics:
contact iff |net normal force| > 1 N, updated every substep):
  * every completed air phase and contact phase per foot (Isaac's last_air_time / last_contact_time on each change),
    so flicker (phases < 20 ms) and bounces at touch-down are counted;
  * fractions of control steps in single stance / double stance / flight (with |cmd_xy| > 0.1);
  * feet_slide recomputed with the foot's COM velocity (Isaac Lab 2.3.2's body_lin_vel_w = body_com_lin_vel_w) next
    to the task's (foot frame origin).

    python scripts/diagnostics/contact_research/air_time_rollout.py CKPT [--contact_tuning P] [--stochastic] [--out J]
"""
import argparse, contextlib, json, os, sys
import numpy as np, torch, warp as wp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import G1VelocityTask, CONTROL_DT
from metalsim.learn.warp_policy import ActorCriticMLP, RolloutBuffers, bump, zero_int
from metalsim.interop import torch_bridge as tb
from metalsim.physics import contact_tuning

TERM_NAMES = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "feet_slide", "joint_deviation_all", "flat_orientation_l2",
              "action_rate_l2", "termination_penalty", "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2", "dof_pos_limits"]

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--envs", type=int, default=1024); ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--contact_tuning", default=None); ap.add_argument("--stochastic", action="store_true"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args(); wp.config.quiet = True
ck = torch.load(a.ckpt, map_location="mps", weights_only=False); sd = ck["net"]
hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.startswith("actor.") and k.endswith("weight")), key=lambda s: int(s.split(".")[1]))[:-1])
with (contact_tuning.g1_model_tuning(a.contact_tuning) if a.contact_tuning else contextlib.nullcontext()):
    task = G1VelocityTask(a.envs, terrain="flat", seed=a.seed, physics_dt=0.0025, scan_ordering=ck.get("scan_ordering") or "ij")
net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.load_state_dict(sd); net.eval()
n = a.envs; cs = task.contact; m = task.model
fb = [int(task.foot_body[0]), int(task.foot_body[1])]; fr = [int(task.foot_root[0]), int(task.foot_root[1])]


class _Pol: step_idx = wp.zeros(1, dtype=int, device=task.device)
pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, n), dtype=float, device=task.device); bufs.done = wp.zeros((1, n), dtype=float, device=task.device)
action = wp.zeros((n, task.act_dim), dtype=float, device=task.device); t_action = tb.mps_tensor(action); t_obs = tb.mps_tensor(task.obs)
task.reset_all(); task.episode_stats()
wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device); task.launch_obs(pol.step_idx); task.sim.synchronize()
ep_sum = np.zeros((n, len(TERM_NAMES))); ep_slide_com = np.zeros(n); ep_len = np.zeros(n, int); done_sums = []; done_slide_com = []; fell = 0; timeout = 0
prev_la = cs._last_air.numpy()[:, :2].copy(); prev_lc = cs._last_con.numpy()[:, :2].copy()
air_phases, con_phases = [], []; modes = np.zeros(3); moving_steps = 0
for t in range(a.steps):
    with torch.no_grad():
        o = t_obs.clone(); act = net.actor(o)
        if a.stochastic: act = act + torch.randn_like(act) * net.log_std.exp()
        t_action.copy_(act)
    v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
    wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device)
    # command in force during this step (before a possible resample in the reward/reset launch)
    cmd = task.cmd.numpy().copy()
    task.launch_apply_action(action); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
    task.sim.synchronize()
    terms = task.terms.numpy(); done = bufs.done.numpy()[0] > 0.5
    # foot COM velocity (Isaac's body_lin_vel_w) from cvel at the subtree COM of the tree root
    d = task.sim.d
    cvel = d.cvel.numpy(); xipos = d.xipos.numpy(); stc = d.subtree_com.numpy()
    hist = cs._hist.numpy()[:, :, :2]                                    # (n, T, 2, 3)
    in_hist = np.linalg.norm(hist, axis=-1).max(1) > 1.0                 # (n, 2) Isaac's feet_slide contact flag
    sp = np.zeros((n, 2))
    for f in range(2):
        w = cvel[:, fb[f], :3]; vl = cvel[:, fb[f], 3:]
        vc = vl + np.cross(w, xipos[:, fb[f]] - stc[:, fr[f]])
        sp[:, f] = np.linalg.norm(vc[:, :2], axis=1)
    slide_com = -0.1 * (sp * in_hist).sum(1)
    # gait bookkeeping (skip envs that were reset this step)
    la = cs._last_air.numpy()[:, :2]; lc = cs._last_con.numpy()[:, :2]; cc = cs._cur_con.numpy()[:, :2]
    ok = ~done
    ch_a = (la != prev_la) & ok[:, None]; ch_c = (lc != prev_lc) & ok[:, None]
    air_phases += la[ch_a].tolist(); con_phases += lc[ch_c].tolist()
    prev_la = la.copy(); prev_lc = lc.copy()
    mv = (np.linalg.norm(cmd[:, :2], axis=1) > 0.1) & ok
    nc = (cc[mv] > 0).sum(1); modes += np.array([(nc == 1).sum(), (nc == 2).sum(), (nc == 0).sum()]); moving_steps += mv.sum()
    ep_sum += terms * CONTROL_DT; ep_slide_com += slide_com * CONTROL_DT; ep_len += 1
    if done.any():
        done_sums.append(ep_sum[done].copy()); done_slide_com.append(ep_slide_com[done].copy())
        fell += int((terms[done, 7] < 0).sum()); timeout += int((terms[done, 7] >= 0).sum())
        ep_sum[done] = 0; ep_slide_com[done] = 0; ep_len[done] = 0
S = np.concatenate(done_sums); per_term = S.mean(0) / 20.0
ap_ = np.array(air_phases); cp_ = np.array(con_phases)
rep = {"ckpt": a.ckpt, "contact_tuning": a.contact_tuning or "default", "stochastic": a.stochastic, "seed": a.seed, "episodes": int(len(S)),
       "falls": fell, "timeouts": timeout, "Episode_Reward": {k: round(float(v), 4) for k, v in zip(TERM_NAMES, per_term)},
       "feet_slide_com_velocity": round(float(np.concatenate(done_slide_com).mean() / 20.0), 4),
       "gait": {"air_phases": int(len(ap_)), "air_phase_mean_s": float(ap_.mean()) if len(ap_) else None,
                "air_phase_median_s": float(np.median(ap_)) if len(ap_) else None,
                "air_phases_lt_20ms_frac": float((ap_ < 0.02).mean()) if len(ap_) else None,
                "contact_phases": int(len(cp_)), "contact_phase_mean_s": float(cp_.mean()) if len(cp_) else None,
                "contact_phase_median_s": float(np.median(cp_)) if len(cp_) else None,
                "contact_phases_lt_20ms_frac": float((cp_ < 0.02).mean()) if len(cp_) else None,
                "single_stance_frac": float(modes[0] / max(moving_steps, 1)), "double_stance_frac": float(modes[1] / max(moving_steps, 1)),
                "flight_frac": float(modes[2] / max(moving_steps, 1))}}
print(json.dumps(rep, indent=1))
if a.out:
    json.dump(rep, open(a.out, "w"), indent=1)
