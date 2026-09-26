"""Falls of a checkpoint on the MuJoCo Warp G1 flat task per contact preset: how many, when in the episode, in which
direction, under which command, and how repeatable across seeds. Same rollout as
scripts/diagnostics/contact_research/air_time_rollout.py (task's own commands / resets, mean action, 1024 envs x 1000
control steps); a fall = the task's termination (torso touch force > 1 N over the 15 ms history), i.e. terms[:, 7] < 0.

    python scripts/diagnostics/competitors/g1_falls.py CKPT [--contact_cfg P] [--seed S] [--envs N] [--steps T] [--out J]"""
import argparse, json, os, sys
import numpy as np, torch, warp as wp
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import G1VelocityTask, CONTROL_DT
from metalsim.learn.warp_policy import ActorCriticMLP, RolloutBuffers, bump, zero_int
from metalsim.interop import torch_bridge as tb

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--envs", type=int, default=1024); ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--contact_cfg", default="recommended"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default=None)
a = ap.parse_args(); wp.config.quiet = True
ck = torch.load(a.ckpt, map_location="mps", weights_only=False); sd = ck["net"]
hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.startswith("actor.") and k.endswith("weight")), key=lambda s: int(s.split(".")[1]))[:-1])
task = G1VelocityTask(a.envs, terrain="flat", seed=a.seed, physics_dt=0.0025, scan_ordering=ck.get("scan_ordering") or "ij", contact_cfg=a.contact_cfg)
net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.load_state_dict(sd); net.eval()
n = a.envs


class _Pol: step_idx = wp.zeros(1, dtype=int, device=task.device)
pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, n), dtype=float, device=task.device); bufs.done = wp.zeros((1, n), dtype=float, device=task.device)
action = wp.zeros((n, task.act_dim), dtype=float, device=task.device); t_action = tb.mps_tensor(action); t_obs = tb.mps_tensor(task.obs)
task.reset_all(); task.episode_stats()
wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device); task.launch_obs(pol.step_idx); task.sim.synchronize()


def pitch_roll(q):
    """pelvis pitch (+ = nose down / forward) and roll from the free-joint quaternion (w, x, y, z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return pitch, roll


ep_len = np.zeros(n, int); falls = []; timeouts = 0
prev_q = task.sim.d.qpos.numpy().copy(); prev_pitch = np.zeros(n); prev_roll = np.zeros(n); pitch_hist = np.zeros((n, 10)); z_hist = np.zeros((n, 10))
fell_envs = np.zeros(n, int)
for t in range(a.steps):
    with torch.no_grad():
        t_action.copy_(net.actor(t_obs.clone()))
    v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
    wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device)
    cmd = task.cmd.numpy().copy()
    task.launch_apply_action(action); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
    task.sim.synchronize()
    terms = task.terms.numpy(); done = bufs.done.numpy()[0] > 0.5
    ep_len += 1
    if done.any():
        fell = done & (terms[:, 7] < 0)
        for e in np.nonzero(fell)[0]:
            falls.append({"env": int(e), "t_step": int(t), "ep_len_s": float(ep_len[e] * CONTROL_DT), "cmd": cmd[e].round(3).tolist(),
                          "pitch_prev": float(prev_pitch[e]), "roll_prev": float(prev_roll[e]), "pitch_hist": pitch_hist[e].round(3).tolist(),
                          "z_hist": z_hist[e].round(3).tolist()})
            fell_envs[e] += 1
        timeouts += int((done & ~fell).sum()); ep_len[done] = 0
    q = task.sim.d.qpos.numpy(); p, r = pitch_roll(q[:, 3:7])
    pitch_hist = np.roll(pitch_hist, -1, axis=1); pitch_hist[:, -1] = p; z_hist = np.roll(z_hist, -1, axis=1); z_hist[:, -1] = q[:, 2]
    prev_pitch, prev_roll = p, r
    pitch_hist[done] = 0; z_hist[done] = 0

F = falls; nf = len(F)
ep_t = np.array([f["ep_len_s"] for f in F]) if nf else np.zeros(0)
pit = np.array([f["pitch_hist"][-3] for f in F]) if nf else np.zeros(0)     # 40 ms before the torso contact
rol = np.array([f["roll_prev"] for f in F]) if nf else np.zeros(0)
cmdn = np.array([np.linalg.norm(f["cmd"][:2]) for f in F]) if nf else np.zeros(0)
direction = {"forward (pitch > 0.3)": int((pit > 0.3).sum()), "backward (pitch < -0.3)": int((pit < -0.3).sum()),
             "sideways (|roll| > 0.3, |pitch| <= 0.3)": int(((np.abs(rol) > 0.3) & (np.abs(pit) <= 0.3)).sum()),
             "upright 40 ms before (|pitch|,|roll| <= 0.3)": int(((np.abs(pit) <= 0.3) & (np.abs(rol) <= 0.3)).sum())}
rep = {"ckpt": a.ckpt, "contact_cfg": a.contact_cfg, "seed": a.seed, "envs": n, "steps": a.steps, "falls": nf, "timeouts": timeouts,
       "episodes": nf + timeouts, "envs_that_fell": int((fell_envs > 0).sum()), "envs_fell_twice_or_more": int((fell_envs > 1).sum()),
       "fall_time_s": {"lt_1": int((ep_t < 1).sum()), "1_to_5": int(((ep_t >= 1) & (ep_t < 5)).sum()), "5_to_15": int(((ep_t >= 5) & (ep_t < 15)).sum()), "ge_15": int((ep_t >= 15).sum()),
                       "median": float(np.median(ep_t)) if nf else None},
       "direction": direction, "cmd_norm_at_fall_mean": float(cmdn.mean()) if nf else None,
       "cmd_norm_at_fall_lt_0.1": int((cmdn < 0.1).sum()), "yaw_cmd_abs_at_fall_mean": float(np.mean([abs(f["cmd"][2]) for f in F])) if nf else None,
       "falls_detail": F[:400]}
print(json.dumps({k: v for k, v in rep.items() if k != "falls_detail"}), flush=True)
if a.out: json.dump(rep, open(a.out, "w"), indent=1)
