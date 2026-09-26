"""Rough (and flat) il3 blow-up diagnosis: play a trained PPOWarp policy (stochastic, as in training) on the il3 task for
STEPS control steps with a given solver preset, stepping every 2.5 ms substep by hand, and record
  * per substep: the largest Newton iteration count over worlds, worlds at the cap, BatchSim overflow flags;
  * per blow-up (a world whose qpos/qvel turns non-finite or > 1000, the task's own rule), before the task resets it:
    terrain level and column, torso mass scale, steps since its last push, and over the previous 10 control steps the
    largest joint-limit excursion (and joint), the largest joint speed (and joint), the deepest contact penetration.
Compared with the same statistics over all worlds (base rates).
usage: python runs/il3/blowup_probe.py TERRAIN CKPT PRESET [STEPS] [OUT.json]"""
import sys, json, collections, numpy as np, torch, warp as wp, mujoco, mujoco_warp as mjw
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP, bump, RolloutBuffers
from metalsim.interop import torch_bridge as tb
terrain, ckpt, preset = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
OUT = sys.argv[5] if len(sys.argv) > 5 else None
N = 4096
task = G1VelocityTask(N, terrain=terrain, seed=0, physics_dt=0.0025, reward_cfg=f"{terrain}_il3", solver_cfg=preset)
m = task.model; d = task.sim.d; cap = int(task.sim.m.opt.iterations)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=tuple(ck["hidden"])); net.load_state_dict(ck["net"]); net = net.to("mps").eval()
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
adr = np.array([m.jnt_qposadr[j] for j in range(1, m.njnt)]); dadr = np.array([m.jnt_dofadr[j] for j in range(1, m.njnt)])
lo = m.jnt_range[1:, 0]; hi = m.jnt_range[1:, 1]
task.reset_all()

class _Pol:
    step_idx = wp.zeros(1, dtype=int, device="metal:0")
    rng_step = wp.zeros(1, dtype=int, device="metal:0")
pol = _Pol(); bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, N), dtype=float, device="metal:0")
t_obs = tb.mps_tensor(task.obs); std = torch.exp(net.log_std.detach())
gen = torch.Generator(device="mps").manual_seed(0)
H = 10
exc_h = np.zeros((H, N)); excj_h = np.zeros((H, N), int); spd_h = np.zeros((H, N)); spdj_h = np.zeros((H, N), int); pen_h = np.zeros((H, N))
last_push = np.full(N, -10**6); prev_left = task.push_left.numpy().copy()
niter_max = []; at_cap = 0; ov = collections.Counter(); events = []
wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0"); wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
task.launch_obs(pol.rng_step)
for k in range(STEPS):
    v = task.sim._signal(); task.sim.after(v)
    with torch.no_grad():
        mu = net.actor_mean(t_obs.clone())
        a = mu + std * torch.randn(mu.shape, generator=gen, device="mps")
    task.action_scratch.assign(a.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch)
    pmax = np.zeros(N)
    for s in range(task.decimation):
        with wp.ScopedDevice("metal:0"):
            mjw.step(task.sim.m, d)
            for h in task.sim._substep_hooks:
                h()
        ni = d.solver_niter.numpy(); niter_max.append(int(ni.max())); at_cap += int((ni >= cap).sum())
        nc = int(d.nacon.numpy()[0])
        if nc:
            dist = d.contact.dist.numpy()[:nc]; w = d.contact.worldid.numpy()[:nc]; ea = d.contact.efc_address.numpy()[:nc, 0]
            act = ea >= 0
            np.maximum.at(pmax, w[act], -dist[act])
    q = d.qpos.numpy(); qv = d.qvel.numpy()
    ex = np.maximum(lo - q[:, adr], q[:, adr] - hi); sp = np.abs(qv[:, dadr])
    exc_h = np.roll(exc_h, 1, 0); excj_h = np.roll(excj_h, 1, 0); spd_h = np.roll(spd_h, 1, 0); spdj_h = np.roll(spdj_h, 1, 0); pen_h = np.roll(pen_h, 1, 0)
    exc_h[0] = np.nan_to_num(ex.max(1), nan=9e9); excj_h[0] = np.nan_to_num(ex, nan=9e9).argmax(1)
    spd_h[0] = np.nan_to_num(sp.max(1), nan=9e9); spdj_h[0] = np.nan_to_num(sp, nan=9e9).argmax(1); pen_h[0] = pmax
    blown = ~np.isfinite(q).all(1) | ~np.isfinite(qv).all(1) | (np.abs(np.nan_to_num(q, nan=1e9)) > 1000).any(1) | (np.abs(np.nan_to_num(qv, nan=1e9)) > 1000).any(1)
    lvl = task.level.numpy(); col = task.col.numpy()
    for e in np.nonzero(blown)[0]:
        i1 = int(np.argmax(exc_h[1:, e])) + 1          # history before the blow-up step
        events.append({"step": k, "env": int(e), "level": int(lvl[e]), "col": int(col[e]), "mass_scale": float(task.mass_scale[e]),
                       "steps_since_push": int(k - last_push[e]),
                       "pre_limit_excursion": float(exc_h[1:, e].max()), "pre_limit_joint": names[int(excj_h[i1, e])],
                       "pre_joint_speed": float(spd_h[1:, e].max()), "pre_speed_joint": names[int(spdj_h[int(np.argmax(spd_h[1:, e])) + 1, e])],
                       "pre_penetration_m": float(pen_h[1:, e].max()),
                       "blowup_speed_joint": names[int(spdj_h[0, e])]})
    wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0"); wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
    task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.rng_step)
    pl = task.push_left.numpy(); last_push[pl > prev_left + 1.0] = k; prev_left = pl.copy()
ov = task.sim.overflow_flags()          # sticky per-world bits: worlds that ever overflowed
nm = np.array(niter_max)
res = {"terrain": terrain, "preset": preset, "ckpt": ckpt, "steps": STEPS, "cap": cap,
       "niter": {"max": int(nm.max()), "p99.9": float(np.percentile(nm, 99.9)), "substep_worlds_at_cap": at_cap, "of": len(nm) * N},
       "overflow": dict(ov), "blowups": len(events)}
if events:
    E = events
    res["blowup_levels"] = collections.Counter(e["level"] for e in E).most_common()
    res["blowup_cols"] = collections.Counter(e["col"] for e in E).most_common(8)
    res["base_rate_levels"] = collections.Counter(task.level.numpy().tolist()).most_common()
    ms = np.array([e["mass_scale"] for e in E]); res["mass_scale_blowups_mean"] = float(ms.mean()); res["mass_scale_all_mean"] = float(task.mass_scale.mean())
    res["mass_scale_blowups_frac_above_1.1"] = float((ms > 1.1).mean()); res["mass_scale_all_frac_above_1.1"] = float((task.mass_scale > 1.1).mean())
    sp = np.array([e["steps_since_push"] for e in E]); res["blowups_within_50_steps_of_push"] = float((sp < 50).mean())
    res["pre_limit_joint"] = collections.Counter(e["pre_limit_joint"] for e in E).most_common(6)
    res["pre_speed_joint"] = collections.Counter(e["pre_speed_joint"] for e in E).most_common(6)
    res["blowup_speed_joint"] = collections.Counter(e["blowup_speed_joint"] for e in E).most_common(6)
    for f in ("pre_limit_excursion", "pre_joint_speed", "pre_penetration_m"):
        x = np.array([e[f] for e in E]); res[f] = {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max())}
    res["events"] = E[:200]
# base rates over all worlds, last step: pushes within 50 steps
res["all_worlds_pushed_within_50_steps"] = float(((STEPS - 1 - last_push) < 50).mean())
print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1))
if OUT: json.dump(res, open(OUT, "w"), indent=1)
