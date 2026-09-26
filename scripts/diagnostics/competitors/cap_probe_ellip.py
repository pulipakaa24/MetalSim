"""Newton iteration-cap probe (runs/il3/cap_probe.py's protocol, by the 3.0 agent) for a chosen contact preset and action
source: does cap 10 (the task's) change states above the cap-100 re-run noise floor? States come from a rollout with
cap 100 (4096 envs; actions uniform random in [-1, 1] or a policy checkpoint's mean action; resets by the task). At 10
checkpoints the state (qpos, qvel, ctrl, qacc_warmstart) is saved; from each, one control step (8 substeps of 2.5 ms)
is run with cap C in {10, 20, 40} and with cap 100 (twice: the second is the noise floor), and compared per world.
usage: python cap_probe_ellip.py TERRAIN CONTACT_CFG random|CKPT.pt [OUT.json]"""
import sys, json, numpy as np, torch, warp as wp, mujoco_warp as mjw
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP, RolloutBuffers, bump
terrain, cfg, src = sys.argv[1], sys.argv[2], sys.argv[3]; OUT = sys.argv[4] if len(sys.argv) > 4 else None
import os
N = int(os.environ.get("CAP_N", "4096")); CAPS = [10, 20, 40, 100]
task = G1VelocityTask(N, terrain=terrain, seed=0, physics_dt=0.0025, reward_cfg="flat" if terrain == "flat" else "rough_isaac", contact_cfg=cfg)
d, m = task.sim.d, task.sim.m
task.reset_all()
net = None
if src != "random":
    ck = torch.load(src, map_location="mps", weights_only=False); sd = ck["net"]
    hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.startswith("actor.") and k.endswith("weight")), key=lambda s: int(s.split(".")[1]))[:-1])
    net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.load_state_dict(sd); net.eval()
class _Pol:
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); rng_step = wp.zeros(1, dtype=int, device="metal:0")
pol = _Pol(); bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, N), dtype=float, device="metal:0")
rng = np.random.default_rng(0)
wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0"); task.launch_obs(pol.rng_step); task.sim.synchronize()
def control_step(cap):
    m.opt.iterations = cap
    hit = np.zeros(N, bool); nmax = 0; nsum = np.zeros(N)
    with wp.ScopedDevice("metal:0"):
        for s in range(task.decimation):
            mjw.step(m, d)
            ni = d.solver_niter.numpy(); hit |= ni >= cap; nmax = max(nmax, int(ni.max())); nsum += ni
    return hit, nmax, nsum / task.decimation
F = ("qpos", "qvel", "ctrl", "qacc_warmstart")
save = lambda: {f: getattr(d, f).numpy().copy() for f in F}
def load(st):
    for f in F: getattr(d, f).assign(st[f])
def action():
    if net is None: return rng.uniform(-1, 1, (N, task.act_dim)).astype(np.float32)
    task.sim.synchronize()
    with torch.no_grad():
        return net.actor(torch.as_tensor(task.obs.numpy(), device="mps")).cpu().numpy().astype(np.float32)
res = {"terrain": terrain, "contact_cfg": cfg, "actions": src, "caps": CAPS, "points": []}
for k in range(200):
    a = wp.array(action(), dtype=float, device="metal:0")
    task.launch_apply_action(a)
    if k % 20 == 19:
        st = save(); hit100, n100, mean100 = control_step(100); ref = save()
        load(st); control_step(100); floor = save()
        row = {"step": k, "max_niter_cap100": n100, "mean_niter_cap100": float(mean100.mean()), "worlds_at_cap100": int(hit100.sum()),
               "pelvis_z_mean": float(st["qpos"][:, 2].mean()), "pelvis_z_below_0.4": int((st["qpos"][:, 2] < 0.4).sum())}
        dvf = np.abs(floor["qvel"] - ref["qvel"]).max(1); dqf = np.abs(floor["qpos"] - ref["qpos"]).max(1)
        row["floor"] = {"max_dqpos": float(dqf.max()), "max_dqvel": float(dvf.max()), "p99_dqvel": float(np.percentile(dvf, 99)), "median_dqvel": float(np.median(dvf)), "worlds_dqvel_gt_0.01": int((dvf > 0.01).sum())}
        for c in CAPS:
            load(st); hit, _, meanc = control_step(c); cur = save()
            dq = np.abs(cur["qpos"] - ref["qpos"]).max(1); dv = np.abs(cur["qvel"] - ref["qvel"]).max(1)
            row[f"cap{c}"] = {"worlds_hitting_cap": int(hit.sum()), "mean_niter": float(meanc.mean()), "max_dqpos": float(dq.max()), "max_dqvel": float(dv.max()),
                              "p99_dqvel": float(np.percentile(dv, 99)), "median_dqvel": float(np.median(dv)),
                              "max_dqvel_in_hitting": float(dv[hit].max()) if hit.any() else None,
                              "worlds_dqvel_gt_0.01": int((dv > 0.01).sum()), "worlds_dqvel_gt_0.1": int((dv > 0.1).sum()),
                              "max_dqvel_in_non_hitting": float(dv[~hit].max()) if (~hit).any() else 0.0}
        res["points"].append(row); print(json.dumps(row), flush=True)
        load(ref)
    else:
        control_step(100)
    wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
    task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.rng_step)
P = res["points"]
res["summary_floor"] = {k: (max(p["floor"][k] for p in P) if k != "worlds_dqvel_gt_0.01" else float(np.mean([p["floor"][k] for p in P]))) for k in P[0]["floor"]}
for c in CAPS:
    res[f"summary_cap{c}"] = {"worlds_hitting_cap_mean_per_step": float(np.mean([p[f"cap{c}"]["worlds_hitting_cap"] for p in P])),
                              "mean_niter": float(np.mean([p[f"cap{c}"]["mean_niter"] for p in P])),
                              "max_dqpos": max(p[f"cap{c}"]["max_dqpos"] for p in P), "max_dqvel": max(p[f"cap{c}"]["max_dqvel"] for p in P),
                              "p99_dqvel_max": max(p[f"cap{c}"]["p99_dqvel"] for p in P), "median_dqvel_max": max(p[f"cap{c}"]["median_dqvel"] for p in P),
                              "worlds_dqvel_gt_0.01_mean": float(np.mean([p[f"cap{c}"]["worlds_dqvel_gt_0.01"] for p in P])),
                              "worlds_dqvel_gt_0.1_mean": float(np.mean([p[f"cap{c}"]["worlds_dqvel_gt_0.1"] for p in P])),
                              "max_dqvel_in_hitting": max((p[f"cap{c}"]["max_dqvel_in_hitting"] or 0.0) for p in P)}
print(json.dumps({k: v for k, v in res.items() if k != "points"}, indent=1))
if OUT: json.dump(res, open(OUT, "w"), indent=1)
