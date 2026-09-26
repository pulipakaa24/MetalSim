"""Does the Newton-solver iteration cap change states under the task's default contact_cfg="recommended"
(tau10_impact_hardlimits)? States come from a rollout with cap 100 (4096 envs, uniform random actions in [-1, 1], the
benchmark protocol: walking, falling, lying; resets by the task). At 10 checkpoints the full state (qpos, qvel, ctrl,
qacc_warmstart) is saved; from each, one control step (8 substeps of 2.5 ms) is run with cap C in {10 (the task's),
20, 40} and with cap 100, and compared: worlds where any substep reached the cap, and the largest |dqpos| / |dqvel| over
worlds after that one control step (per-world solver exit is at tolerance 1e-8 here, MuJoCo's default).
usage: python runs/il3/cap_probe.py TERRAIN [OUT.json]"""
import sys, json, numpy as np, warp as wp, mujoco_warp as mjw
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import RolloutBuffers, bump
terrain = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
N = 4096; CAPS = [10, 20, 40]
task = G1VelocityTask(N, terrain=terrain, seed=0, physics_dt=0.0025, reward_cfg="flat" if terrain == "flat" else "rough_isaac",
                      contact_cfg="recommended")
d, m = task.sim.d, task.sim.m
task.reset_all()
class _Pol:
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); rng_step = wp.zeros(1, dtype=int, device="metal:0")
pol = _Pol(); bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, N), dtype=float, device="metal:0")
rng = np.random.default_rng(0)
def control_step(cap):
    m.opt.iterations = cap
    hit = np.zeros(N, bool); nmax = 0
    with wp.ScopedDevice("metal:0"):
        for s in range(task.decimation):
            mjw.step(m, d)
            ni = d.solver_niter.numpy(); hit |= ni >= cap; nmax = max(nmax, int(ni.max()))
    return hit, nmax
F = ("qpos", "qvel", "ctrl", "qacc_warmstart")
save = lambda: {f: getattr(d, f).numpy().copy() for f in F}
def load(st):
    for f in F: getattr(d, f).assign(st[f])
res = {"terrain": terrain, "contact_cfg": "recommended", "caps": CAPS, "points": []}
for k in range(200):
    a = wp.array(rng.uniform(-1, 1, (N, task.act_dim)).astype(np.float32), dtype=float, device="metal:0")
    task.launch_apply_action(a)
    if k % 20 == 19:
        st = save(); hit100, n100 = control_step(100); ref = save()
        row = {"step": k, "max_niter_cap100": n100, "worlds_at_cap100": int(hit100.sum())}
        for c in CAPS:
            load(st); hit, _ = control_step(c); cur = save()
            dq = np.abs(cur["qpos"] - ref["qpos"]).max(1); dv = np.abs(cur["qvel"] - ref["qvel"]).max(1)
            row[f"cap{c}"] = {"worlds_hitting_cap": int(hit.sum()), "max_dqpos": float(dq.max()), "max_dqvel": float(dv.max()),
                              "worlds_dqvel_gt_0.01": int((dv > 0.01).sum()), "max_dqvel_in_non_hitting": float(dv[~hit].max()) if (~hit).any() else 0.0}
        res["points"].append(row); print(json.dumps(row), flush=True)
        load(ref)                                           # continue the rollout on the cap-100 trajectory
    else:
        control_step(100)
    wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
    task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.rng_step)
for c in CAPS:
    P = res["points"]
    res[f"summary_cap{c}"] = {"worlds_hitting_cap_mean_per_step": float(np.mean([p[f"cap{c}"]["worlds_hitting_cap"] for p in P])),
                              "max_dqpos": max(p[f"cap{c}"]["max_dqpos"] for p in P), "max_dqvel": max(p[f"cap{c}"]["max_dqvel"] for p in P)}
print(json.dumps({k: v for k, v in res.items() if k != "points"}, indent=1))
if OUT: json.dump(res, open(OUT, "w"), indent=1)
