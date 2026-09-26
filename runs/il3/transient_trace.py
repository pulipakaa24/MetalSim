"""Early-transient mechanism. Training (PPOWarp) never calls task.reset_all(): every env's FIRST episode starts from
BatchSim's initial data = MuJoCo's qpos0 (root at the world origin, z = 0, all joints at zero; feet ~0.75 m below a flat
plane). This script plays a PPOWarp-initialised policy (seed 0, std 1, as iteration 1) for STEPS control steps from
(a) that initial data, as training does, or (b) after task.reset_all() (Isaac's spawn), stepping each 2.5 ms substep and
keeping a ring buffer of the last 24 substeps' state (qpos, qvel, ctrl) per world. When a world turns non-finite or
> 1000, its ring buffer is dumped for offline analysis in MuJoCo C (runs/il3/transient_analyze.py).
usage: python runs/il3/transient_trace.py TERRAIN SOLVER_CFG|contact:NAME noreset|reset STEPS OUT.npz"""
import sys, json, numpy as np, torch, warp as wp, mujoco_warp as mjw
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP, bump, RolloutBuffers
from metalsim.interop import torch_bridge as tb
terrain, preset, mode, STEPS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
N = 4096
kw = dict(contact_cfg=preset.split(":", 1)[1], solver_cfg=None) if preset.startswith("contact:") else dict(solver_cfg=preset)
task = G1VelocityTask(N, terrain=terrain, seed=0, physics_dt=0.0025, reward_cfg=f"{terrain}_il3", **kw)
d = task.sim.d
torch.manual_seed(0)                                   # PPOWarp: torch.manual_seed(cfg.seed) then ActorCriticMLP
net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=(512, 256, 128) if terrain != "flat" else (256, 128, 128)).to("mps").eval()
if mode == "reset":
    task.reset_all()
q0 = d.qpos.numpy()
start = {"root_z_mean": float(q0[:, 2].mean()), "root_xy_absmax": float(np.abs(q0[:, :2]).max()), "joints_absmax": float(np.abs(q0[:, 7:]).max()),
         "nacon": int(d.nacon.numpy()[0]), "min_dist": float(d.contact.dist.numpy()[:int(d.nacon.numpy()[0])].min()) if int(d.nacon.numpy()[0]) else None}
print("start state", json.dumps(start), flush=True)
class _Pol:
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); rng_step = wp.zeros(1, dtype=int, device="metal:0")
pol = _Pol(); bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, N), dtype=float, device="metal:0")
t_obs = tb.mps_tensor(task.obs); gen = torch.Generator(device="mps").manual_seed(0)
R = 24; RQ = np.zeros((R, N, d.qpos.shape[1]), np.float32); RV = np.zeros((R, N, d.qvel.shape[1]), np.float32); RC = np.zeros((R, N, d.ctrl.shape[1]), np.float32)
reset_ever = np.zeros(N, bool); dumps = []; blown_total = 0; first_ep_blown = 0; falls_first_ep = 0; ep_end_step = np.full(N, -1)
wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0"); wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
task.launch_obs(pol.rng_step); r = 0
for k in range(STEPS):
    v = task.sim._signal(); task.sim.after(v)
    with torch.no_grad():
        mu = net.actor_mean(t_obs.clone()); a = mu + torch.randn(mu.shape, generator=gen, device="mps")
    task.action_scratch.assign(a.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch)
    for s in range(task.decimation):
        RQ[r % R] = d.qpos.numpy(); RV[r % R] = d.qvel.numpy(); RC[r % R] = d.ctrl.numpy(); r += 1
        with wp.ScopedDevice("metal:0"):
            mjw.step(task.sim.m, d)
            for h in task.sim._substep_hooks:
                h()
    q = d.qpos.numpy(); qv = d.qvel.numpy()
    blown = ~np.isfinite(q).all(1) | ~np.isfinite(qv).all(1) | (np.abs(np.nan_to_num(q, nan=1e9)) > 1000).any(1) | (np.abs(np.nan_to_num(qv, nan=1e9)) > 1000).any(1)
    blown_total += int(blown.sum()); first_ep_blown += int((blown & ~reset_ever).sum())
    for e in np.nonzero(blown)[0][: max(0, 8 - len(dumps))]:
        order = [(r - R + i) % R for i in range(R)]
        dumps.append({"env": int(e), "step": k, "first_episode": bool(not reset_ever[e]), "qpos": RQ[order, e], "qvel": RV[order, e], "ctrl": RC[order, e],
                      "mass_scale": float(task.mass_scale[e]) if getattr(task, "mass_scale", None) is not None else 1.0})
    wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
    task.launch_reward_done_reset(pol, bufs)
    done = bufs.done.numpy()[0] > 0.5
    falls_first_ep += int((done & ~blown & ~reset_ever).sum()); ep_end_step[done & ~reset_ever] = k
    reset_ever |= done
    task.launch_obs(pol.rng_step)
res = {"terrain": terrain, "preset": preset, "mode": mode, "steps": STEPS, "start": start, "blown_total": blown_total,
       "blown_in_first_episode": first_ep_blown, "first_episode_falls": falls_first_ep, "worlds_never_reset": int((~reset_ever).sum()),
       "first_episode_end_step_median": float(np.median(ep_end_step[ep_end_step >= 0])) if (ep_end_step >= 0).any() else None}
print(json.dumps(res), flush=True)
np.savez_compressed(OUT, res=json.dumps(res), **{f"d{i}_{k}": v for i, dd in enumerate(dumps) for k, v in dd.items() if isinstance(v, np.ndarray)},
                    meta=json.dumps([{k: v for k, v in dd.items() if not isinstance(v, np.ndarray)} for dd in dumps]))
