"""Dump G1-task states (qpos, qvel, ctrl, qacc_warmstart) from a policy rollout, for CPU-side solver analysis
(elliptic-cone constraint-state changes per Newton iteration, iteration counts). Saves the task's MjModel as .mjb.
    python g1_state_dump.py CKPT OUT_PREFIX [--envs 256] [--steps 400] [--contact_cfg tau10_impact_hardlimits_ellip10] [--seed 1]"""
import argparse, os, sys, numpy as np, torch, warp as wp, mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP, RolloutBuffers, bump, zero_int
from metalsim.interop import torch_bridge as tb
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--envs", type=int, default=256); ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--contact_cfg", default="tau10_impact_hardlimits_ellip10"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--at", default="2,10,30,60,100,150,200,300,399")
a = ap.parse_args(); wp.config.quiet = True
ck = torch.load(a.ckpt, map_location="mps", weights_only=False); sd = ck["net"]
hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.startswith("actor.") and k.endswith("weight")), key=lambda s: int(s.split(".")[1]))[:-1])
task = G1VelocityTask(a.envs, terrain="flat", seed=a.seed, physics_dt=0.0025, scan_ordering=ck.get("scan_ordering") or "ij", contact_cfg=a.contact_cfg)
net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.load_state_dict(sd); net.eval()
n = a.envs; mujoco.mj_saveModel(task.model, a.out + ".mjb")
class _Pol: step_idx = wp.zeros(1, dtype=int, device=task.device)
pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, n), dtype=float, device=task.device); bufs.done = wp.zeros((1, n), dtype=float, device=task.device)
action = wp.zeros((n, task.act_dim), dtype=float, device=task.device); t_action = tb.mps_tensor(action); t_obs = tb.mps_tensor(task.obs)
task.reset_all(); task.episode_stats()
wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device); task.launch_obs(pol.step_idx); task.sim.synchronize()
AT = set(int(x) for x in a.at.split(",")); out = {}
for t in range(a.steps):
    with torch.no_grad():
        t_action.copy_(net.actor(t_obs.clone()))
    v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
    wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device)
    task.launch_apply_action(action)
    if t in AT:   # state right before the physics of this control step (ctrl already applied)
        task.sim.synchronize(); d = task.sim.d
        out[f"t{t}"] = dict(qpos=d.qpos.numpy().copy(), qvel=d.qvel.numpy().copy(), ctrl=d.ctrl.numpy().copy(), qacc_warmstart=d.qacc_warmstart.numpy().copy(), act=d.act.numpy().copy())
    task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
task.sim.synchronize()
np.savez_compressed(a.out + ".npz", steps=np.array(sorted(AT)), **{f"{k}_{f}": v for k, dd in out.items() for f, v in dd.items()},
                    opt=np.array([task.model.opt.iterations, task.model.opt.ls_iterations, task.model.opt.cone, task.model.opt.impratio, task.model.opt.timestep]),
                    njmax=task.sim.d.njmax, naconmax=task.sim.d.naconmax, nconmax=task.sim.d.naconmax // n)
print("saved", a.out + ".npz", "worlds", n, "snapshots", sorted(AT), "njmax", task.sim.d.njmax, "naconmax", task.sim.d.naconmax, flush=True)
