"""Policy side of the early transient: roll a checkpoint (stochastic, as in training) for 100 control steps x 4096 envs on
a task and report the action statistics per joint group (mean |mu|, std, fraction |a| > 1 / > 3) and, for each joint, the
commanded position target default + 0.5 a against the joint range (fraction of steps beyond, largest overshoot).
usage: python runs/il3/policy_stats.py CKPT REWARD_CFG SOLVER_CFG [OUT.json]"""
import sys, json, re, numpy as np, torch, warp as wp, mujoco
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP, bump, RolloutBuffers
from metalsim.interop import torch_bridge as tb
ckpt, rc, sc = sys.argv[1], sys.argv[2], sys.argv[3]; OUT = sys.argv[4] if len(sys.argv) > 4 else None
N = 4096
task = G1VelocityTask(N, terrain="rough", seed=0, physics_dt=0.0025, reward_cfg=rc, solver_cfg=None if sc == "none" else sc)
m = task.model
if ckpt == "init":
    torch.manual_seed(0); net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=(512, 256, 128)).to("mps").eval()
else:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=tuple(ck["hidden"])); net.load_state_dict(ck["net"]); net = net.to("mps").eval()
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
jr = np.array([m.jnt_range[m.actuator_trnid[i][0]] for i in range(m.nu)]); default = m.key_qpos[0][7:]
grp = lambda nm: "finger" if re.search(r"_(zero|one|two|three|four|five|six)_joint", nm) else ("arm" if re.search("shoulder|elbow", nm) else "leg/torso")
G = np.array([grp(nm) for nm in names])
task.reset_all()
class _Pol:
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); rng_step = wp.zeros(1, dtype=int, device="metal:0")
pol = _Pol(); bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, N), dtype=float, device="metal:0")
t_obs = tb.mps_tensor(task.obs); std = torch.exp(net.log_std.detach()); gen = torch.Generator(device="mps").manual_seed(0)
MU, A = [], []
wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0"); wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
task.launch_obs(pol.rng_step)
for k in range(100):
    v = task.sim._signal(); task.sim.after(v)
    with torch.no_grad():
        mu = net.actor_mean(t_obs.clone()); a = mu + std * torch.randn(mu.shape, generator=gen, device="mps")
    MU.append(mu.cpu().numpy()); A.append(a.cpu().numpy())
    task.action_scratch.assign(A[-1].astype(np.float32)); task.launch_apply_action(task.action_scratch); task.sim.step()
    wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
    task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.rng_step)
MU = np.concatenate(MU); A = np.concatenate(A)
tgt = default[None] + 0.5 * A
over = np.maximum(jr[None, :, 0] - tgt, tgt - jr[None, :, 1])
res = {"ckpt": ckpt, "reward_cfg": rc, "solver_cfg": sc, "std_param": std.cpu().numpy().round(3).tolist(), "groups": {}}
for g in ("leg/torso", "arm", "finger"):
    s = G == g
    res["groups"][g] = {"mean_abs_mu": float(np.abs(MU[:, s]).mean()), "max_abs_mu": float(np.abs(MU[:, s]).max()),
                        "frac_abs_a_gt1": float((np.abs(A[:, s]) > 1).mean()), "frac_abs_a_gt3": float((np.abs(A[:, s]) > 3).mean()),
                        "frac_target_beyond_limit": float((over[:, s] > 0).mean()), "max_target_overshoot_rad": float(over[:, s].max())}
res["worst_joints_target_beyond"] = sorted([(names[i], round(float((over[:, i] > 0).mean()), 3), round(float(over[:, i].max()), 3)) for i in range(m.nu)],
                                           key=lambda x: -x[1])[:8]
print(json.dumps(res), flush=True)
if OUT: json.dump(res, open(OUT, "w"), indent=1)
