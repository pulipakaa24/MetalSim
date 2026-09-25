"""Cross-simulator transfer on the Newton engine: Isaac's PhysX-trained rsl_rl checkpoints played on the G1 flat
task exactly as metalsim/parity/side_by_side.py does on MuJoCo Warp (policy loading and joint-name mapping reused):
Isaac's default state at the origin, command (0.5, 0, 0) held, mean action, 400 control steps, 4 envs (observation
noise on, as in training). Reports x travelled and final pelvis z (mean over envs, and the range) per checkpoint
and engine setting; MuJoCo Warp at 2.5 ms runs the same protocol as the in-script control.

usage: python scripts/diagnostics/newton_transfer.py [--its 100,500,1000,1499] [--settings mjwarp,4:1.25,4:0.625]
       Newton fork recipe: 4:1.25:solver:color:relax=0.8:nolim"""
import sys, numpy as np, torch, warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP
from metalsim.learn.warp_policy import bump
from metalsim.interop import torch_bridge as tb
from metalsim.parity.side_by_side import load_policy
import mujoco

ITS = [100, 500, 1000, 1499]; SETTINGS = ["mjwarp", "4:1.25", "4:0.625"]
if "--its" in sys.argv: ITS = [int(x) for x in sys.argv[sys.argv.index("--its") + 1].split(",")]
if "--settings" in sys.argv: SETTINGS = sys.argv[sys.argv.index("--settings") + 1].split(",")
ISAAC_JOINTS = [str(s) for s in np.load("runs/parity/isaac/parity_out/play/isaac_it500/traj.npz")["joint_names"]]
N, STEPS = 4, 400


def make(setting):
    if setting == "mjwarp":
        return G1VelocityTask(N, terrain="flat", seed=0, physics_dt=0.0025)
    p = setting.split(":"); it, ms = p[0], p[1]; kw = {}
    for x in p[2:]:                        # Newton fork options: solver (PD drive), color, relax=R, nolim
        if x == "solver": kw["drive"] = "solver"
        elif x == "color": kw["joint_coloring"] = True
        elif x.startswith("relax="): kw["relaxation"] = float(x.split("=")[1])
        elif x == "nolim": kw["limit_margin"] = None
        elif x == "crb": kw["actuator_kw"] = {"joint_inertia": "crb"}
    return G1VelocityTask(N, terrain="flat", seed=0, engine="newton", newton_iterations=int(it), newton_dt=float(ms) * 1e-3, newton_kw=kw)


def play(task, ckpt):
    m = task.model; nj = task.nj
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    sd, pol_joints, src = load_policy(ckpt, ours)
    pol_joints = pol_joints or ISAAC_JOINTS
    hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.endswith("weight")), key=lambda s: int(s.split(".")[0]))[:-1])
    net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.actor.load_state_dict(sd); net.eval()
    perm_obs = torch.as_tensor(np.array([ours.index(n) for n in pol_joints]), device="mps")
    perm_act = torch.as_tensor(np.array([pol_joints.index(n) for n in ours]), device="mps")
    task.origins.assign(np.zeros((N, 3), np.float32)); task.reset_all()
    q0 = np.tile(m.key_qpos[0], (N, 1)).astype(np.float32)
    if task.engine == "newton":
        task.sim.d.qpos.assign(q0); task.sim.d.qvel.zero_(); task.sim._reset_mask.fill_(True); task.sim.launch_reset()
    else:
        task.sim.t.qpos.copy_(torch.as_tensor(q0)); task.sim.t.qvel.zero_(); v = task.sim.forward(); task.sim.after(v)
    task.sim.synchronize()
    task.cmd.assign(np.tile(np.array([0.5, 0.0, 0.0], np.float32), (N, 1))); task.resample.assign(np.zeros(N, bool))
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); t_obs = tb.mps_tensor(task.obs)
    for _ in range(STEPS):
        wp.launch(bump, dim=1, inputs=[step_idx], device="metal:0"); task.launch_obs(step_idx)
        vs = task.sim._signal(); task.sim.after(vs)
        with torch.no_grad():
            o = t_obs.clone()
            if src == "rsl_rl":
                o = torch.cat([o[:, :12], o[:, 12:12 + nj][:, perm_obs], o[:, 12 + nj:12 + 2 * nj][:, perm_obs], o[:, 12 + 2 * nj:][:, perm_obs]], 1)
            act = net.actor(o)
            if src == "rsl_rl": act = act[:, perm_act]
        task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch)
        task.sim.step()
    task.sim.synchronize()
    q = task.sim.d.qpos.numpy()
    return q[:, 0], q[:, 2]


def main():
    import subprocess
    print("install:", subprocess.run([sys.executable, "scripts/diagnostics/newton_stamp.py"], capture_output=True, text=True).stdout.strip())
    print("| checkpoint | " + " | ".join(f"{s} x [m] / pelvis z [m]" for s in SETTINGS) + " |")
    print("|---|" + "---|" * len(SETTINGS), flush=True)
    tasks = {s: make(s) for s in SETTINGS}
    for it in ITS:
        row = [f"Isaac it {it}"]
        for s in SETTINGS:
            x, z = play(tasks[s], f"runs/parity/isaac/ckpt_out/model_{it}.pt")
            row.append(f"{x.mean():.2f} ({x.min():.2f}-{x.max():.2f}) / {z.mean():.3f} ({z.min():.2f}-{z.max():.2f})")
        print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
