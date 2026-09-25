"""Push-off diagnostic for the Newton stride question: Isaac's checkpoint played with the transfer protocol (default
state, command (0.5, 0, 0), mean action, 400 steps, 4 envs) on each engine setting. Only the ground acts horizontally on
the robot, so the forward ground-reaction force is M a_com_x (whole-body COM, finite differences at 50 Hz). Over steps
100-400: propulsive (sum of positive forward impulse) and braking (negative) impulse per foot touch-down [N s], vertical
impulse per touch-down minus the weight share [N s], peak foot contact force [N], stride [m], cadence.

usage: python scripts/diagnostics/g1_pushoff.py [--it 1000] [--settings mjwarp,4:1.25,4:0.625]"""
import sys, numpy as np, torch, warp as wp, mujoco
wp.config.quiet = True
sys.path.insert(0, "scripts/diagnostics")
import newton_transfer as T

IT = int(sys.argv[sys.argv.index("--it") + 1]) if "--it" in sys.argv else 1000
SET = sys.argv[sys.argv.index("--settings") + 1].split(",") if "--settings" in sys.argv else ["mjwarp", "4:1.25", "4:0.625"]
DT = 0.02


def run(setting):
    task = T.make(setting); m = task.model; nj = task.nj; N = T.N
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    sd, pol_joints, src = T.load_policy(f"runs/parity/isaac/ckpt_out/model_{IT}.pt", ours); pol_joints = pol_joints or T.ISAAC_JOINTS
    hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.endswith("weight")), key=lambda s: int(s.split(".")[0]))[:-1])
    net = T.ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.actor.load_state_dict(sd); net.eval()
    po = torch.as_tensor(np.array([ours.index(n) for n in pol_joints]), device="mps"); pa = torch.as_tensor(np.array([pol_joints.index(n) for n in ours]), device="mps")
    task.origins.assign(np.zeros((N, 3), np.float32)); task.reset_all()
    q0 = np.tile(m.key_qpos[0], (N, 1)).astype(np.float32); newt = task.engine == "newton"
    if newt:
        task.sim.d.qpos.assign(q0); task.sim.d.qvel.zero_(); task.sim._reset_mask.fill_(True); task.sim.launch_reset()
        M = task.sim.model; bm = M.body_mass.numpy().reshape(N, -1); bc = M.body_com.numpy().reshape(N, -1, 3); feet = [int(task.sim.foot_nb[0]), int(task.sim.foot_nb[1])]
    else:
        task.sim.t.qpos.copy_(torch.as_tensor(q0)); task.sim.t.qvel.zero_(); v = task.sim.forward(); task.sim.after(v); fb = [int(task.foot_body[0]), int(task.foot_body[1])]
    task.sim.synchronize()
    mass = float(m.body_subtreemass[1])
    task.cmd.assign(np.tile(np.array([0.5, 0.0, 0.0], np.float32), (N, 1))); task.resample.assign(np.zeros(N, bool))
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); t_obs = T.tb.mps_tensor(task.obs); ta = task.touch_adr
    com, con, frc, fx = [], [], [], []
    for t in range(T.STEPS):
        wp.launch(T.bump, dim=1, inputs=[step_idx], device="metal:0"); task.launch_obs(step_idx); vs = task.sim._signal(); task.sim.after(vs)
        with torch.no_grad():
            o = t_obs.clone()
            if src == "rsl_rl": o = torch.cat([o[:, :12], o[:, 12:12 + nj][:, po], o[:, 12 + nj:12 + 2 * nj][:, po], o[:, 12 + 2 * nj:][:, po]], 1)
            act = net.actor(o)
            if src == "rsl_rl": act = act[:, pa]
        task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch); task.sim.step(); task.sim.synchronize()
        s = task.sim.d.sensordata.numpy(); f = np.stack([s[:, ta[0]], s[:, ta[1]]], 1); frc.append(f); con.append(f > 1.0)
        if newt:
            bq = task.sim.s0.body_q.numpy().reshape(N, -1, 7)
            R = np.stack([[_rot(bq[e, b, 3:7]) for b in range(bq.shape[1])] for e in range(N)])
            cw = bq[..., :3] + np.einsum("ebij,ebj->ebi", R, bc)
            com.append((bm[..., None] * cw).sum(1) / bm.sum(1)[:, None]); fx.append(bq[:, feet, 0])
        else:
            com.append(task.sim.d.subtree_com.numpy()[:, 1].copy()); fx.append(task.sim.d.xpos.numpy()[:, fb, 0])
    C = np.stack(com); Cn = np.stack(con)[100:]; Fz = np.stack(frc)[100:]; X = np.stack(fx)[100:]
    v = np.gradient(C, DT, axis=0); dp = mass * np.diff(v, axis=0)[99:]            # momentum change per control step [N s]
    tds = sum(len(np.nonzero(Cn[1:, e, f] & ~Cn[:-1, e, f])[0]) for e in range(N) for f in range(2))
    prop = dp[..., 0].clip(min=0).sum() / tds; brake = dp[..., 0].clip(max=0).sum() / tds
    vert = (dp[..., 2] + mass * 9.81 * DT).sum() / tds
    strides = []
    for e in range(N):
        for f in range(2):
            c = Cn[:, e, f]; down = np.nonzero(c[1:] & ~c[:-1])[0] + 1; strides += list(np.diff(X[down, e, f]))
    return prop, brake, vert, float(Fz.max()), float(np.percentile(Fz[Cn], 95)), np.mean(strides), tds / (N * 2 * len(Cn) * DT)


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


if __name__ == "__main__":
    import subprocess
    print("install:", subprocess.run([sys.executable, "scripts/diagnostics/newton_stamp.py"], capture_output=True, text=True).stdout.strip())
    print(f"Isaac checkpoint {IT}, command (0.5, 0, 0), 4 envs, steps 100-400; impulses per foot touch-down")
    print("| setting | propulsive [N s] | braking [N s] | vertical [N s] | peak foot force [N] | p95 stance force [N] | stride [m] | cadence [/s per foot] |")
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for s in SET:
        r = run(s)
        print(f"| {s} | {r[0]:.2f} | {r[1]:.2f} | {r[2]:.1f} | {r[3]:.0f} | {r[4]:.0f} | {r[5]:.3f} | {r[6]:.2f} |", flush=True)
