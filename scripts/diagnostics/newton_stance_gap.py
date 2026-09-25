"""Why does Isaac's policy over-travel on Newton? Plays an Isaac checkpoint with the transfer protocol (default state,
command (0.5, 0, 0), mean action, 400 steps, 4 envs) and measures, per engine setting, over the last 300 steps:
  stance slide : horizontal speed of a foot while it is in contact (touch > 1 N) [m/s]
  anchor gap   : distance between the stance foot and where forward kinematics of the reported joint angles puts it
                 (Newton only; 0 by construction in MuJoCo Warp) [mm]
  stride       : forward displacement of a foot between consecutive touch-downs [m]; cadence: touch-downs per s
  base speed   : mean forward base speed [m/s]

usage: python scripts/diagnostics/newton_stance_gap.py [--it 1000] [--settings mjwarp,4:1.25,...]"""
import sys, numpy as np, torch, warp as wp, newton, mujoco
wp.config.quiet = True
sys.path.insert(0, "scripts/diagnostics")
import newton_transfer as T

IT = int(sys.argv[sys.argv.index("--it") + 1]) if "--it" in sys.argv else 1000
SET = sys.argv[sys.argv.index("--settings") + 1].split(",") if "--settings" in sys.argv else ["mjwarp", "4:1.25", "4:0.625"]


def run(setting):
    task = T.make(setting); m = task.model; nj = task.nj; N = T.N
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    sd, pol_joints, src = T.load_policy(f"runs/parity/isaac/ckpt_out/model_{IT}.pt", ours); pol_joints = pol_joints or T.ISAAC_JOINTS
    hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.endswith("weight")), key=lambda s: int(s.split(".")[0]))[:-1])
    net = T.ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.actor.load_state_dict(sd); net.eval()
    po = torch.as_tensor(np.array([ours.index(n) for n in pol_joints]), device="mps"); pa = torch.as_tensor(np.array([pol_joints.index(n) for n in ours]), device="mps")
    task.origins.assign(np.zeros((N, 3), np.float32)); task.reset_all()
    q0 = np.tile(m.key_qpos[0], (N, 1)).astype(np.float32)
    newt = task.engine == "newton"
    if newt:
        task.sim.d.qpos.assign(q0); task.sim.d.qvel.zero_(); task.sim._reset_mask.fill_(True); task.sim.launch_reset()
        M = task.sim.model; fk = M.state(); feet_nb = [int(task.sim.foot_nb[0]), int(task.sim.foot_nb[1])]
    else:
        task.sim.t.qpos.copy_(torch.as_tensor(q0)); task.sim.t.qvel.zero_(); v = task.sim.forward(); task.sim.after(v)
        fb = [int(task.foot_body[0]), int(task.foot_body[1])]
    task.sim.synchronize()
    task.cmd.assign(np.tile(np.array([0.5, 0.0, 0.0], np.float32), (N, 1))); task.resample.assign(np.zeros(N, bool))
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); t_obs = T.tb.mps_tensor(task.obs)
    ta = task.touch_adr; pos, con, gap, base = [], [], [], []
    for t in range(T.STEPS):
        wp.launch(T.bump, dim=1, inputs=[step_idx], device="metal:0"); task.launch_obs(step_idx); vs = task.sim._signal(); task.sim.after(vs)
        with torch.no_grad():
            o = t_obs.clone()
            if src == "rsl_rl": o = torch.cat([o[:, :12], o[:, 12:12 + nj][:, po], o[:, 12 + nj:12 + 2 * nj][:, po], o[:, 12 + 2 * nj:][:, po]], 1)
            act = net.actor(o)
            if src == "rsl_rl": act = act[:, pa]
        task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch); task.sim.step(); task.sim.synchronize()
        sdat = task.sim.d.sensordata.numpy(); con.append(np.stack([sdat[:, ta[0]], sdat[:, ta[1]]], 1) > 1.0)
        base.append(task.sim.d.qvel.numpy()[:, 0].copy())
        if newt:
            bq = task.sim.s0.body_q.numpy().reshape(N, task.sim.nb, 7)
            newton.eval_fk(M, task.sim.joint_q, task.sim.joint_qd, fk); bf = fk.body_q.numpy().reshape(N, task.sim.nb, 7)
            pos.append(bq[:, feet_nb, :3]); gap.append(np.linalg.norm(bq[:, feet_nb, :3] - bf[:, feet_nb, :3], axis=-1))
        else:
            d = task.sim.d; pos.append(d.xpos.numpy()[:, fb, :].copy())
    P = np.stack(pos)[100:]; Cn = np.stack(con)[100:]; B = np.stack(base)[100:]
    vel = np.linalg.norm(np.diff(P[..., :2], axis=0), axis=-1) / 0.02; st = Cn[1:] & Cn[:-1]
    slide = vel[st].mean() if st.any() else float("nan")
    strides, tds = [], 0
    for e in range(N):
        for f in range(2):
            c = Cn[:, e, f]; down = np.nonzero(c[1:] & ~c[:-1])[0] + 1; tds += len(down)
            xs = P[down, e, f, 0]; strides += list(np.diff(xs))
    G = np.stack(gap)[100:] * 1e3 if gap else None
    g = f"{np.median(G[Cn]):.1f} / {np.percentile(G[Cn], 95):.1f}" if G is not None else "0 (reduced coordinates)"
    return slide, g, (np.mean(strides) if strides else float("nan")), tds / (N * 2 * len(P) * 0.02), B.mean()


print(f"Isaac checkpoint {IT}, command (0.5, 0, 0), 4 envs, steps 100-400")
print("| setting | stance slide [m/s] | stance anchor gap p50 / p95 [mm] | stride [m] | cadence [touch-downs/s per foot] | base speed [m/s] |")
print("|---|---|---|---|---|---|", flush=True)
for s in SET:
    sl, g, stride, cad, v = run(s)
    print(f"| {s} | {sl:.3f} | {g} | {stride:.3f} | {cad:.2f} | {v:.3f} |", flush=True)
