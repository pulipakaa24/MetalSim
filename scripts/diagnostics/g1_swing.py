"""Swing-leg diagnostic for the Newton stride question. Isaac's checkpoint played with the transfer protocol (default
state, command (0.5, 0, 0), mean action, 400 steps, 4 envs, steps 100-400) on each engine setting. Per swing phase
(foot lift-off to touch-down, touch > 1 N): duration, peak foot clearance above its stance height, forward foot velocity
at touch-down, step length (swing foot minus stance foot along x at touch-down), and hip-pitch / knee angle trajectories
of the swing leg resampled to 11 points over the normalized swing phase (mean over swings), plus the RMS difference of
those mean trajectories between the first setting and each other setting.

If the joint trajectories agree but the swing is longer/faster, the difference is in joint-to-foot kinematics under load;
if the trajectories differ, it is the drive/dynamics.

usage: python scripts/diagnostics/g1_swing.py [--it 1000] [--settings mjwarp,4:0.625]"""
import sys, numpy as np, torch, warp as wp, mujoco
wp.config.quiet = True
sys.path.insert(0, "scripts/diagnostics")
import newton_transfer as T

IT = int(sys.argv[sys.argv.index("--it") + 1]) if "--it" in sys.argv else 1000
SET = sys.argv[sys.argv.index("--settings") + 1].split(",") if "--settings" in sys.argv else ["mjwarp", "4:0.625"]
DT = 0.02; PH = np.linspace(0, 1, 11)


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
        task.sim.d.qpos.assign(q0); task.sim.d.qvel.zero_(); task.sim._reset_mask.fill_(True); task.sim.launch_reset(); feet = [int(task.sim.foot_nb[0]), int(task.sim.foot_nb[1])]
    else:
        task.sim.t.qpos.copy_(torch.as_tensor(q0)); task.sim.t.qvel.zero_(); v = task.sim.forward(); task.sim.after(v); fb = [int(task.foot_body[0]), int(task.foot_body[1])]
    task.sim.synchronize()
    task.cmd.assign(np.tile(np.array([0.5, 0.0, 0.0], np.float32), (N, 1))); task.resample.assign(np.zeros(N, bool))
    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    J = {side: (act_names.index(f"{side}_hip_pitch_joint"), act_names.index(f"{side}_knee_joint")) for side in ("left", "right")}
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); t_obs = T.tb.mps_tensor(task.obs); ta = task.touch_adr
    P, C, Q = [], [], []
    for t in range(T.STEPS):
        wp.launch(T.bump, dim=1, inputs=[step_idx], device="metal:0"); task.launch_obs(step_idx); vs = task.sim._signal(); task.sim.after(vs)
        with torch.no_grad():
            o = t_obs.clone()
            if src == "rsl_rl": o = torch.cat([o[:, :12], o[:, 12:12 + nj][:, po], o[:, 12 + nj:12 + 2 * nj][:, po], o[:, 12 + 2 * nj:][:, po]], 1)
            act = net.actor(o)
            if src == "rsl_rl": act = act[:, pa]
        task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch); task.sim.step(); task.sim.synchronize()
        s = task.sim.d.sensordata.numpy(); C.append(np.stack([s[:, ta[0]], s[:, ta[1]]], 1) > 1.0)
        Q.append(task.sim.d.qpos.numpy()[:, 7:].copy())
        P.append(task.sim.s0.body_q.numpy().reshape(N, -1, 7)[:, feet, :3] if newt else task.sim.d.xpos.numpy()[:, fb, :].copy())
    P = np.stack(P)[100:]; C = np.stack(C)[100:]; Q = np.stack(Q)[100:]
    V = np.gradient(P, DT, axis=0)
    dur, clr, vtd, steplen, hip, knee = [], [], [], [], [], []
    for e in range(N):
        for f, side in enumerate(("left", "right")):
            c = C[:, e, f]; lift = np.nonzero(~c[1:] & c[:-1])[0] + 1; down = np.nonzero(c[1:] & ~c[:-1])[0] + 1
            for a in lift:
                b = down[down > a]
                if not len(b): continue
                b = b[0]
                if b - a < 2: continue
                dur.append((b - a) * DT)
                clr.append(P[a:b, e, f, 2].max() - P[a - 1, e, f, 2])
                vtd.append(V[b, e, f, 0]); steplen.append(P[b, e, f, 0] - P[b, e, 1 - f, 0])
                ph = np.linspace(0, 1, b - a + 1); jh, jk = J[side]
                hip.append(np.interp(PH, ph, Q[a:b + 1, e, jh])); knee.append(np.interp(PH, ph, Q[a:b + 1, e, jk]))
    return dict(n=len(dur), dur=np.mean(dur), clr=np.mean(clr), vtd=np.mean(vtd), step=np.mean(steplen),
                hip=np.mean(hip, 0), knee=np.mean(knee, 0))


if __name__ == "__main__":
    import subprocess
    print("install:", subprocess.run([sys.executable, "scripts/diagnostics/newton_stamp.py"], capture_output=True, text=True).stdout.strip())
    R = {s: run(s) for s in SET}
    print(f"Isaac checkpoint {IT}, command (0.5, 0, 0), 4 envs, steps 100-400; means over swing phases")
    print("| setting | swings | swing duration [s] | peak clearance [m] | foot fwd velocity at touch-down [m/s] | step length vs stance foot [m] |")
    print("|---|---|---|---|---|---|")
    for s, r in R.items():
        print(f"| {s} | {r['n']} | {r['dur']:.3f} | {r['clr']:.3f} | {r['vtd']:.3f} | {r['step']:.3f} |")
    print("\nmean swing-leg joint trajectories over the normalized swing phase (0, 0.1, ..., 1) [rad]")
    for s, r in R.items():
        print(f"  {s:10s} hip pitch " + " ".join(f"{x:+.3f}" for x in r["hip"]))
        print(f"  {s:10s} knee      " + " ".join(f"{x:+.3f}" for x in r["knee"]))
    ref = R[SET[0]]
    for s in SET[1:]:
        r = R[s]
        print(f"RMS difference vs {SET[0]}: {s}: hip pitch {np.sqrt(((r['hip'] - ref['hip']) ** 2).mean()):.3f} rad (range {np.ptp(ref['hip']):.3f} vs {np.ptp(r['hip']):.3f}), "
              f"knee {np.sqrt(((r['knee'] - ref['knee']) ** 2).mean()):.3f} rad (range {np.ptp(ref['knee']):.3f} vs {np.ptp(r['knee']):.3f})")
