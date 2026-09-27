"""Over-travel probe (speculative contacts, 2026-09-27): Isaac's PhysX-trained checkpoints played with the transfer
protocol of scripts/diagnostics/newton_transfer.py (Isaac's default state at the origin, command (0.5, 0, 0), mean
action, 400 control steps, observation noise on, task seed 0) while logging, for every physics substep, per env and
foot, what the foot-ground contacts do:

  sub[step, substep, env, foot, k], k =
    0 Fn      normal force, all foot-ground contacts            1 Ft   sum of |friction force| over those contacts
    2 Fn_s    normal force of speculative contacts (pos > 0)    3 Ft_s |friction| of speculative contacts
    4 Fn_u    normal force of speculative contacts that do NOT predict a crossing in the step (pos + h v_n >= 0):
              force the PhysX rule would not apply (PhysX: bias -sep/dt only acts on the approach that would cross)
    5 Ft_u    |friction| of those contacts
    6 n       foot-ground contacts with rows                    7 n_s  of which speculative
    8 n_sF    speculative contacts with Fn > 1e-3 N             9 dmin deepest signed distance (m), 1 = none
   10 Ftx     world-x friction force on the foot (+ = forward)  11 Ftx_s the same, speculative contacts only
  subw[step, substep, env, j], j = 0 nefc, 1 Newton iterations

and per control step (host): actions (Isaac joint order), base x / vx, foot COM horizontal speed (Isaac's
body_lin_vel_w), foot COM height, joint positions (Isaac order) and root position, the task ContactSensor's feet |net normal force| (last substep) and its contact flags.

    python scripts/diagnostics/speculative/overtravel_probe.py PRESET OUT.npz [--its 500,1000,1499] [--envs 4] [--steps 400]
MuJoCo Warp from PYTHONPATH (the prototype needs MJW_SPECULATIVE_GAP=1; MJW_SPEC_FRICTION / MJW_SPEC_NULL select variants).
"""
import argparse, os, sys, time
import numpy as np, torch, warp as wp, mujoco
wp.config.quiet = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP, bump
from metalsim.interop import torch_bridge as tb
from metalsim.parity.side_by_side import load_policy
from metalsim.physics import contact_tuning
import mujoco_warp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

K = 12


@wp.kernel
def _foot_contacts(ctr: wp.array(dtype=int), opt_cone: int, h: float,
                   frame: wp.array(dtype=wp.mat33), friction: wp.array(dtype=vec5), dim: wp.array(dtype=int),
                   efc_address: wp.array2d(dtype=int), adhesion: wp.array(dtype=float), geom: wp.array(dtype=wp.vec2i),
                   worldid: wp.array(dtype=int), dist: wp.array(dtype=float), includemargin: wp.array(dtype=float),
                   efc_force: wp.array2d(dtype=float), efc_vel: wp.array2d(dtype=float), njmax: int, nacon: wp.array(dtype=int),
                   geom_foot: wp.array(dtype=int), ground: int, out: wp.array3d(dtype=float)):
    c = wp.tid()
    if c >= nacon[0]:
        return
    g = geom[c]
    foot = int(-1); sgn = float(1.0)
    if g[0] == ground and geom_foot[g[1]] >= 0:
        foot = geom_foot[g[1]]
    elif g[1] == ground and geom_foot[g[0]] >= 0:
        foot = geom_foot[g[0]]; sgn = -1.0
    if foot < 0:
        return
    e0 = efc_address[c, 0]
    if e0 < 0:
        return
    w = worldid[c]; s = ctr[0]
    f6 = contact_force_fn(opt_cone, frame, friction, dim, efc_address, adhesion, efc_force, njmax, nacon, w, c, False)
    fr = frame[c]
    fn = f6[0]; ft = wp.sqrt(f6[1] * f6[1] + f6[2] * f6[2])
    ftx = sgn * (fr[1, 0] * f6[1] + fr[2, 0] * f6[2])
    pos = dist[c] - includemargin[c]
    vn = efc_vel[w, e0]
    wp.atomic_add(out, s, w, foot * K + 0, fn); wp.atomic_add(out, s, w, foot * K + 1, ft)
    wp.atomic_add(out, s, w, foot * K + 6, 1.0); wp.atomic_add(out, s, w, foot * K + 10, ftx)
    wp.atomic_min(out, s, w, foot * K + 9, dist[c] - 1.0)   # zero-initialised buffer: stored as dist - 1
    if pos > 0.0:
        wp.atomic_add(out, s, w, foot * K + 2, fn); wp.atomic_add(out, s, w, foot * K + 3, ft)
        wp.atomic_add(out, s, w, foot * K + 7, 1.0); wp.atomic_add(out, s, w, foot * K + 11, ftx)
        if fn > 1.0e-3:
            wp.atomic_add(out, s, w, foot * K + 8, 1.0)
        if pos + h * vn >= 0.0:
            wp.atomic_add(out, s, w, foot * K + 4, fn); wp.atomic_add(out, s, w, foot * K + 5, ft)


@wp.kernel
def _world_stats(ctr: wp.array(dtype=int), nefc: wp.array(dtype=int), niter: wp.array(dtype=int), out: wp.array3d(dtype=float)):
    w = wp.tid(); s = ctr[0]
    out[s, w, 0] = float(nefc[w]); out[s, w, 1] = float(niter[w])


@wp.kernel
def _bump(ctr: wp.array(dtype=int)):
    ctr[0] = ctr[0] + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preset"); ap.add_argument("out")
    ap.add_argument("--its", default="500,1000,1499"); ap.add_argument("--envs", type=int, default=4); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); N = a.envs
    isaac_joints = [str(s) for s in np.load(os.path.join(ROOT, "runs/parity/isaac/parity_out/play/isaac_it500/traj.npz"))["joint_names"]]
    with contact_tuning.g1_model_tuning(a.preset):
        task = G1VelocityTask(N, terrain="flat", seed=a.seed, physics_dt=0.0025)
    m = task.model; sim = task.sim; d = sim.d; dev = sim.device; dec = task.decimation; h = float(m.opt.timestep)
    fb = [int(task.foot_body[0]), int(task.foot_body[1])]; fr = [int(task.foot_root[0]), int(task.foot_root[1])]
    gf = np.full(m.ngeom, -1, np.int32)
    for f in range(2):
        gf[m.geom_bodyid == fb[f]] = f
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    geom_foot = wp.array(gf, dtype=int, device=dev)
    ctr = wp.zeros(1, dtype=int, device=dev)
    sub3 = wp.zeros((dec, N, 2 * K), dtype=float, device=dev); subw = wp.zeros((dec, N, 2), dtype=float, device=dev)
    return run(a, task, m, sim, d, dev, dec, h, fb, fr, geom_foot, ground, ctr, sub3, subw, isaac_joints)


def run(a, task, m, sim, d, dev, dec, h, fb, fr, geom_foot, ground, ctr, sub3, subw, isaac_joints):
    N = a.envs
    def hook():
        wp.launch(_foot_contacts, dim=d.naconmax, device=dev, inputs=[
            ctr, int(sim.m.opt.cone), h, d.contact.frame, d.contact.friction, d.contact.dim, d.contact.efc_address, d.contact.adhesion,
            d.contact.geom, d.contact.worldid, d.contact.dist, d.contact.includemargin, d.efc.force, d.efc.vel, d.njmax, d.nacon,
            geom_foot, ground, sub3])
        wp.launch(_world_stats, dim=N, device=dev, inputs=[ctr, d.nefc, d.solver_niter, subw])
        wp.launch(_bump, dim=1, device=dev, inputs=[ctr])
    sim.add_substep_hook(hook)
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    cs = task.contact; nj = task.nj
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in isaac_joints])
    res = {"preset": a.preset, "mjw": os.path.dirname(mujoco_warp.__file__), "env": {k: v for k, v in os.environ.items() if k.startswith("MJW_")}}
    for it in [int(x) for x in a.its.split(",")]:
        t0 = time.time()
        sd, pol_joints, src = load_policy(os.path.join(ROOT, f"runs/parity/isaac/ckpt_out/model_{it}.pt"), ours)
        pol_joints = pol_joints or isaac_joints
        hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.endswith("weight")), key=lambda s: int(s.split(".")[0]))[:-1])
        net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.actor.load_state_dict(sd); net.eval()
        perm_obs = torch.as_tensor(np.array([ours.index(n) for n in pol_joints]), device="mps")
        perm_act = torch.as_tensor(np.array([pol_joints.index(n) for n in ours]), device="mps")
        task.origins.assign(np.zeros((N, 3), np.float32)); task.reset_all()
        q0 = np.tile(m.key_qpos[0], (N, 1)).astype(np.float32)
        sim.t.qpos.copy_(torch.as_tensor(q0)); sim.t.qvel.zero_(); v = sim.forward(); sim.after(v); sim.synchronize()
        task.cmd.assign(np.tile(np.array([0.5, 0.0, 0.0], np.float32), (N, 1))); task.resample.assign(np.zeros(N, bool))
        step_idx = wp.zeros(1, dtype=int, device=dev); t_obs = tb.mps_tensor(task.obs)
        S = a.steps
        rec = {k: [] for k in ("sub", "subw", "act", "x", "vx", "slide", "footz", "sens_f", "sens_flag", "vz_foot", "jpos", "root")}
        for _ in range(S):
            wp.launch(bump, dim=1, inputs=[step_idx], device=dev); task.launch_obs(step_idx)
            vs = sim._signal(); sim.after(vs)
            with torch.no_grad():
                o = t_obs.clone()
                if src == "rsl_rl":
                    o = torch.cat([o[:, :12], o[:, 12:12 + nj][:, perm_obs], o[:, 12 + nj:12 + 2 * nj][:, perm_obs], o[:, 12 + 2 * nj:][:, perm_obs]], 1)
                act = net.actor(o)
                act_isaac = act.cpu().numpy().astype(np.float32)
                if src == "rsl_rl": act = act[:, perm_act]
            task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch)
            ctr.zero_(); sub3.zero_(); subw.zero_()
            sim.step(); sim.synchronize()
            x = sub3.numpy().reshape(dec, N, 2, K); x[..., 9] += 1.0; x[..., 9][x[..., 6] == 0] = np.nan
            rec["sub"].append(x.astype(np.float32)); rec["subw"].append(subw.numpy().astype(np.float32)); rec["act"].append(act_isaac)
            q = d.qpos.numpy(); qv = d.qvel.numpy(); rec["x"].append(q[:, 0].copy()); rec["jpos"].append(q[:, qadr].astype(np.float32)); rec["root"].append(q[:, :3].astype(np.float32)); rec["vx"].append(qv[:, 0].copy())
            cvel = d.cvel.numpy(); xipos = d.xipos.numpy(); stc = d.subtree_com.numpy()
            sp = np.zeros((N, 2)); fz = np.zeros((N, 2)); vzf = np.zeros((N, 2))
            for f in range(2):
                wv = cvel[:, fb[f], :3]; vl = cvel[:, fb[f], 3:]
                vc = vl + np.cross(wv, xipos[:, fb[f]] - stc[:, fr[f]])
                sp[:, f] = np.linalg.norm(vc[:, :2], axis=1); fz[:, f] = xipos[:, fb[f], 2]; vzf[:, f] = vc[:, 2]
            rec["slide"].append(sp); rec["footz"].append(fz); rec["vz_foot"].append(vzf)
            net_f = cs._net.numpy()[:, :2]; rec["sens_f"].append(np.linalg.norm(net_f, axis=-1)); rec["sens_flag"].append(cs._cur_con.numpy()[:, :2] > 0)
        for k, v in rec.items():
            res[f"it{it}_{k}"] = np.array(v)
        xs = np.array(rec["x"])[-1]
        print(f"it {it}: x {xs.mean():.3f} ({xs.min():.2f}-{xs.max():.2f}) +- {xs.std(ddof=1) / np.sqrt(N) if N > 1 else 0:.3f} (SE), {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(a.out, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)}, meta=np.array(repr({k: v for k, v in res.items() if not isinstance(v, np.ndarray)})))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
