"""Cross-simulator transfer on rough terrain, MetalSim side: the protocol of isaac_side/play_policy.py
(--task Isaac-Velocity-Rough-G1-v0) in MuJoCo Warp. Isaac's rough terrain for env seed --seed (the training run's),
4 envs, env i at terrain row --level in Isaac's column floor(i / (4 / 20)) = 0, 4, 9, 14 in float32, default joint pose at the
cell origin with zero yaw, command (0.5, 0, 0), no observation noise, no termination, mean action, 400 control steps.
Reports x travelled and final pelvis z (both relative to the env origin, as play_policy.py records them).

With --isaac_play <dir> (the play_rough tree fetched from the VM) it also checks the terrain against Isaac's: the
terrain origin table and the step-0 height scan (both sides start from the same state, so the scans must agree).

    python scripts/diagnostics/g1_rough_transfer.py --isaac_play runs/parity/isaac/play_rough --level 3 \
        --ckpt isaac_it1000=runs/parity/isaac/rsl_rl_g1_rough/<run>/model_1000.pt --ckpt ours_it1000=runs/policies/g1_rough_ppowarp_fixed_it1000.pt
"""
import argparse, glob, json, os
import numpy as np, torch, mujoco, warp as wp

from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.terrain import scan_perm
from metalsim.learn.warp_policy import bump
from metalsim.interop import torch_bridge as tb


def quat_mat(q):          # (n, 4) wxyz -> (n, 3, 3)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def ground_fn(hf):
    """Terrain height at world (x, y) by bilinear interpolation of the heightfield grid (Isaac's heights exactly at
    grid points, PARITY 1.2), for pelvis height above the local ground on both simulators' trajectories."""
    H = hf["H"]; res = hf["res"]; nrow, ncol = H.shape; sx = (ncol - 1) * res / 2; sy = (nrow - 1) * res / 2
    def g(x, y):
        fx = np.clip((np.asarray(x) + sx) / res, 0, ncol - 1.001); fy = np.clip((np.asarray(y) + sy) / res, 0, nrow - 1.001)
        i, j = np.floor(fy).astype(int), np.floor(fx).astype(int); u, v = fy - i, fx - j
        return (1 - u) * (1 - v) * H[i, j] + (1 - u) * v * H[i, j + 1] + u * (1 - v) * H[i + 1, j] + u * v * H[i + 1, j + 1]
    return g


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:
        sd, src, order = {k[6:]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}, "rsl_rl", "xy"
    else:          # MetalSim checkpoints before 2026-09-25 carry no scan_ordering and were trained on the "ij" ray order
        sd, src, order = {k[6:]: v for k, v in ck["net"].items() if k.startswith("actor.")}, "metalsim", ck.get("scan_ordering") or "ij"
    idx = sorted({int(k.split(".")[0]) for k in sd})
    layers = []
    for j, i in enumerate(idx):
        w = sd[f"{i}.weight"]; lin = torch.nn.Linear(w.shape[1], w.shape[0]); lin.weight.data.copy_(w); lin.bias.data.copy_(sd[f"{i}.bias"])
        layers.append(lin)
        if j < len(idx) - 1: layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to("mps").eval(), src, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="label=path (rsl_rl model_*.pt or a MetalSim checkpoint)")
    ap.add_argument("--level", type=int, nargs="+", default=[3]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--physics_dt", type=float, default=0.0025)
    ap.add_argument("--isaac_play", default=None, help="play_rough tree from the VM (L<level>/<label>/meta.json, traj.npz)")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    ap.add_argument("--terrain_collision", default="boxes_local", help="collision surface (metalsim.learn.terrain."
                    "TERRAIN_COLLISION): boxes_local (default since 2026-09-25), hfield (0.1 m, before), boxes, meshes, "
                    "hfield_fine[:res], boxes_fine[:res]")
    ap.add_argument("--reps", type=int, default=1, help="envs per cell type: rep 0 is the protocol start, reps 1.. start "
                    "at a seeded uniform +-jitter xy offset (fall rates with more samples; Isaac comparison only for reps 1)")
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--traj_dir", default=None, help="save root position/quaternion and joint positions every step here")
    ap.add_argument("--scan_surface", default="exact", help="height scan: exact (default since 2026-09-25) or grid (0.1 m "
                    "interpolation, before)")
    a = ap.parse_args(); wp.config.quiet = True
    n = 4 * a.reps
    coll, _, fine_res = a.terrain_collision.partition(":")
    if fine_res:                                  # hfield_fine:<res>
        import metalsim.learn.terrain as T
        _orig = T.isaac_rough_terrain; T.isaac_rough_terrain = lambda **kw: _orig(fine_res=float(fine_res), **kw)
    task = G1VelocityTask(n, terrain="rough", seed=a.seed, physics_dt=a.physics_dt, scan_ordering="xy",   # Isaac's ray order
                          terrain_collision=coll, scan_surface=a.scan_surface)
    m = task.model
    our_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    nj = task.nj
    tab = task.hfield["origin_table"]()                       # (rows, cols, 3)
    ground = ground_fn(task.hfield)
    cols = task.col.numpy()
    if a.reps > 1:        # the protocol's four columns (Isaac's floor(i / (4 / 20)) in float32 for 4 envs), repeated
        cols = np.tile(np.array([0, 4, 9, 14]), a.reps)
        jit = np.random.default_rng(1234).uniform(-a.jitter, a.jitter, (n, 2)).astype(np.float32); jit[:4] = 0.0
    isaac_joints = None
    if a.isaac_play:
        tr = sorted(glob.glob(os.path.join(a.isaac_play, "L*", "*", "traj.npz")))
        if tr: isaac_joints = [str(s) for s in np.load(tr[0])["joint_names"]]
        metas = sorted(glob.glob(os.path.join(a.isaac_play, "L*", "*", "meta.json")))
        if metas:
            mt = json.load(open(metas[0]))
            it = np.array(mt["terrain_origins"], np.float32)
            print(f"[terrain] origin table vs Isaac ({metas[0]}): max |diff| {np.abs(it - tab).max():.2e} m over {it.shape}; "
                  f"Isaac terrain_types {mt['terrain_types']} ours {cols.tolist()}", flush=True)
    default_q = torch.as_tensor(m.key_qpos[0].astype(np.float32), device="mps")
    t_scan = tb.mps_tensor(task.height_scan)
    step_idx = wp.zeros(1, dtype=int, device="metal:0")
    cmd = torch.tensor([[0.5, 0.0, 0.0]], device="mps").expand(n, 3)
    results = []
    for lv in a.level:
        origins = tab[lv, cols].astype(np.float32)
        if a.reps > 1:
            origins = origins.copy(); origins[:, :2] += jit    # start offset (the origin z stays the cell's)
        for spec in a.ckpt:
            label, path = spec.split("=", 1)
            actor, src, order = load_actor(path)
            sperm = torch.as_tensor(scan_perm("xy", order), device="mps")     # Isaac-ordered scan -> the policy's order
            if src == "rsl_rl":
                assert isaac_joints is not None, "rsl_rl checkpoint needs Isaac's joint order (--isaac_play)"
                perm_obs = torch.as_tensor([our_joints.index(j) for j in isaac_joints], device="mps")     # policy joint k <- our joint
                perm_act = torch.as_tensor([isaac_joints.index(j) for j in our_joints], device="mps")     # our joint i <- policy output
            # deterministic start: default state at the cell origin, zero yaw, zero velocity
            task.origins.assign(origins); task.reset_all()
            q0 = default_q.clone()[None].repeat(n, 1); q0[:, :3] = torch.as_tensor(origins, device="mps") + default_q[:3]
            task.sim.t.qpos.copy_(q0); task.sim.t.qvel.zero_(); task.last_action.zero_(); task.prev_action.zero_()
            torch.mps.synchronize()          # the copies run on the MPS queue; forward kinematics (Warp queue) must see them
            v = task.sim.forward(); task.sim.after(v); task.sim.synchronize()
            last = torch.zeros(n, nj, device="mps"); obs0 = None; hag = []; traj = []
            for t in range(a.steps):
                wp.launch(bump, dim=1, inputs=[step_idx], device="metal:0"); task.scanner.launch(step_idx)
                vs = task.sim._signal(); task.sim.after(vs)
                qp, qv = task.sim.t.qpos, task.sim.t.qvel
                R = quat_mat(qp[:, 3:7]); Rt = R.transpose(1, 2)
                v_b = (Rt @ qv[:, :3, None])[..., 0]; g_b = (Rt @ torch.tensor([0.0, 0.0, -1.0], device="mps")[None, :, None].expand(n, 3, 1))[..., 0]
                o = torch.cat([v_b, qv[:, 3:6], g_b, cmd, qp[:, 7:] - default_q[7:], qv[:, 6:], last, t_scan[:, :task.n_scan].clamp(-1, 1)], 1)
                if t == 0: obs0 = o.cpu().numpy()
                with torch.no_grad():
                    if src == "rsl_rl":
                        oi = torch.cat([o[:, :12], o[:, 12:12 + nj][:, perm_obs], o[:, 12 + nj:12 + 2 * nj][:, perm_obs],
                                        o[:, 12 + 2 * nj:12 + 3 * nj][:, perm_obs], o[:, 12 + 3 * nj:]], 1)
                        act = actor(oi)[:, perm_act]
                    else:
                        act = actor(torch.cat([o[:, :12 + 3 * nj], o[:, 12 + 3 * nj:][:, sperm]], 1))
                act = act.clamp(-100, 100); last = act
                task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch)
                task.sim.step()
                if a.traj_dir:
                    traj.append(task.sim.d.qpos.numpy().copy())
                if t % 10 == 9:
                    qn = task.sim.d.qpos.numpy(); hag.append(qn[:, 2] - ground(qn[:, 0], qn[:, 1]))
            task.sim.synchronize()
            qf = task.sim.d.qpos.numpy()
            x = (qf[:, 0] - origins[:, 0]).tolist(); z = (qf[:, 2] - origins[:, 2]).tolist()
            hag = np.array(hag)
            r = {"label": label, "ckpt": path, "src": src, "scan_ordering": order, "terrain_collision": a.terrain_collision, "scan_surface": a.scan_surface,
                 "level": lv, "columns": cols.tolist(), "x": x, "z": z,
                 "hag_final": hag[-1].tolist(), "hag_min": hag.min(0).tolist()}
            if a.reps > 1:
                r["reps"] = a.reps; r["jitter"] = a.jitter
            im = os.path.join(a.isaac_play or "", f"L{lv}", label, "meta.json")
            if a.isaac_play and os.path.exists(im) and a.reps == 1:
                mi = json.load(open(im)); io0 = np.array(mi["obs0"], np.float32)
                r["isaac_x"] = mi["final_root_x"]; r["isaac_z"] = mi["final_root_z"]
                tr = np.load(im.replace("meta.json", "traj.npz")); eo = np.array(mi["env_origins"], np.float32)
                pw = tr["root_pos"] + eo[None]                     # (T, n, 3) world pelvis position
                ih = pw[..., 2] - ground(pw[..., 0], pw[..., 1])
                r["isaac_hag_final"] = ih[-1].tolist(); r["isaac_hag_min"] = ih[9::10].min(0).tolist()
                r["env_origin_max_abs_diff"] = float(np.abs(eo - origins).max())
                r["scan0_max_abs_diff"] = float(np.abs(io0[:, 12 + 3 * nj:] - obs0[:, 12 + 3 * nj:]).max())
                r["scan0_mean_abs_diff"] = float(np.abs(io0[:, 12 + 3 * nj:] - obs0[:, 12 + 3 * nj:]).mean())
                r["head0_max_abs_diff"] = float(np.abs(io0[:, :12] - obs0[:, :12]).max())
                dsc = np.abs(io0[:, 12 + 3 * nj:] - obs0[:, 12 + 3 * nj:])
                r["scan0_max_abs_diff_per_env"] = dsc.max(1).tolist(); r["scan0_rays_over_2cm_per_env"] = (dsc > 0.02).sum(1).tolist()
            if a.traj_dir:
                os.makedirs(a.traj_dir, exist_ok=True)
                np.savez(os.path.join(a.traj_dir, f"L{lv}_{label}_{a.terrain_collision.replace(':', '_')}_{a.scan_surface}.npz"),
                         qpos=np.array(traj), origins=origins)
            print(json.dumps(r), flush=True); results.append(r)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(r) + "\n")
    print("| level | policy | Isaac x [m] per env (cols 0/4/9/14) | Isaac pelvis above ground final (min) | MetalSim x | MetalSim pelvis above ground final (min) |")
    fmt2 = lambda f, m_: " / ".join(f"{u:.2f} ({w:.2f})" for u, w in zip(f, m_)) if f is not None else "-"
    fmt = lambda v: " / ".join(f"{u:.2f}" for u in v) if v is not None else "-"
    for r in results:
        print(f"| {r['level']} | {r['label']} | {fmt(r.get('isaac_x'))} | {fmt2(r.get('isaac_hag_final'), r.get('isaac_hag_min'))} | "
              f"{fmt(r['x'])} | {fmt2(r['hag_final'], r['hag_min'])} |")
    # falls: final pelvis height above the local ground below 0.3 m (standing ~0.66-0.75 m, fallen < 0.1 m)
    names = {0: "pyramid stairs", 4: "inverted stairs", 9: "boxes", 14: "random rough"}
    print(f"falls (pelvis above ground < {FALL_HAG} m at the end) per cell type, terrain_collision={a.terrain_collision}, "
          f"scan_surface={a.scan_surface}, reps={a.reps}:")
    for c in (0, 4, 9, 14):
        idx = [i for i, cc in enumerate(cols.tolist()) if cc == c]
        f = sum(r["hag_final"][i] < FALL_HAG for r in results for i in idx); dx = np.mean([r["x"][i] for r in results for i in idx])
        print(f"  {names.get(c, c)}: {f} / {len(results) * len(idx)} falls, mean x {dx:.2f} m")
    print(f"  total: {sum(sum(h < FALL_HAG for h in r['hag_final']) for r in results)} / {len(results) * len(cols)}")


FALL_HAG = 0.3


if __name__ == "__main__":
    main()
