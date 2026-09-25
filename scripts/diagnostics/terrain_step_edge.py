"""A G1 foot at a stair riser on Isaac's rough terrain (inverted pyramid stairs, row 3, column 4: the pit where Isaac's
policies fall in MetalSim), MuJoCo Warp vs MuJoCo C on the same model, for each terrain collision surface.

The foot is the G1's own foot collider (the 8-vertex ``ankle_roll_link`` mesh, 0.20 x 0.065 m sole) on a free body of
5 kg. World 0 "stub": sole 3 mm above the pit platform, toe 1 cm before the 0.11 m riser, moving into it at 1 m/s.
World 1 "edge": foot centred over the riser edge (60 % on the upper tread), dropped from 1 cm. 0.4 s at 2.5 ms.
Isaac's geometry is a vertical wall at the riser: the stub must stop at the wall (toe x <= riser x) with a near-horizontal
contact normal and without climbing; a 0.1 m heightfield turns the wall into a 48 degree ramp starting 0.1 m early.

    scripts/gpu_run.sh terrain_step_edge render 5 -- python scripts/diagnostics/terrain_step_edge.py --out runs/terrain_walls/step_edge.jsonl
"""
import argparse, json
import numpy as np
import mujoco

ROW, COL = 3, 4
DT = 0.0025
STEPS = 160


def foot_vertices():
    from metalsim.learn.g1_velocity import build_g1_model
    m, _ = build_g1_model("flat")
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "Cube")
    mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return m.mesh_vert[a:a + nv] @ R.reshape(3, 3).T + m.geom_pos[g]      # ankle_roll_link frame (x forward, z up)


def build(mode, fine_res=0.025, _hf=None):
    from metalsim.learn.terrain import isaac_rough_terrain, add_terrain_geoms
    hf = isaac_rough_terrain(seed=0, collision=mode, fine_res=fine_res)
    spec = mujoco.MjSpec()
    spec.option.timestep = DT; spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL; spec.option.iterations = 10; spec.option.ls_iterations = 20
    h = spec.add_hfield(); h.name = "terrain"; h.nrow, h.ncol = hf["nrow"], hf["ncol"]; h.size = hf["size"]
    h.userdata = hf["data"].reshape(-1)
    g = spec.worldbody.add_geom(); g.name = "ground"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "terrain"
    g.pos = [0, 0, hf["zmin"]]; g.friction = [0.8, 0.005, 0.0001]
    add_terrain_geoms(spec, hf, g)
    V = foot_vertices()
    mesh = spec.add_mesh(); mesh.name = "foot"; mesh.uservert = V.reshape(-1).tolist()
    b = spec.worldbody.add_body(); b.name = "foot"; b.add_freejoint()
    fg = b.add_geom(); fg.type = mujoco.mjtGeom.mjGEOM_MESH; fg.meshname = "foot"; fg.mass = 5.0; fg.friction = [0.8, 0.005, 0.0001]
    m = spec.compile()
    # riser: +x edge of the pit's platform (the cell's last box), 0.11 m up to the inner ring
    from metalsim.learn.terrain import terrain_boxes
    B = terrain_boxes(hf["generator"]); cb = B[(B[:, 6] == ROW) & (B[:, 7] == COL)]
    plat = cb[-1]; x_riser = plat[0] + plat[3]; z_low = plat[2] + plat[5]
    ring = cb[(np.abs(cb[:, 0] - (x_riser + cb[:, 3])) < 1e-6) & (np.abs(cb[:, 1] - plat[1]) < 1e-6)][0]
    z_high = ring[2] + ring[5]
    return m, V, dict(x_riser=float(x_riser), z_low=float(z_low), z_high=float(z_high), y=float(plat[1])), hf


def initial_states(V, geo):
    x_toe, z_sole = V[:, 0].max(), V[:, 2].min()
    q = np.zeros((2, 7)); qd = np.zeros((2, 6)); q[:, 3] = 1.0
    q[0, :3] = (geo["x_riser"] - 0.01 - x_toe, geo["y"], geo["z_low"] - z_sole + 3e-3); qd[0, 0] = 1.0
    xc = 0.5 * (V[:, 0].max() + V[:, 0].min()); L = V[:, 0].max() - V[:, 0].min()
    q[1, :3] = (geo["x_riser"] + 0.1 * L - xc, geo["y"], geo["z_high"] - z_sole + 0.01)
    return q, qd


def run_c(m, q, qd):
    out = []
    for w in range(len(q)):
        d = mujoco.MjData(m); d.qpos[:] = q[w]; d.qvel[:] = qd[w]
        traj, normals = [], []
        for t in range(STEPS):
            mujoco.mj_step(m, d)
            traj.append(d.qpos.copy())
            normals += [c.frame[:3].copy() for c in d.contact[:d.ncon]]
        out.append((np.array(traj), np.array(normals).reshape(-1, 3)))
    return out


def run_warp(m, q, qd, hf=None):
    """``hf``: the terrain dict of a "boxes_local" model (per-world box slots filled around the foot by BoxWindow)."""
    import torch
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.learn.terrain import BoxWindow
    pwf = BoxWindow.FIELDS if hf is not None else ()
    sim = BatchSim(m, len(q), options=BatchSimOptions(substeps=1, nconmax=64, njmax=256, per_world_fields=pwf))
    sim.t.qpos.copy_(torch.as_tensor(q, dtype=torch.float32)); sim.t.qvel.copy_(torch.as_tensor(qd, dtype=torch.float32))
    torch.mps.synchronize()
    if hf is not None:
        win = BoxWindow(sim, hf); sim.add_substep_hook(win.launch); win.launch()
    v = sim.forward(); sim.after(v); sim.synchronize()
    trajs = [[] for _ in q]; normals = [[] for _ in q]
    d = sim.d
    for t in range(STEPS):
        sim.step(); sim.synchronize()
        qp = d.qpos.numpy()
        for w in range(len(q)): trajs[w].append(qp[w].copy())
        na = int(d.nacon.numpy()[0])
        if na:
            fr = d.contact.frame.numpy()[:na]; wid = d.contact.worldid.numpy()[:na]
            for k in range(na): normals[wid[k]].append(fr[k][0] if fr.ndim == 3 else fr[k][:3])
    return [(np.array(trajs[w]), np.array(normals[w]).reshape(-1, 3)) for w in range(len(q))]


def metrics(V, geo, res):
    (ta, na), (tb, nb) = res
    x_toe = V[:, 0].max(); z_sole = V[:, 2].min()
    return {"stub_toe_x_max_minus_riser": float((ta[:, 0] + x_toe).max() - geo["x_riser"]),     # > 0: went through / up the wall
            "stub_sole_rise": float(ta[:, 2].max() + z_sole - geo["z_low"]),                   # climbing the riser
            "stub_min_abs_normal_z": float(np.abs(na[:, 2]).min()) if len(na) else None,        # ~0: a wall contact exists
            "stub_final_x": float(ta[-1, 0]), "stub_final_z": float(ta[-1, 2]),
            "edge_final_x": float(tb[-1, 0]), "edge_final_z": float(tb[-1, 2]), "edge_final_quat": tb[-1, 3:7].round(4).tolist()}


def run(mode, fine_res=0.025, warp=True):
    m, V, geo, hf = build(mode, fine_res)
    q, qd = initial_states(V, geo)
    if mode == "boxes_local":        # MuJoCo C on the full box model (the same geometry the window holds near the foot)
        mc = build("boxes")[0]
        rc = run_c(mc, q, qd)
    else:
        rc = run_c(m, q, qd)
    r = {"mode": mode if mode != "hfield_fine" else f"hfield_fine:{fine_res}", "geometry": geo, "C": metrics(V, geo, rc)}
    if warp:
        rw = run_warp(m, q, qd, hf if mode == "boxes_local" else None)
        r["warp"] = metrics(V, geo, rw)
        r["warp_vs_C_pos_max"] = float(max(np.abs(rw[w][0][:, :3] - rc[w][0][:, :3]).max() for w in range(2)))
        r["warp_vs_C_pos_final"] = [float(np.linalg.norm(rw[w][0][-1, :3] - rc[w][0][-1, :3])) for w in range(2)]
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--modes", nargs="+", default=["hfield", "boxes", "meshes", "hfield_fine:0.05", "hfield_fine:0.025"])
    ap.add_argument("--out", default=None); ap.add_argument("--no_warp", action="store_true")
    a = ap.parse_args()
    import warp as wp; wp.config.quiet = True
    for s in a.modes:
        mode, _, fr = s.partition(":")
        r = run(mode, float(fr or 0.025), warp=not a.no_warp)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(r) + "\n")
