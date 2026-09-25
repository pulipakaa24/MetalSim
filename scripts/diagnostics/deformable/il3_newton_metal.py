"""The Isaac Lab 3.0 Newton-backend deformable recording, replayed with the same solver (Newton 1.5.2 SolverVBD) on the
MetalSim Warp fork (.venv-newton152). Scene built as Isaac Lab's Newton backend builds it (isaaclab_contrib deformable
builder hooks: add_cloth_mesh / add_soft_mesh with the registry's density, stiffnesses and particle radius; ground and
box shapes with NewtonShapeCfg defaults ke 2.5e3, kd 100, gap 0.01 and the USD material friction 0.5; soft contact
ke 1e5, kd 1, mu 0.5; VBD 20 iterations, 4 substeps at 200 Hz; rods pinned by inverse mass 0 / inactive flag, as
isaaclab_contrib's enforce_kinematic_targets). Meshes: the recording's cooked cloth mesh and make_tetmeshes.py rods and
cubes. Writes record.npz / meta.json in the recorder's format so il3_analysis.py reads both.
usage: il3_newton_metal.py REC_DIR(newton_vbd) OUT_DIR [--device metal:0] [--worlds N] [--bench]"""
import argparse, json, os, sys, time
import numpy as np
import warp as wp

import newton

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "metalsim", "parity", "isaac_side"))
from make_tetmeshes import grid_tets  # noqa: E402


def build(meta, cooked, worlds=1, scenes=("cloth", "rope", "cube"), only=None):
    env = newton.ModelBuilder()
    sc = meta["scenes"]
    groups = {}
    shape_cfg = newton.ModelBuilder.ShapeConfig(ke=2.5e3, kd=100.0, mu=0.5, gap=0.01)
    if "cloth" in scenes:
        env.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.2, cfg=shape_cfg)
        pts = cooked["|World|cloth_scene|cloth|sim_mesh:points"].astype(np.float32)
        tri = cooked["|World|cloth_scene|cloth|sim_mesh:faceVertexIndices"].astype(np.int32)
        mt = sc["cloth"]["material"]
        a = env.particle_count
        env.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.5), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0), vertices=[wp.vec3(*p) for p in pts],
                           indices=tri.tolist(), density=mt["density"], tri_ke=mt["tri_ke"], tri_ka=mt["tri_ka"], tri_kd=mt["tri_kd"],
                           edge_ke=mt["edge_ke"], edge_kd=mt["edge_kd"], particle_radius=mt["particle_radius"])
        groups["cloth"] = (a, env.particle_count)
    pins = []
    for key in [k for k in sc if k.startswith("rod_n") or k.startswith("cube_n")]:
        kind = "rope" if key.startswith("rod") else "cube"
        if kind not in scenes or (only is not None and key not in only):
            continue
        n = int(key.split("_n")[1])
        pts, tets = grid_tets((0.5, 0.02, 0.02), (10 * n, n, n)) if kind == "rope" else grid_tets((0.2, 0.2, 0.2), (n, n, n))
        mt = sc[key]["material"]
        x0 = sc[key]["x0"] + (0.25 if kind == "rope" else 0.0)
        a = env.particle_count
        env.add_soft_mesh(pos=wp.vec3(x0, sc[key]["y0"], sc[key]["z0"]), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
                          vertices=[wp.vec3(*p) for p in pts.astype(np.float32)], indices=tets.reshape(-1).tolist(), density=mt["density"],
                          k_mu=mt["k_mu"], k_lambda=mt["k_lambda"], k_damp=mt["k_damp"], particle_radius=mt["particle_radius"])
        groups[key] = (a, env.particle_count)
        if kind == "rope":
            pins += list(a + np.where(pts[:, 0] < pts[:, 0].min() + 1e-6)[0])
    b = newton.ModelBuilder()
    b.add_ground_plane(cfg=shape_cfg)
    if worlds == 1:
        b.add_builder(env)
    else:
        b.replicate(env, worlds)
    b.color()
    m = b.finalize()
    pc = env.particle_count
    allpins = np.array([p + w * pc for w in range(worlds) for p in pins], np.int64)
    if len(allpins):
        im = m.particle_inv_mass.numpy(); fl = m.particle_flags.numpy()
        im[allpins] = 0.0; fl[allpins] = 0
        m.particle_inv_mass.assign(im); m.particle_flags.assign(fl)
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 1.0e5, 1.0, 0.5
    return m, groups, pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec"); ap.add_argument("out")
    ap.add_argument("--device", default="metal:0"); ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=5.0); ap.add_argument("--bench", default="")
    a = ap.parse_args()
    wp.config.quiet = True
    meta = json.load(open(os.path.join(a.rec, "meta.json")))
    cooked = np.load(os.path.join(a.rec, "cooked.npz"))
    scenes = tuple(a.bench.split(",")) if a.bench else ("cloth", "rope", "cube")
    with wp.ScopedDevice(a.device):
        # benchmark one object per scene type: the 1-across rod (PhysX 5.1's mesh) and the 4-per-edge cube
        m, groups, pc = build(meta, cooked, a.worlds, scenes, only=("rod_n1", "cube_n4") if a.bench else None)
        solver = newton.solvers.SolverVBD(m, iterations=20)
        s0, s1 = m.state(), m.state(); ctrl = m.control()
        pipe = newton.CollisionPipeline(m); contacts = pipe.contacts()
        sub, dt = 4, 1.0 / 200.0 / 4

        def frame():
            nonlocal s0, s1
            for _ in range(sub):
                s0.clear_forces(); pipe.collide(s0, contacts); solver.step(s0, s1, ctrl, contacts, dt); s0, s1 = s1, s0

        frame(); wp.synchronize_device()                    # eager frame first (VBD sizes its buffers on the first step)
        graph = None
        if not wp.get_device().is_cpu:
            with wp.ScopedCapture() as cap:
                frame()
            graph = cap.graph
        nframes = int(round(a.seconds * 200))
        if a.bench:
            for _ in range(5):
                wp.capture_launch(graph) if graph else frame()
            wp.synchronize_device(); t = time.perf_counter(); n = 0
            while time.perf_counter() - t < 3.0:
                for _ in range(5):
                    wp.capture_launch(graph) if graph else frame()
                wp.synchronize_device(); n += 5
            el = time.perf_counter() - t
            print(json.dumps({"bench": a.bench, "worlds": a.worlds, "particles_per_world": pc, "env_steps_per_s": a.worlds * n / el,
                              "ms_per_frame": el / n * 1e3, "finite": bool(np.isfinite(s0.particle_q.numpy()).all())}), flush=True)
            return
        rec = {k + "_pos": [] for k in groups}
        rec.update({k + "_vel": [] for k in groups})
        # frames 0,1 were run above (eager + capture); restart from the initial state for the recording
        s0.particle_q.assign(m.particle_q); s0.particle_qd.assign(m.particle_qd)
        t0 = time.perf_counter()
        for k in range(nframes + 1):
            if k:
                wp.capture_launch(graph) if graph else frame()
            q = s0.particle_q.numpy(); v = s0.particle_qd.numpy()
            for g, (lo, hi) in groups.items():
                rec[g + "_pos"].append(q[lo:hi].copy()); rec[g + "_vel"].append(v[lo:hi].copy())
        wall = time.perf_counter() - t0
    os.makedirs(a.out, exist_ok=True)
    out = {k: np.array(v, np.float32) for k, v in rec.items()}
    for g, (lo, hi) in groups.items():
        if g.startswith("rod_n"):
            out[g.replace("rod_n", "rod_n") + "_pinned"] = (m.particle_inv_mass.numpy()[lo:hi] == 0)
    np.savez_compressed(os.path.join(a.out, "record.npz"), **out)
    meta2 = dict(meta); meta2["replay"] = {"device": a.device, "newton": newton.__version__, "wall_s": wall, "graph": graph is not None}
    json.dump(meta2, open(os.path.join(a.out, "meta.json"), "w"), indent=1, default=str)
    print("NEWTON_METAL_DONE", a.device, wall, {g: hi - lo for g, (lo, hi) in groups.items()})


if __name__ == "__main__":
    main()
