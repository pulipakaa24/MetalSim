"""Newton VBD on the MetalSim Warp fork: cloth drop onto a box and a soft cube drop, CPU vs Metal.

Newton source is chosen by PYTHONPATH (Isaac Lab 3.0 EA pins Newton 1.5.2: upstream/newton-1.5.2;
the MetalSim fork 1.7.0.dev: upstream/newton, the .venv-newtonfork default). Run Metal only through
scripts/gpu_run.sh. Writes trajectories to --out (npz) and prints one JSON line per scene.

    PYTHONPATH=upstream/newton-1.5.2 .venv-newtonfork/bin/python scripts/diagnostics/newton_vbd/vbd_probe.py \
        --device cpu --out runs/newton_vbd/cpu_152.npz
"""
from __future__ import annotations

import argparse, json, time, traceback

import numpy as np
import warp as wp

import newton


def _finalize(env, worlds: int, include_bending: bool):
    """Ground + `worlds` copies of `env` (Newton worlds, all at the origin), coloured for VBD."""
    b = newton.ModelBuilder()
    b.add_ground_plane()
    if worlds == 1:
        b.add_builder(env)
    else:
        b.replicate(env, worlds)
    b.color(include_bending=include_bending)
    return b.finalize()


def cloth_model(n: int = 21, size: float = 1.0, z0: float = 0.5, box: float = 0.4, worlds: int = 1):
    """PhysX-protocol-like cloth: n x n vertices, size m, released flat at z0 over a static box of edge `box`."""
    e = newton.ModelBuilder()
    e.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, box / 2), wp.quat_identity()), hx=box / 2, hy=box / 2, hz=box / 2)
    cell = size / (n - 1)
    e.add_cloth_grid(pos=wp.vec3(-size / 2, -size / 2, z0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=n - 1, dim_y=n - 1,
                     cell_x=cell, cell_y=cell, mass=0.02, tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.0e-5, edge_ke=1.0e-2, edge_kd=0.0,
                     particle_radius=cell * 0.5)
    m = _finalize(e, worlds, True)
    m.soft_contact_ke = 1.0e4
    m.soft_contact_kd = 1.0e-2
    m.soft_contact_mu = 0.6
    return m


def cube_model(n: int = 4, size: float = 0.2, z0: float = 0.5, worlds: int = 1):
    """Soft cube of edge `size`, n cells per side, centre at z0 (PhysX protocol: E 1e5, nu 0.4, density 1000)."""
    e = newton.ModelBuilder()
    E, nu = 1.0e5, 0.4
    mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    cell = size / n
    e.add_soft_grid(pos=wp.vec3(-size / 2, -size / 2, z0 - size / 2), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=n, dim_y=n, dim_z=n,
                    cell_x=cell, cell_y=cell, cell_z=cell, density=1000.0, k_mu=mu, k_lambda=lam, k_damp=1.0e-3,
                    particle_radius=cell * 0.25)
    m = _finalize(e, worlds, False)
    m.soft_contact_ke = 1.0e5
    m.soft_contact_kd = 1.0e-2
    m.soft_contact_mu = 0.5
    return m


def run(scene: str, device: str, steps: int, substeps: int, iterations: int, tile: bool, self_contact: bool, capture: bool,
        perturb: float = 0.0, worlds: int = 1, record: bool = True):
    with wp.ScopedDevice(device):
        t0 = time.perf_counter()
        m = cloth_model(worlds=worlds) if scene == "cloth" else cube_model(worlds=worlds)
        kw = dict(iterations=iterations, particle_enable_tile_solve=tile)
        if scene == "cloth" and self_contact:
            kw.update(particle_enable_self_contact=True, particle_self_contact_radius=0.01, particle_self_contact_margin=0.015)
        solver = newton.solvers.SolverVBD(m, **kw)
        s0, s1 = m.state(), m.state()
        if perturb:   # float32-level control: displace every particle by `perturb` m (seeded)
            q = s0.particle_q.numpy()
            q += np.random.default_rng(0).uniform(-perturb, perturb, q.shape).astype(np.float32)
            s0.particle_q.assign(q)
        ctrl = m.control()
        pipe = newton.CollisionPipeline(m)
        contacts = pipe.contacts()
        dt = 1.0 / 200.0 / substeps
        build_s = time.perf_counter() - t0

        def sim():
            nonlocal s0, s1
            for _ in range(substeps):
                s0.clear_forces()
                pipe.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0

        t0 = time.perf_counter()
        sim()                                      # compiles every kernel
        wp.synchronize_device()
        first_s = time.perf_counter() - t0
        graph = None
        if capture and not wp.get_device().is_cpu:
            with wp.ScopedCapture() as cap:
                sim()                               # even number of swaps per step keeps s0/s1 fixed
            graph = cap.graph
        traj = [s0.particle_q.numpy().copy()]
        t0 = time.perf_counter()
        for _ in range(steps - (2 if graph is not None else 1)):
            if graph is not None:
                wp.capture_launch(graph)
            else:
                sim()
            if graph is None and record:
                traj.append(s0.particle_q.numpy().copy())
        wp.synchronize_device()
        loop_s = time.perf_counter() - t0
        q = s0.particle_q.numpy()
        return {"scene": scene, "device": device, "newton_src": newton.__file__.split("/upstream/")[-1].split("/newton/")[0], "n_particles": int(m.particle_count),
                "steps": steps, "substeps": substeps, "iterations": iterations, "tile": tile, "self_contact": self_contact,
                "graph": graph is not None, "perturb": perturb, "worlds": worlds,
                "env_steps_per_s": worlds * (steps - (2 if graph is not None else 1)) / loop_s, "build_s": build_s, "first_step_s": first_s, "loop_s": loop_s,
                "finite": bool(np.isfinite(q).all()), "z_min": float(q[:, 2].min()), "z_mean": float(q[:, 2].mean()),
                "z_max": float(q[:, 2].max())}, np.asarray(traj), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scenes", nargs="+", default=["cloth", "cube"])
    ap.add_argument("--steps", type=int, default=200)          # 1 s at 200 Hz
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--no-tile", action="store_true")
    ap.add_argument("--self-contact", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--perturb", type=float, default=0.0, help="initial particle jitter (m), a float32-level control")
    ap.add_argument("--worlds", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    wp.config.quiet = True
    out = {}
    for sc, nw in [(sc, nw) for sc in a.scenes for nw in a.worlds]:
        try:
            r, traj, q = run(sc, a.device, a.steps, a.substeps, a.iterations, not a.no_tile, a.self_contact, a.capture, a.perturb,
                             nw, record=nw == 1)
            out[f"{sc}_final"] = q
            if len(traj) > 1:
                out[f"{sc}_traj"] = traj
        except Exception as e:  # report the exact failure
            r = {"scene": sc, "worlds": nw, "device": a.device, "newton_src": newton.__file__.split("/upstream/")[-1].split("/newton/")[0], "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc()[-4000:]}
        print(json.dumps(r), flush=True)
    if a.out and out:
        np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
