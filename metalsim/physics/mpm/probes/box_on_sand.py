"""A free box dropped onto a sand bed: MetalSim's main simulator (BatchSim + ``MPMCoupling``) against the same scene
through Isaac Lab 3.0-EA's coupling path (Newton 1.5.2 ``SolverCoupledProxy``: ``SolverMuJoCo`` arm entry +
``SolverImplicitMPM`` media entry, staggered proxy with MuJoCo's effective mass).

    .venv/bin/python           box_on_sand.py --engine metalsim --out A.npz    # main environment
    .venv-newton152/bin/python box_on_sand.py --engine newton   --out B.npz    # Isaac's path (Newton 1.5.2)
    python box_on_sand.py --compare A.npz B.npz

Same everything: box 0.1 m, 2 kg, from 0.25 m; sand bed (Franka-Pour's media material) on a frictional floor; coupled
step dt 1/120 s with 3 MuJoCo substeps and 1 MPM substep; implicitfast integrator; the box touches only sand (MuJoCo
contacts off on the Newton side, and the box never reaches the floor). Records the box pose and velocity and the
particle positions every step, and the sand wrench on the box.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np

DT = 1.0 / 120.0
ARM_SUBSTEPS = 3
BOX_H, BOX_M, BOX_Z = 0.05, 2.0, 0.25
BED_LO, BED_HI = (-0.15, -0.15, 0.0), (0.15, 0.15, 0.08)
SAND = dict(density=1500.0, friction=0.7)


def mpm_settings(voxel):
    return dict(voxel_size=voxel, max_iterations=50, tolerance=1.0e-4, solver="auto", warmstart_mode="auto",
                collider_velocity_mode="forward", grid_type="sparse", grid_padding=0, transfer_scheme="apic",
                integration_scheme="pic", critical_fraction=0.0, air_drag=1.0, collider_basis="S2",
                strain_basis="P0", velocity_basis="Q1")


XML = f"""<mujoco><option timestep="{DT / ARM_SUBSTEPS!r}" integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" friction="0.5"/>
<body name="box" pos="0 0 {BOX_Z}"><freejoint/><geom name="box" type="box" size="{BOX_H} {BOX_H} {BOX_H}" mass="{BOX_M}"
 friction="0.5"/></body></worldbody></mujoco>"""


def run_metalsim(a):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    model = mujoco.MjModel.from_xml_string(XML)
    sim = BatchSim(model, a.worlds, options=BatchSimOptions(substeps=ARM_SUBSTEPS))
    s = mpm_settings(a.voxel)
    cfg = MPMConfig(**s, max_active_cell_count=a.cells, capture=bool(a.capture), proxy_mass=a.mass, mode=a.mode,
                    mpm_substeps=a.mpm_substeps)
    mpm = MPMCoupling(sim, cfg)
    mpm.add_emitter(GridEmitter(lower=BED_LO, upper=BED_HI, material=MPMMaterial(**SAND)))
    mpm.finalize()
    rec = _recorder(a)
    for _ in range(a.steps):
        mpm.step()
        rec(sim.d.xpos.numpy()[:, 1], _wxyz_to_xyzw(sim.d.xquat.numpy()[:, 1]), sim.d.qvel.numpy()[:, :6],
            mpm.particles(), mpm.wrenches()[:, 0])
    return rec.result(meta=dict(engine="metalsim", proxy_mass=mpm.proxy_mass.tolist()))


def _wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:], q[..., :1]], -1)


def run_newton(a):
    import warp as wp

    import newton
    from newton.solvers import SolverImplicitMPM, SolverMuJoCo
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

    from metalsim.physics.mpm import newton_compat

    newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=True, gpu_exit=True,
                          rigid_deterministic=True, device_args=None, metal_fast_math="off")
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    SolverImplicitMPM.register_custom_attributes(b)
    from metalsim.physics.mpm.coupled import GridEmitter, MPMMaterial

    em = GridEmitter(lower=BED_LO, upper=BED_HI, material=MPMMaterial(**SAND))
    floors, boxes = [], []
    for _w in range(a.worlds):
        b.begin_world()
        floors.append(b.add_shape_plane(width=0.0, length=0.0, body=-1,
                                        cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0)))
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, BOX_Z), wp.quat_identity()), mass=BOX_M,
                          inertia=wp.mat33(*(np.eye(3) * BOX_M * (2 * BOX_H) ** 2 / 6.0).astype(np.float32).ravel()),
                          lock_inertia=True)
        b.add_joint_free(body)
        b.add_shape_box(body, hx=BOX_H, hy=BOX_H, hz=BOX_H,
                        cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0))
        boxes.append(body)
        em.emit(b, a.voxel)
        b.end_world()
    with wp.ScopedDevice("metal:0"):
        m = b.finalize()
        s = mpm_settings(a.voxel)
        cfg = SolverImplicitMPM.Config(**s)
        ppw = m.particle_count // a.worlds
        cfg.max_active_cell_count = ((ppw + 511) // 512) * 512 * a.worlds if a.cells == "auto" else int(a.cells)
        cfg.separate_worlds = True
        shape_body = m.shape_body.numpy()
        box_shapes = [i for i in range(m.shape_count) if shape_body[i] >= 0]
        floor_shapes = [i for i in range(m.shape_count) if shape_body[i] < 0]
        entries = [
            SolverCoupled.Entry(name="arm", solver=lambda view: SolverMuJoCo(view, use_mujoco_contacts=False,
                                                                              integrator="implicitfast"),
                                bodies=boxes, particles=[], joints=list(range(m.joint_count)), shapes=box_shapes,
                                substeps=ARM_SUBSTEPS),
            SolverCoupled.Entry(name="media", solver=lambda view: SolverImplicitMPM(view, cfg), bodies=[],
                                particles=list(range(m.particle_count)), joints=[], shapes=floor_shapes,
                                substeps=a.mpm_substeps, in_place=True),
        ]
        mass_scale = 1.0
        coupling = SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="arm", destination="media", bodies=boxes, particles=[],
                                              mode=a.mode, mass_scale=mass_scale, collide_interval=None,
                                              collision_pipeline=None)], iterations=1)
        solver = SolverCoupledProxy(model=m, entries=entries, coupling=coupling)
        s0, s1 = m.state(), m.state()
        control = m.control()
        newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
        newton.eval_fk(m, m.joint_q, m.joint_qd, s1)
        rec = _recorder(a)
        nb = m.body_count // a.worlds
        for _ in range(a.steps):
            solver.step(s0, s1, control, None, DT)
            s0, s1 = s1, s0
            s0.clear_forces()
            bq = s0.body_q.numpy().reshape(a.worlds, nb, 7)[:, 0]
            jqd = s0.joint_qd.numpy().reshape(a.worlds, 6)
            # the harvested proxy wrenches the next step applies (SolverCoupledProxy's body mapping, global body ids)
            mp = [x for x in solver._proxy_mappings if x.coupling_forces is not None
                  and x.coupling_forces.dtype == wp.spatial_vector]
            fa = mp[0].coupling_forces.numpy().reshape(-1, 6) if mp else np.zeros((m.body_count, 6))
            f = fa[boxes] if fa.shape[0] == m.body_count else fa[:a.worlds]
            rec(bq[:, :3], bq[:, 3:], jqd, s0.particle_q.numpy().reshape(a.worlds, -1, 3), f)
    return rec.result(meta=dict(engine="newton"))


def _recorder(a):
    out = {"pos": [], "quat": [], "qvel": [], "wrench": [], "phash": []}
    keep = {"particles": []}

    def rec(pos, quat, qvel, particles, wrench):
        out["pos"].append(np.asarray(pos, np.float64))
        out["quat"].append(np.asarray(quat, np.float64))
        out["qvel"].append(np.asarray(qvel, np.float64))
        out["wrench"].append(np.asarray(wrench, np.float64))
        p = np.ascontiguousarray(particles, np.float32)
        out["phash"].append(hashlib.sha256(p.tobytes()).hexdigest()[:16])
        keep["particles"].append(p)

    def result(meta):
        r = {k: np.stack(v) if k != "phash" else np.array(v) for k, v in out.items()}
        r["particles"] = np.stack(keep["particles"][:: max(1, a.particle_stride)])
        r["particles_last"] = keep["particles"][-1]
        r["meta"] = np.array(repr(meta))
        return r

    rec.result = result
    return rec


def compare(fa, fb):
    A, B = np.load(fa), np.load(fb)
    print(A["meta"], B["meta"])
    n = min(len(A["pos"]), len(B["pos"]))
    dz = np.abs(A["pos"][:n] - B["pos"][:n]).max(axis=(1, 2))
    same = int((A["phash"][:n] == B["phash"][:n]).sum())
    pa, pb = A["particles_last"], B["particles_last"]
    print(f"steps {n}: particle hashes equal on {same} steps; box position max |diff| per step "
          f"first {dz[0]:.3e} mid {dz[n // 2]:.3e} last {dz[-1]:.3e}")
    print(f"final box z {A['pos'][n - 1][:, 2]} vs {B['pos'][n - 1][:, 2]}")
    if pa.shape == pb.shape:
        print(f"final particle RMS {np.sqrt(((pa - pb) ** 2).sum(-1).mean()):.3e} m, max {np.abs(pa - pb).max():.3e}")
    first = next((i for i in range(n) if A["phash"][i] != B["phash"][i]), None)
    print("first differing particle step:", first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["metalsim", "newton"])
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--out")
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--cells", default="auto")
    ap.add_argument("--capture", type=int, default=0)
    ap.add_argument("--mass", default="effective")
    ap.add_argument("--mode", default="staggered")
    ap.add_argument("--mpm-substeps", type=int, default=1)
    ap.add_argument("--particle-stride", type=int, default=10)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    import warp as wp

    wp.config.quiet = True
    if a.cells != "auto":
        a.cells = int(a.cells)
    with wp.ScopedDevice("metal:0"):
        r = run_metalsim(a) if a.engine == "metalsim" else run_newton(a)
    np.savez_compressed(a.out, **r)
    print(a.engine, "box z final", r["pos"][-1][:, 2], "particles finite", bool(np.isfinite(r["particles_last"]).all()))


if __name__ == "__main__":
    main()
