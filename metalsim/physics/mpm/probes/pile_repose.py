"""Axisymmetric granular column collapse on a floor: MetalSim's main simulator (BatchSim floor + MPMCoupling) and the
same scene in Newton alone (SolverImplicitMPM with a Newton plane), for a parity check and the deposit shape.

A column (radius R, height H) released on a frictional floor slumps; once at rest the deposit is measured: maximum
height, runout and the flank slope (95th percentile of particle heights per radial bin, linear fit where
0.2 h_max < h < 0.8 h_max). The deposit slope of a *collapse* is not the angle of repose: the flow is inertial and
leaves a flatter deposit than a quasi-static heap (Lajeunesse et al. 2004, Lube et al. 2004). The angle of repose is
measured by ``repose_lift.py`` (cylinder lifted slowly).

    python pile_repose.py --engine metalsim|newton [--friction 0.68] [--voxel 0.02] [--seconds 2.0] [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

XML = """<mujoco><option timestep="{dt!r}"/><worldbody>
<geom name="floor" type="plane" size="0 0 1" friction="{floor_mu}"/></worldbody></mujoco>"""


def surface_angle(q: np.ndarray, bins: int = 24) -> dict:
    """Flank slope [deg] of a pile centred at the origin from particle positions (P, 3)."""
    r = np.hypot(q[:, 0], q[:, 1])
    z = q[:, 2]
    edges = np.linspace(0.0, np.quantile(r, 0.995), bins + 1)
    rc, h = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() >= 8:
            rc.append(0.5 * (a + b))
            h.append(np.quantile(z[m], 0.95))
    rc, h = np.asarray(rc), np.asarray(h)
    sel = (h > 0.2 * h.max()) & (h < 0.8 * h.max())
    out = {"h_max": float(h.max()), "runout": float(rc[h > 0.02 * h.max()].max()), "bins": int(sel.sum())}
    if sel.sum() >= 3:
        slope, _ = np.polyfit(rc[sel], h[sel], 1)
        out["angle_deg"] = float(math.degrees(math.atan(-slope)))
    else:
        out["angle_deg"] = float("nan")
    return out


def column(radius, height, voxel):
    """Isaac's cell_center lattice (3 particles per voxel) inside the cylinder."""
    sp = voxel / 3.0
    n = int(math.ceil(radius / sp))
    g = (np.arange(-n, n) + 0.5) * sp
    x, y, zz = np.meshgrid(g, g, (np.arange(int(math.ceil(height / sp))) + 0.5) * sp, indexing="ij")
    p = np.stack([x, y, zz], -1).reshape(-1, 3)
    return p[np.hypot(p[:, 0], p[:, 1]) <= radius].astype(np.float32), sp


def run(engine="metalsim", friction=0.68, voxel=0.02, seconds=2.0, radius=0.08, height=0.16, floor_mu=0.7,
        ke_rest=1e-4, fps=60, substeps=2):
    import warp as wp

    from metalsim.physics.mpm.coupled import MPMConfig, MPMMaterial, PointsEmitter

    p, sp = column(radius, height, voxel)
    mat = MPMMaterial(density=1500.0, friction=friction)
    emitter = PointsEmitter(positions=p, mass=sp ** 3 * mat.density, radius=0.5 * sp, material=mat)
    cfg = MPMConfig(voxel_size=voxel)
    dt_sub = 1.0 / fps / substeps
    if engine == "metalsim":
        import mujoco

        from metalsim.physics.batch import BatchSim, BatchSimOptions
        from metalsim.physics.mpm.coupled import MPMCoupling

        model = mujoco.MjModel.from_xml_string(XML.format(dt=dt_sub, floor_mu=floor_mu))
        sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
        mpm = MPMCoupling(sim, cfg)
        mpm.add_emitter(emitter)
        mpm.finalize()
        step, state = mpm.step, mpm.state
    else:
        import newton

        from metalsim.physics.mpm import newton_compat
        from metalsim.physics.mpm.solver import solver_class

        newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=cfg.chunk_graph,
                              gpu_exit=cfg.gpu_exit, rigid_deterministic=None, device_args=None,
                              metal_fast_math=cfg.metal_fast_math)
        S = solver_class()
        b = newton.ModelBuilder()
        S.register_custom_attributes(b)
        b.begin_world()
        b.add_shape_plane(width=0.0, length=0.0, body=-1,
                          cfg=newton.ModelBuilder.ShapeConfig(mu=floor_mu, density=0.0, has_shape_collision=False))
        emitter.emit(b, voxel)
        b.end_world()
        m = b.finalize()
        solver = S(m, config=cfg.solver_config(S, m.particle_count, 1))
        state = m.state()
        solver.reset(state)                             # as MPMCoupling.finalize
        dt =float(np.float64(dt_sub) * substeps)       # BatchSim's dt: timestep * substeps

        def step():
            solver.step(state, state, None, None, dt)
    steps = int(round(seconds * fps))
    ke, hashes = [], []
    for s in range(steps):
        step()
        if s % 10 == 9 or s == steps - 1:
            v = state.particle_qd.numpy()
            ke.append(float(0.5 * (v * v).sum(-1).mean()))
            hashes.append(hashlib.sha256(state.particle_q.numpy().tobytes() + v.tobytes()).hexdigest()[:16])
    q = state.particle_q.numpy()
    wp.synchronize()
    return {"engine": engine, "friction": friction, "friction_angle_deg": math.degrees(math.atan(friction)),
            "voxel": voxel, "particles": int(p.shape[0]), "seconds": seconds, "ke_per_mass_last": ke[-1],
            "at_rest": ke[-1] < ke_rest, "ke_history": ke, "hashes": hashes, "deposit": surface_angle(q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="metalsim", choices=["metalsim", "newton"])
    ap.add_argument("--friction", type=float, default=0.68)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.engine, a.friction, a.voxel, a.seconds)
    print(json.dumps({k: v for k, v in r.items() if k not in ("ke_history", "hashes")}, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
