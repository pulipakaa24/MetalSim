"""Angle of repose of a granular pile in MetalSim's main simulator (BatchSim floor + MPMCoupling).

An axisymmetric column of sand (radius R, height H) released on a frictional MuJoCo floor slumps into a cone; once at
rest (kinetic energy below a threshold) the flank angle is fitted from the free-surface height profile h(r) (95th
percentile of particle heights per radial bin, linear fit where 0.2 h_max < h < 0.8 h_max).

    .venv/bin/python pile_repose.py [--friction 0.68] [--voxel 0.02] [--seconds 2.0] [--worlds 1] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

XML = """<mujoco><option timestep="{dt!r}"/><worldbody>
<geom name="floor" type="plane" size="0 0 1" friction="{floor_mu}"/></worldbody></mujoco>"""


def surface_angle(q: np.ndarray, bins: int = 24) -> dict:
    """Flank angle [deg] of a pile centred at the origin from particle positions (P, 3)."""
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
    if sel.sum() < 3:
        return {"angle_deg": float("nan"), "bins": int(sel.sum())}
    slope, icpt = np.polyfit(rc[sel], h[sel], 1)
    return {"angle_deg": float(math.degrees(math.atan(-slope))), "bins": int(sel.sum()), "h_max": float(h.max()),
            "runout": float(rc[h > 0.02 * h.max()].max())}


def run(friction=0.68, voxel=0.02, seconds=2.0, worlds=1, radius=0.08, height=0.16, floor_mu=0.7, ke_rest=1e-4,
        fps=60, substeps=2):
    import mujoco

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import MPMConfig, MPMCoupling, MPMMaterial, PointsEmitter

    model = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / fps / substeps, floor_mu=floor_mu))
    sim = BatchSim(model, worlds, options=BatchSimOptions(substeps=substeps))
    # Isaac's cell_center lattice (3 particles per voxel) inside the cylinder
    sp = voxel / 3.0
    n = int(math.ceil(radius / sp))
    g = (np.arange(-n, n) + 0.5) * sp
    x, y, zz = np.meshgrid(g, g, (np.arange(int(math.ceil(height / sp))) + 0.5) * sp, indexing="ij")
    p = np.stack([x, y, zz], -1).reshape(-1, 3)
    p = p[np.hypot(p[:, 0], p[:, 1]) <= radius]
    mat = MPMMaterial(density=1500.0, friction=friction)
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel))
    mpm.add_emitter(PointsEmitter(positions=p, mass=sp ** 3 * mat.density, radius=0.5 * sp, material=mat))
    mpm.finalize()
    steps = int(round(seconds * fps))
    ke = []
    for s in range(steps):
        mpm.step()
        if s % 10 == 9 or s == steps - 1:
            v = mpm.particle_qd.numpy()
            ke.append(float(0.5 * (v * v).sum(-1).mean()))
    q = mpm.particles()
    res = [surface_angle(q[w]) for w in range(worlds)]
    return {"friction": friction, "friction_angle_deg": math.degrees(math.atan(friction)), "voxel": voxel,
            "particles": int(p.shape[0]), "seconds": seconds, "ke_per_mass_last": ke[-1], "at_rest": ke[-1] < ke_rest,
            "ke_history": ke, "worlds": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", type=float, default=0.68)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    r = run(a.friction, a.voxel, a.seconds, a.worlds)
    print(json.dumps({k: v for k, v in r.items() if k != "ke_history"}, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
