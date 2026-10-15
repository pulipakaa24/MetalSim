"""Sphere impact on a sand bed (benchmark G4 / B4 of research_notes/sand_fidelity/PLAN_2026-10-14.md; batch 01 arm 6,
batch 02a arm 1, plan section 8.6).

Uehara et al. 2003 (arXiv:cond-mat/0301549): final penetration d = 0.14 / mu (rho_ball / rho_grain)^(1/2) D^(2/3)
H^(1/3), with H = h + d and mu = tan(draining repose angle). A MuJoCo sphere (D 0.05 m, 2500 kg/m^3, free joint) starts
just above the bed surface (1 mm gap) with the free-fall speed sqrt(2 g h) of a drop from height h. Bed 0.2 x 0.2 x
0.12 m, 8 particles per cell (2 per voxel edge); lagged exchange, 250 iterations (tolerance 1e-6, never reached: plan
section 8.1). Records the penetration of the sphere's bottom below the initial bed surface every step, the peak and final
penetration, the stopping time (first step after which |v| < 1 cm/s holds), and the particles overlapping the sphere.

Batch 02a options (defaults reproduce batch 01's a6_sphere):

* ``--walls``: frictionless static walls around the bed (batch 01's bed had none and is a free column of aspect ~1.2
  that collapses on the impact's time scale: plan 8.6, S-bed). The whole scene is then translated by half a voxel in x
  and y so that no wall face lies on a grid node (batch 01's tilting box, plan 8.3).
* ``--no-ball``: the same bed without the sphere (control: how far the bed surface at the centre subsides by itself).
* every run logs the bed surface at the centre (column tops within D/2 of the axis, outside the sphere) and in a ring
  (1 to 2 D) every 4 steps.
* ``--kinematic SPEED``: the sphere is a mocap body (kinematic proxy, ``collider_velocity_mode="backward"`` so the
  sand sees its finite-difference velocity) pushed down at SPEED [m/s] from 1 mm above the surface to ``--kin-depth``;
  records the sand's vertical force on it against depth (S-coupling: quasi-static bed strength without the dynamic
  exchange).
* ``--ball-density``, ``--diameter``, ``--bed LX LY LZ``, ``--fps``, ``--voxel``, ``--log-iterations``.

Exit code 3 if anything is non-finite.

    python sphere_drop.py [--h 0.2] [--fps 480] [--voxel 0.005] [--walls] [--no-ball] [--kinematic 0.02] [--json out]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
import numpy as np  # noqa: E402
import sand_metrics as SM  # noqa: E402

XML = """<mujoco><option timestep="{dt!r}" integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" friction="0.5"/>
<body name="ball" pos="0 0 {z}"><freejoint/><geom name="ball" type="sphere" size="{r}" mass="{m}" friction="0.5"/>
</body></worldbody></mujoco>"""


def scene_xml(dt, r, mass, z0, ball=True, kinematic=False, walls=None, ox=0.0, oy=0.0):
    """``walls``: None or the bed (lx, ly, lz): four frictionless static boxes around it, 1.5 lz tall."""
    parts = ['<geom name="floor" type="plane" size="0 0 1" friction="0.5"/>']
    if walls is not None:
        lx, ly, lz = walls
        t, hz = 0.01, 0.75 * lz
        nf = 'contype="0" conaffinity="0" friction="0 0.005 0.0001"'
        parts += [
            f'<geom name="wall_xlo" type="box" size="{t} {0.5 * ly + 2 * t} {hz}" pos="{ox - 0.5 * lx - t} {oy} {hz}" {nf}/>',
            f'<geom name="wall_xhi" type="box" size="{t} {0.5 * ly + 2 * t} {hz}" pos="{ox + 0.5 * lx + t} {oy} {hz}" {nf}/>',
            f'<geom name="wall_ylo" type="box" size="{0.5 * lx + 2 * t} {t} {hz}" pos="{ox} {oy - 0.5 * ly - t} {hz}" {nf}/>',
            f'<geom name="wall_yhi" type="box" size="{0.5 * lx + 2 * t} {t} {hz}" pos="{ox} {oy + 0.5 * ly + t} {hz}" {nf}/>']
    if ball and kinematic:
        parts.append(f'<body name="ball" mocap="true" pos="{ox} {oy} {z0}"><geom name="ball" type="sphere" size="{r}" '
                     f'contype="0" conaffinity="0" friction="0.5"/></body>')
    elif ball:
        parts.append(f'<body name="ball" pos="{ox} {oy} {z0}"><freejoint/><geom name="ball" type="sphere" size="{r}" '
                     f'mass="{mass}" friction="0.5"/></body>')
    return (f'<mujoco><option timestep="{dt!r}" integrator="implicitfast"/><worldbody>' + "".join(parts)
            + "</worldbody></mujoco>")


def surface_probe(q, ox, oy, r, voxel):
    """Mean column top within r of the axis and in the ring [2r, 4r] (column width two particle spacings)."""
    col = 2.0 * voxel / 2.0
    cxy, top, _ = SM.column_tops(q, col)
    d = np.hypot(cxy[:, 0] - ox, cxy[:, 1] - oy)
    c = d < r
    ring = (d >= 2 * r) & (d <= 4 * r)
    return (float(np.mean(top[c])) if c.any() else float("nan"), float(np.mean(top[ring])) if ring.any() else float("nan"))


def run(h=0.2, fps=480, voxel=0.005, diameter=0.05, ball_density=2500.0, bed=(0.2, 0.2, 0.12), ppc_axis=2.0,
        friction=0.7, iterations=250, tolerance=1e-6, seconds=0.5, substeps=3, mode="lagged", walls=False,
        ball=True, kinematic=None, kin_depth=0.06, log_iterations=False, solver_variant="parity",
        strain_basis="P0"):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    r = 0.5 * diameter
    mass = ball_density * math.pi / 6.0 * diameter ** 3
    depth = bed[2]
    v0 = math.sqrt(2 * 9.81 * h)
    ox = oy = 0.5 * voxel if walls else 0.0
    z0 = depth + r + 0.001
    dt = 1.0 / fps / substeps
    if walls or not ball or kinematic:
        xml = scene_xml(dt, r, mass, z0, ball=ball, kinematic=bool(kinematic), walls=bed if walls else None, ox=ox, oy=oy)
    else:
        xml = XML.format(dt=dt, z=z0, r=r, m=mass)          # batch 01's scene, byte for byte
    model = mujoco.MjModel.from_xml_string(xml)
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    extra = {}
    if kinematic:
        extra.update(proxy_mass="kinematic", collider_velocity_mode="backward")
    if log_iterations:
        extra["capture"] = False
    if solver_variant != "parity":
        extra["solver_variant"] = solver_variant
    if strain_basis != "P0":
        extra["strain_basis"] = strain_basis
    cfg = MPMConfig(voxel_size=voxel, max_iterations=iterations, tolerance=tolerance, mode=mode, **extra)
    mpm = MPMCoupling(sim, cfg)
    mpm.add_emitter(GridEmitter(lower=(ox - 0.5 * bed[0], oy - 0.5 * bed[1], 0.0),
                                upper=(ox + 0.5 * bed[0], oy + 0.5 * bed[1], depth),
                                particles_per_cell=ppc_axis, material=MPMMaterial(density=1500.0, friction=friction)))
    if walls or not ball or kinematic:
        # the walls and the mocap ball do not collide in MuJoCo (contype 0), so name every geom explicitly
        mpm.set_colliders(geoms=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)])
    mpm.finalize()
    logs = []
    if log_iterations:
        import solve_log

        solve_log.enable(cfg)
    if ball and not kinematic:
        qvel = sim.d.qvel.numpy()
        qvel[0, :] = 0.0
        qvel[0, 2] = -v0
        sim.d.qvel.assign(qvel)
    if kinematic:
        seconds = (kin_depth + 0.001) / kinematic
    n = int(round(seconds * fps))
    zs, vz, fz, surf_t, surf_c, surf_ring = [], [], [], [], [], []
    for s in range(n):                                     # bounded
        if kinematic:
            sim.d.mocap_pos.assign(np.array([[[ox, oy, z0 - kinematic * (s + 1) / fps]]], np.float32))
        mpm.step()
        if log_iterations:
            logs.append(solve_log.drain())
        if ball:
            z = float(sim.d.mocap_pos.numpy()[0, 0, 2]) if kinematic else float(sim.d.xpos.numpy()[0, 1, 2])
            zs.append(z)
            vz.append(-kinematic if kinematic else float(sim.d.qvel.numpy()[0, 2]))
            fz.append(float(mpm.wrenches()[0, 0, 2]))
            if not np.isfinite(zs[-1]) or not np.isfinite(fz[-1]):
                print(json.dumps({"error": "non-finite ball state", "step": s}))
                sys.exit(3)
        if s % 4 == 3 or s == n - 1:
            q = mpm.particle_q.numpy()
            if not np.isfinite(q).all():
                print(json.dumps({"error": "non-finite particles", "step": s}))
                sys.exit(3)
            c, ring = surface_probe(q, ox, oy, r, voxel)
            surf_t.append((s + 1) / fps)
            surf_c.append(c)
            surf_ring.append(ring)
    wp.synchronize()
    q = mpm.particle_q.numpy()
    if not np.isfinite(q).all():
        print(json.dumps({"error": "non-finite particles"}))
        sys.exit(3)
    top0 = depth - 0.5 * (depth / math.ceil(ppc_axis * depth / voxel))   # top particle centre at t = 0
    out = {"probe": "sphere_drop", "h": h, "v_impact": v0, "fps": fps, "dt": 1.0 / fps, "voxel": voxel,
           "diameter": diameter, "diameter_cells": diameter / voxel, "ball_mass": mass, "ball_density": ball_density,
           "bed": list(bed), "ppc": ppc_axis ** 3, "friction": friction, "iterations": iterations,
           "tolerance": tolerance, "mode": mode, "particles": mpm.particles_per_world, "walls": walls, "ball": ball,
           "kinematic": kinematic, "offset": [ox, oy], "solver_variant": solver_variant, "strain_basis": strain_basis,
           "surface": {"t": surf_t, "centre": surf_c, "ring": surf_ring, "top0": top0},
           "centre_subsidence_m": float(top0 - surf_c[-1]) if surf_c else None,
           "centre_subsidence_0p2s_m": float(top0 - np.interp(0.2, surf_t, surf_c)) if surf_c else None,
           "uehara_mu0.7_m": SM.uehara_solve(0.7, diameter, h, ball_density / 2500.0),
           "solves": solve_log.summary(logs, iterations) if log_iterations else None}
    if not ball:
        return out
    pen = depth - (np.asarray(zs) - r)
    out.update(pen=pen.tolist(), vz=vz, fz=fz)
    if kinematic:
        # force against depth, binned every millimetre (the quasi-static resistance curve)
        bins = np.arange(0.0, kin_depth + 1e-9, 0.001)
        k = np.digitize(pen, bins)
        f = np.asarray(fz)
        curve = [(float(bins[i - 1]), float(np.mean(f[k == i]))) for i in range(1, len(bins)) if (k == i).any()]
        out.update(force_depth=curve, force_at_16mm_N=float(np.interp(0.016, pen, f)),
                   force_at_30mm_N=float(np.interp(0.030, pen, f)))
        return out
    speed = np.abs(np.asarray(vz))
    moving = np.flatnonzero(speed >= 0.01)
    k_stop = int(moving[-1]) + 1 if moving.size else 0
    stopped = k_stop < n
    tail = slice(int(0.8 * n), n)
    out.update(peak_penetration_m=float(pen.max()), final_penetration_m=float(pen[-1]), stopped=bool(stopped),
               t_stop_s=(k_stop + 1) / fps if stopped else None,
               creep_m_per_s=float(-(pen[tail][0] - pen[tail][-1]) / max(len(pen[tail]) - 1, 1) * fps),
               reached_floor=bool((np.asarray(zs) - r).min() < 0.003),
               overlap=SM.sphere_overlap(q, sim.d.xpos.numpy()[0, 1].astype(np.float64), r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=480)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--diameter", type=float, default=0.05)
    ap.add_argument("--ball-density", type=float, default=2500.0)
    ap.add_argument("--bed", type=float, nargs=3, default=(0.2, 0.2, 0.12))
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--walls", action="store_true")
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--kinematic", type=float, default=None, help="push the sphere down at this speed [m/s]")
    ap.add_argument("--kin-depth", type=float, default=0.06)
    ap.add_argument("--log-iterations", action="store_true")
    ap.add_argument("--solver-variant", default="parity")
    ap.add_argument("--strain-basis", default="P0")
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--ppc-axis", type=float, default=2.0, help="particles per voxel edge (2: 8 per cell, 3: 27)")
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.h, a.fps, a.voxel, a.diameter, a.ball_density, tuple(a.bed), friction=a.friction,
                seconds=a.seconds, walls=a.walls, ball=not a.no_ball, kinematic=a.kinematic, kin_depth=a.kin_depth,
                log_iterations=a.log_iterations, solver_variant=a.solver_variant,
                strain_basis=a.strain_basis, iterations=a.iterations, ppc_axis=a.ppc_axis)
    print(json.dumps({k: v for k, v in r.items() if k not in ("pen", "vz", "fz", "surface", "force_depth")}))
    if a.json:
        json.dump(r, open(a.json, "w"))


if __name__ == "__main__":
    main()
