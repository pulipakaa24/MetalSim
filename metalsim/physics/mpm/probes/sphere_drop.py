"""Sphere impact on a sand bed (benchmark G4 / B4 of research_notes/sand_fidelity/PLAN_2026-10-14.md), smoke test
for sand_fidelity batch 01 (arm 6); the Uehara sweep (h = 0.1-0.8 m) is batch 02.

Uehara et al. 2003 (arXiv:cond-mat/0301549): final penetration d = 0.14 / mu (rho_ball / rho_grain)^(1/2) D^(2/3)
H^(1/3), with H = h + d and mu = tan(draining repose angle). A MuJoCo sphere (D 0.05 m, 2500 kg/m^3, free joint) starts
just above the bed surface (1 mm gap) with the free-fall speed sqrt(2 g h) of a drop from height h. Bed 0.2 x 0.2 x
0.12 m, 8 particles per cell (2 per voxel edge); lagged exchange, converged solve (250 iterations, tolerance 1e-6).
Records the penetration of the sphere's bottom below the initial bed surface every step, the peak and final
penetration, the stopping time (first step after which |v| < 1 cm/s holds), and the particles overlapping the sphere.

Exit code 3 if anything is non-finite.

    python sphere_drop.py [--h 0.2] [--fps 480] [--voxel 0.005] [--json out]
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


def run(h=0.2, fps=480, voxel=0.005, diameter=0.05, ball_density=2500.0, bed=(0.2, 0.2, 0.12), ppc_axis=2.0,
        friction=0.7, iterations=250, tolerance=1e-6, seconds=0.5, substeps=3, mode="lagged"):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    r = 0.5 * diameter
    mass = ball_density * math.pi / 6.0 * diameter ** 3
    depth = bed[2]
    v0 = math.sqrt(2 * 9.81 * h)
    model = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / fps / substeps, z=depth + r + 0.001, r=r, m=mass))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, tolerance=tolerance, mode=mode))
    mpm.add_emitter(GridEmitter(lower=(-0.5 * bed[0], -0.5 * bed[1], 0.0), upper=(0.5 * bed[0], 0.5 * bed[1], depth),
                                particles_per_cell=ppc_axis, material=MPMMaterial(density=1500.0, friction=friction)))
    mpm.finalize()
    qvel = sim.d.qvel.numpy()
    qvel[0, :] = 0.0
    qvel[0, 2] = -v0
    sim.d.qvel.assign(qvel)
    n = int(round(seconds * fps))
    zs, vz = [], []
    for _ in range(n):                                     # bounded
        mpm.step()
        zs.append(float(sim.d.xpos.numpy()[0, 1, 2]))
        vz.append(float(sim.d.qvel.numpy()[0, 2]))
        if not np.isfinite(zs[-1]):
            print(json.dumps({"error": "non-finite ball state", "step": len(zs)}))
            sys.exit(3)
    wp.synchronize()
    q = mpm.particle_q.numpy()
    if not np.isfinite(q).all():
        print(json.dumps({"error": "non-finite particles"}))
        sys.exit(3)
    pen = depth - (np.asarray(zs) - r)
    speed = np.abs(np.asarray(vz))
    moving = np.flatnonzero(speed >= 0.01)
    k_stop = int(moving[-1]) + 1 if moving.size else 0
    stopped = k_stop < n
    d_final = float(pen[-1])
    tail = slice(int(0.8 * n), n)
    return {"probe": "sphere_drop", "h": h, "v_impact": v0, "fps": fps, "dt": 1.0 / fps, "voxel": voxel,
            "diameter": diameter, "diameter_cells": diameter / voxel, "ball_mass": mass, "ball_density": ball_density,
            "bed": list(bed), "ppc": ppc_axis ** 3, "friction": friction, "iterations": iterations,
            "tolerance": tolerance, "mode": mode, "particles": mpm.particles_per_world,
            "peak_penetration_m": float(pen.max()), "final_penetration_m": d_final, "stopped": bool(stopped),
            "t_stop_s": (k_stop + 1) / fps if stopped else None, "creep_m_per_s": float(-(pen[tail][0] - pen[tail][-1])
                                                                                          / max(len(pen[tail]) - 1, 1) * fps),
            "reached_floor": bool((np.asarray(zs) - r).min() < 0.003),
            "overlap": SM.sphere_overlap(q, sim.d.xpos.numpy()[0, 1].astype(np.float64), r),
            "uehara_mu0.7_m": SM.uehara_solve(0.7, diameter, h), "pen": pen.tolist(), "vz": vz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=480)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.h, a.fps, a.voxel, friction=a.friction, seconds=a.seconds)
    print(json.dumps({k: v for k, v in r.items() if k not in ("pen", "vz")}))
    if a.json:
        json.dump(r, open(a.json, "w"))


if __name__ == "__main__":
    main()
