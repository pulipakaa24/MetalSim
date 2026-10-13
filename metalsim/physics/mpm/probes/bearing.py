"""Does a sand bed carry a box resting on it? Sinkage of a box placed at rest on the bed (MetalSim main simulator).

Classical bearing capacity of a cohesionless bed (Terzaghi / Meyerhof, square footing B on the surface:
q_ult ~ 0.4 gamma B N_gamma, N_gamma ~ 20-45 for friction angles 30-35 deg) is about 6-25 kPa for B = 0.1 m of sand
at 1,500 kg/m^3; a 2 kg, 0.1 m box loads 2 kPa, so the bed should carry it with millimetres of settlement, not let it
sink to the floor. This probe measures the settlement against resolution, rheology iterations, bed depth and friction.

    python bearing.py [--voxel 0.02] [--iterations 50] [--depth 0.08] [--friction 0.7] [--seconds 1.0] [--json out]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

XML = """<mujoco><option timestep="{dt!r}" integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" friction="0.5"/>
<body name="box" pos="0 0 {z}"><freejoint/><geom name="box" type="box" size="{h} {h} {h}" mass="{m}"
 friction="0.5"/></body></worldbody></mujoco>"""


def run(voxel=0.02, iterations=50, depth=0.08, friction=0.7, seconds=1.0, box_half=0.05, box_mass=2.0,
        solver="auto", fps=120, substeps=3):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    z0 = depth + box_half + 0.001
    model = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / fps / substeps, z=z0, h=box_half, m=box_mass))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, solver=solver))
    mpm.add_emitter(GridEmitter(lower=(-0.15, -0.15, 0.0), upper=(0.15, 0.15, depth),
                                material=MPMMaterial(density=1500.0, friction=friction)))
    mpm.finalize()
    zs, fz = [], []
    for _ in range(int(round(seconds * fps))):
        mpm.step()
        zs.append(float(sim.d.xpos.numpy()[0, 1, 2]))
        fz.append(float(mpm.wrenches()[0, 0, 2]))
    wp.synchronize()
    zs = np.asarray(zs)
    bottom = zs - box_half
    return {"voxel": voxel, "iterations": iterations, "depth": depth, "friction": friction, "solver": solver,
            "particles": mpm.particles_per_world, "settlement_m": float(depth - bottom[-1]),
            "bottom_final_m": float(bottom[-1]), "reached_floor": bool(bottom.min() < 0.005),
            "z_history": zs[::6].tolist(), "fz_last_mean": float(np.mean(fz[-20:])), "weight_N": box_mass * 9.81}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--depth", type=float, default=0.08)
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.voxel, a.iterations, a.depth, a.friction, a.seconds, solver=a.solver)
    print(json.dumps({k: v for k, v in r.items() if k != "z_history"}, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
