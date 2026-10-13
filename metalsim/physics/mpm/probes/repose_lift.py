"""Angle of repose by the lifted-cylinder test, in MetalSim's main simulator.

Sand fills a tube standing on a frictional floor; the tube (24 thin boxes on one body with a slide joint, driven by a
velocity actuator, gravity-compensated, frictionless towards the sand, a kinematic MPM collider) rises slowly and the
sand flows out into a heap. Once at rest the heap's flank angle is fitted as in ``pile_repose.surface_angle``. A slow
lift is the standard quasi-static repose test (cf. the granular-heap literature); in a continuum Drucker-Prager
model the heap angle tends to the internal friction angle atan(mu) as the resolution increases.

    python repose_lift.py [--friction 0.68] [--voxel 0.01] [--lift-speed 0.05] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pile_repose import column, surface_angle  # noqa: E402

N_STAVES = 24


def tube_xml(radius, height, floor_mu, dt):
    t = 0.004
    half_w = radius * math.tan(math.pi / N_STAVES) * 1.15
    staves = []
    for k in range(N_STAVES):
        a = 2 * math.pi * k / N_STAVES
        c = radius + t
        staves.append(f'<geom name="stave{k}" type="box" size="{t} {half_w:.6f} {0.5 * height}" '
                      f'pos="{c * math.cos(a):.6f} {c * math.sin(a):.6f} {0.5 * height + 0.001}" '
                      f'euler="0 0 {math.degrees(a):.6f}" contype="0" conaffinity="0" friction="0 0.005 0.0001"/>')
    return f"""<mujoco><option timestep="{dt!r}"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" friction="{floor_mu}"/>
<body name="tube" gravcomp="1"><joint name="lift" type="slide" axis="0 0 1"/><inertial pos="0 0 0" mass="1"
 diaginertia="0.01 0.01 0.01"/>{''.join(staves)}</body></worldbody>
<actuator><velocity name="lift" joint="lift" kv="2000"/></actuator></mujoco>"""


def run(friction=0.68, voxel=0.01, radius=0.05, fill=0.08, lift_speed=0.05, settle=0.3, rest=1.5, floor_mu=0.7,
        fps=120, substeps=2, iterations=250):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import MPMConfig, MPMCoupling, MPMMaterial, PointsEmitter

    height = fill + 0.03
    model = mujoco.MjModel.from_xml_string(tube_xml(radius, height, floor_mu, 1.0 / fps / substeps))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    p, sp = column(radius - 0.3 * voxel, fill, voxel)
    mat = MPMMaterial(density=1500.0, friction=friction)
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, proxy_mass="kinematic"))
    mpm.add_emitter(PointsEmitter(positions=p, mass=sp ** 3 * mat.density, radius=0.5 * sp, material=mat))
    geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ["floor"] + [f"stave{k}" for k in range(N_STAVES)]]
    mpm.set_colliders(geoms=geoms)
    mpm.finalize()
    lift_time = (height + 0.05) / lift_speed
    steps = int(round((settle + lift_time + rest) * fps))
    ke = []
    for s in range(steps):
        t = s / fps
        v = lift_speed if settle <= t < settle + lift_time else 0.0
        sim.d.ctrl.assign(np.full((1, 1), v, np.float32))
        mpm.step()
        if s % 20 == 19 or s == steps - 1:
            qd = mpm.particle_qd.numpy()
            ke.append(float(0.5 * (qd * qd).sum(-1).mean()))
    q = mpm.particle_q.numpy()
    wp.synchronize()
    dep = surface_angle(q)
    return {"friction": friction, "friction_angle_deg": math.degrees(math.atan(friction)), "voxel": voxel,
            "particles": int(p.shape[0]), "lift_speed": lift_speed, "tube_z_final": float(sim.d.qpos.numpy()[0, 0]),
            "ke_per_mass_last": ke[-1], "at_rest": ke[-1] < 1e-5, "ke_history": ke, "heap": dep,
            "angle_over_friction_angle": dep["angle_deg"] / math.degrees(math.atan(friction))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", type=float, default=0.68)
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--lift-speed", type=float, default=0.05)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.friction, a.voxel, lift_speed=a.lift_speed)
    print(json.dumps({k: v for k, v in r.items() if k != "ke_history"}, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
