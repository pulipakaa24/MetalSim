"""Does a sand bed carry a box? Drop and rest tests in MetalSim's main simulator (BatchSim + MPMCoupling).

Expected physics (research_notes/MPM in main simulator/bearing_and_sinking.md): the bed's bearing capacity is 5-10x the
2 kPa a 2 kg, 0.1 m box exerts (mpm:friction 0.7 ~ phi 30-35 deg); a box dropped from 0.15 m (1.7 m/s) bottoms out
after 1-2.5 cm and stops; a box placed at rest settles < 2 mm; the bed fails (box sinks) only below mu ~ 0.3-0.45.
batch_02 measured a steady sink at 0.082 m/s = g dt in "staggered" exchange whatever the voxel, iterations or friction:
the coupling lag, not the bed. This probe records, per run: box bottom height and velocity every step, the sand force
on the box, the peak penetration, and the creep speed over the last 0.3 s.

    python bearing.py [--mode lagged|staggered] [--drop 0.15 | --rest] [--fps 120] [--voxel 0.02] [--iterations 50]
                      [--tolerance 1e-4] [--basis S2] [--depth 0.08] [--friction 0.7] [--seconds 1.5] [--json out]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

XML = """<mujoco><option timestep="{dt!r}" integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" friction="0.5"/>
<body name="box" pos="0 0 {z}"><freejoint/><geom name="box" type="box" size="{h} {h} {h}" mass="{m}"
 friction="0.5"/></body></worldbody></mujoco>"""


def run(mode="lagged", drop=0.15, rest=False, fps=120, voxel=0.02, iterations=50, tolerance=1e-4, basis="S2",
        depth=0.08, friction=0.7, seconds=1.5, box_half=0.05, box_mass=2.0, solver="auto", substeps=3):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    gap = 0.001 if rest else drop
    z0 = depth + box_half + gap
    model = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / fps / substeps, z=z0, h=box_half, m=box_mass))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, tolerance=tolerance, solver=solver,
                                     collider_basis=basis, mode=mode))
    mpm.add_emitter(GridEmitter(lower=(-0.15, -0.15, 0.0), upper=(0.15, 0.15, depth),
                                material=MPMMaterial(density=1500.0, friction=friction)))
    mpm.finalize()
    n = int(round(seconds * fps))
    zs, vz, fz = [], [], []
    for _ in range(n):
        mpm.step()
        zs.append(float(sim.d.xpos.numpy()[0, 1, 2]))
        vz.append(float(sim.d.qvel.numpy()[0, 2]))
        fz.append(float(mpm.wrenches()[0, 0, 2]))
    wp.synchronize()
    bottom = np.asarray(zs) - box_half
    pen = depth - bottom                                   # penetration below the initial bed surface
    tail = slice(int(0.8 * n), n)
    creep = float(-(bottom[tail][-1] - bottom[tail][0]) / (len(bottom[tail]) - 1) * fps)
    w = box_mass * 9.81
    return {"mode": mode, "rest": rest, "drop": drop, "fps": fps, "voxel": voxel, "iterations": iterations,
            "tolerance": tolerance, "basis": basis, "depth": depth, "friction": friction, "seconds": seconds,
            "particles": mpm.particles_per_world, "peak_penetration_m": float(pen.max()),
            "final_penetration_m": float(pen[-1]), "creep_m_per_s": creep, "reached_floor": bool(bottom.min() < 0.003),
            "force_over_weight_tail": float(np.mean(fz[tail.start:]) / w), "weight_N": w,
            "bottom": bottom.tolist(), "vz": vz, "fz": fz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lagged")
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--rest", action="store_true")
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--basis", default="S2")
    ap.add_argument("--depth", type=float, default=0.08)
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.mode, a.drop, a.rest, a.fps, a.voxel, a.iterations, a.tolerance, a.basis, a.depth, a.friction,
                a.seconds)
    print(json.dumps({k: v for k, v in r.items() if k not in ("bottom", "vz", "fz")}))
    if a.json:
        json.dump(r, open(a.json, "w"))


if __name__ == "__main__":
    main()
