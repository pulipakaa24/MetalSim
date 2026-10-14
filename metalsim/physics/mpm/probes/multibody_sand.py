"""Several free boxes on one sand bed: an MPMCoupling scene with many proxy bodies per world (for timing the proxy
wrench harvest, `SolverImplicitMPM.coupling_harvest_proxy_wrenches`, which scans the active collider nodes once per
proxy body; and the rest of the coupled step).

    python multibody_sand.py [--bodies 16] [--worlds 1] [--steps 60] [--warmup 10] [--capture 1] [--json out.json]
    python multibody_sand.py --bodies 16 --profile     # one eager step with Warp's ScopedTimer per section

Scene: a 0.6 x 0.6 x 0.06 m bed (Isaac's Franka-Pour media material: density 1,500, friction 0.7), voxel 0.02, with
``--bodies`` boxes (0.05 m, 0.3 kg) on a square grid just above the surface, dropping onto it; MuJoCo timestep
1/360 s x 3 substeps per coupled step (1/120 s), 1 MPM substep, the default (lagged) exchange.

    from metalsim.physics.mpm.probes.multibody_sand import build   # (sim, mpm) for custom timing (scripts: add the
                                                                   # probes directory to sys.path)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402


def scene_xml(bodies: int, half=0.025, mass=0.3, gap=0.004, top=0.06, dt=1.0 / 360.0) -> str:
    k = int(math.ceil(math.sqrt(bodies)))
    pitch = 0.5 / k
    out = []
    for i in range(bodies):
        x = (i % k - (k - 1) / 2) * pitch
        y = (i // k - (k - 1) / 2) * pitch
        out.append(f'<body name="box{i}" pos="{x:.4f} {y:.4f} {top + half + gap:.4f}"><freejoint/>'
                   f'<geom type="box" size="{half} {half} {half}" mass="{mass}"/></body>')
    return (f'<mujoco><option timestep="{dt!r}" integrator="implicitfast"/><worldbody>'
            f'<geom name="floor" type="plane" size="0 0 1"/>{"".join(out)}</worldbody></mujoco>')


def build(bodies=16, worlds=1, voxel=0.02, iterations=50, capture=True):
    import mujoco

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import SAND, GridEmitter, MPMConfig, MPMCoupling

    model = mujoco.MjModel.from_xml_string(scene_xml(bodies))
    sim = BatchSim(model, worlds, options=BatchSimOptions(substeps=3))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, capture=capture))
    mpm.add_emitter(GridEmitter(lower=(-0.3, -0.3, 0.0), upper=(0.3, 0.3, 0.06), material=SAND))
    mpm.finalize()
    return sim, mpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--capture", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        sim, mpm = build(a.bodies, a.worlds, capture=bool(a.capture) and not a.profile)
        for _ in range(a.warmup):
            mpm.step()
        sim.synchronize()
        if a.profile:
            with wp.ScopedTimer("coupled step", synchronize=True):
                mpm._launch_coupled_step()
            with wp.ScopedTimer("harvest only", synchronize=True):
                mpm._harvest(mpm.coupling_f)
            return
        ts = []
        for _ in range(a.steps):
            t0 = time.perf_counter()
            mpm.step()
            sim.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        mpm.check_status()
    res = {"bodies": a.bodies, "worlds": a.worlds, "particles_per_world": mpm.particles_per_world,
           "proxies_per_world": mpm.nb, "ms_per_step_median": float(np.median(ts)), "ms_p10": float(np.percentile(ts, 10)),
           "ms_p90": float(np.percentile(ts, 90)), "box_z_mean": float(sim.d.xpos.numpy()[:, 1:, 2].mean())}
    print(json.dumps(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
