"""The coupled main simulator at batch scale: a box dropped on a small sand bed in N worlds (BatchSim + MPMCoupling).

Repeatability (``--runs 2``): the scene is built and run twice in one process; every step's box state, sand wrench
and particle state are hashed and compared. Timing (``--time``): median wall time per captured coupled step
(synchronized each step) after ``--warmup`` steps; run it as a timing-class job on the 16-inch only.

    python scale_coupled.py --worlds 64 --steps 60 --runs 2 [--solver auto|jacobi] [--json out.json]
    python scale_coupled.py --worlds 1024 --steps 40 --time --warmup 10 --json out.json
    python scale_coupled.py --worlds 64 --steps 100 --split --json out.json

``--split``: the coupled step's sections as separate captured graphs (MuJoCo Warp substeps, coupling launches
before / after them, MPM substeps + harvest) and the whole step, replayed round-robin (synchronized each replay) so
that drift of the GPU clock hits every section alike; medians per section. The sections run out of order here, so
only their timings mean anything. ``--rigid-only --time``: the BatchSim step of the same scene without sand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

XML = """<mujoco><option timestep="{dt!r}" integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" friction="0.5"/>
<body name="box" pos="0 0 0.15"><freejoint/><geom name="box" type="box" size="0.03 0.03 0.03" mass="0.5"/></body>
</worldbody></mujoco>"""


def build(a):
    import mujoco

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import SAND, GridEmitter, MPMConfig, MPMCoupling

    model = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / 120.0 / 3))
    sim = BatchSim(model, a.worlds, options=BatchSimOptions(substeps=3, deterministic=bool(getattr(a, "deterministic", 0))))
    if a.rigid_only:
        return sim, None
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=a.voxel, max_iterations=a.iterations, solver=a.solver,
                                     max_active_cell_count=a.cells_per_world * a.worlds,
                                     capture=bool(a.capture)))
    mpm.add_emitter(GridEmitter(lower=(-0.06, -0.06, 0.0), upper=(0.06, 0.06, 0.05), material=SAND))
    mpm.finalize()
    return sim, mpm


def run_hashes(a):
    sim, mpm = build(a)
    out = []

    def hx(*arrs):
        h = hashlib.sha256()
        for arr in arrs:
            h.update(arr.numpy().tobytes())
        return h.hexdigest()[:16]

    for _ in range(a.steps):
        if mpm is None:
            sim.step()
            out.append({"rigid": hx(sim.d.qpos, sim.d.qvel)})
        else:
            mpm.step()
            out.append({"rigid": hx(sim.d.qpos, sim.d.qvel), "wrench": hx(mpm.coupling_f),
                        "particles": hx(mpm.particle_q, mpm.particle_qd)})
    z = sim.d.xpos.numpy()[:, 1, 2]
    info = {"box_z_min": float(z.min()), "box_z_max": float(z.max()), "finite": True}
    if mpm is not None:
        mpm.check_status()
        info.update(particles_per_world=mpm.particles_per_world, finite=bool(np.isfinite(mpm.particle_q.numpy()).all()))
    return out, info


def run_time(a):
    sim, mpm = build(a)
    step = sim.step if mpm is None else mpm.step
    ts = []
    for s in range(a.warmup + a.steps):
        sim.synchronize()
        t0 = time.perf_counter()
        step()
        sim.synchronize()
        if s >= a.warmup:
            ts.append(time.perf_counter() - t0)
    if mpm is not None:
        mpm.check_status()
    ts = np.asarray(ts) * 1e3
    return {"ms_per_step_median": float(np.median(ts)), "ms_p10": float(np.percentile(ts, 10)),
            "ms_p90": float(np.percentile(ts, 90)),
            "particles_per_world": 0 if mpm is None else mpm.particles_per_world,
            "env_steps_per_s": float(a.worlds / (np.median(ts) / 1e3))}


def run_split(a):
    import warp as wp

    sim, mpm = build(a)
    for _ in range(a.warmup):
        mpm.step()
    sim.synchronize()
    sections = {"rigid": sim.launch_step, "coupling_before": mpm._launch_before_rigid,
                "coupling_after": mpm._launch_after_rigid, "mpm": mpm._launch_mpm, "full": mpm._launch_coupled_step}
    graphs = {}
    for name, fn in sections.items():
        with wp.ScopedCapture(device=sim.device, force_module_load=False) as cap:
            fn()
        graphs[name] = cap.graph
    ts = {k: [] for k in graphs}
    for _ in range(a.steps):
        for name, g in graphs.items():
            sim.synchronize()
            t0 = time.perf_counter()
            wp.capture_launch(g)
            sim.synchronize()
            ts[name].append((time.perf_counter() - t0) * 1e3)
    mpm.check_status()
    out = {f"{k}_ms_median": float(np.median(v)) for k, v in ts.items()}
    out.update({f"{k}_ms_p10": float(np.percentile(v, 10)) for k, v in ts.items()})
    out.update({f"{k}_ms_p90": float(np.percentile(v, 90)) for k, v in ts.items()})
    coupling = out["coupling_before_ms_median"] + out["coupling_after_ms_median"]
    out["coupling_ms_median"] = coupling
    out["coupling_share_of_full"] = coupling / out["full_ms_median"]
    out["particles_per_world"] = mpm.particles_per_world
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--capture", type=int, default=1)
    ap.add_argument("--cells-per-world", type=int, default=512, help="sparse-grid cell reservation per world")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--deterministic", type=int, default=0, help="BatchSimOptions(deterministic=...)")
    ap.add_argument("--rigid-only", action="store_true", help="the box on the floor without sand (BatchSim only)")
    ap.add_argument("--split", action="store_true", help="per-section captured timings (see the module docstring)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    res = {"worlds": a.worlds, "solver": a.solver, "steps": a.steps, "capture": a.capture,
           "deterministic": a.deterministic, "rigid_only": a.rigid_only}
    with wp.ScopedDevice("metal:0"):
        if a.split:
            res.update(run_split(a))
        elif a.time:
            res.update(run_time(a))
        else:
            runs = [run_hashes(a) for _ in range(a.runs)]
            h0 = runs[0][0]
            res["first_differing_step"] = [next((i for i in range(a.steps) if h[i] != h0[i]), None)
                                           for h, _ in runs[1:]]
            res["first_differing_by_part"] = [{k: next((i for i in range(a.steps) if h[i][k] != h0[i][k]), None)
                                               for k in h0[0]} for h, _ in runs[1:]]
            res["repeatable"] = all(d is None for d in res["first_differing_step"])
            res.update(runs[0][1])
    print(json.dumps(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if not a.time and not a.split and not res["repeatable"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
