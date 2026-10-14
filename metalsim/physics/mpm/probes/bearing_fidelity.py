"""Box dropped on / resting on a sand bed, for the coupling-fidelity arms of sand_fidelity batch 01 (arms 4 and 5;
research_notes/sand_fidelity/PLAN_2026-10-14.md section 3, C1 and C2).

``bearing.py``'s scene and stepping (same XML, bed, MPMConfig fields and per-step records, so with ``--project``
off it reproduces runs/mpm_main/bearing3 and bearing4 at the same settings), plus:

* ``--project``: after every coupled step, ``SolverImplicitMPM.project_outside(state, state, dt)``, the particle-level
  push-out Isaac Lab's ``MPMSolverCfg.project_outside_colliders`` runs after each MPM substep (``mpm_manager.py``
  ``_step_solver``). Isaac refuses that flag inside a coupled-solver entry (``coupler.py``: "cannot yet run inside a
  coupled-solver entry"), so with a two-way proxy this is not an Isaac configuration: it is the C2 experiment;
* the impact speed (box velocity on the step its bottom first reaches the bed surface) and the time of peak
  penetration, for the C1 check peak(dt) ~ d0 + v_impact dt;
* at the end: the particles inside the box (count, deepest overlap below its bottom face) and the heave of the bed
  surface in a ring around it, to separate grid-resolution contact (box overlaps the particles: H_grid) from real bed
  deformation (material displaced: H_bed).

Exit code 3 if a particle or the box state is non-finite.

    python bearing_fidelity.py [--rest | --drop 0.15] [--fps 120] [--voxel 0.02] [--iterations 250] [--tolerance 1e-6]
                               [--basis S2] [--project] [--seconds 1.5] [--json out]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
import numpy as np  # noqa: E402
import sand_metrics as SM  # noqa: E402
from bearing import XML  # noqa: E402


def run(mode="lagged", drop=0.15, rest=False, fps=120, voxel=0.02, iterations=250, tolerance=1e-6, basis="S2",
        depth=0.08, friction=0.7, seconds=1.5, box_half=0.05, box_mass=2.0, solver="auto", substeps=3, project=False,
        log_iterations=False):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    gap = 0.001 if rest else drop
    z0 = depth + box_half + gap
    model = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / fps / substeps, z=z0, h=box_half, m=box_mass))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    cfg = MPMConfig(voxel_size=voxel, max_iterations=iterations, tolerance=tolerance, solver=solver,
                    collider_basis=basis, mode=mode, capture=not log_iterations)
    mpm = MPMCoupling(sim, cfg)
    mpm.add_emitter(GridEmitter(lower=(-0.15, -0.15, 0.0), upper=(0.15, 0.15, depth),
                                material=MPMMaterial(density=1500.0, friction=friction)))
    mpm.finalize()
    logs = []
    if log_iterations:
        import solve_log

        solve_log.enable(cfg)
    n = int(round(seconds * fps))
    zs, vz, fz, tilt = [], [], [], []
    for _ in range(n):                                     # bounded
        mpm.step()
        if log_iterations:
            logs.append(solve_log.drain())
        if project:
            mpm.solver.project_outside(mpm.state, mpm.state, mpm.dt)
        zs.append(float(sim.d.xpos.numpy()[0, 1, 2]))
        qw = sim.d.xquat.numpy()[0, 1].astype(np.float64)
        tilt.append(float(np.degrees(np.arccos(np.clip(SM.quat_wxyz_to_mat(qw)[2, 2], -1.0, 1.0)))))
        vz.append(float(sim.d.qvel.numpy()[0, 2]))
        fz.append(float(mpm.wrenches()[0, 0, 2]))
        if not np.isfinite(zs[-1]):
            print(json.dumps({"error": "non-finite box state", "step": len(zs)}))
            sys.exit(3)
    wp.synchronize()
    q = mpm.particle_q.numpy()
    if not np.isfinite(q).all():
        print(json.dumps({"error": "non-finite particles"}))
        sys.exit(3)
    pos = sim.d.xpos.numpy()[0, 1].astype(np.float64)
    quat = sim.d.xquat.numpy()[0, 1].astype(np.float64)
    bottom = np.asarray(zs) - box_half
    pen = depth - bottom
    tail = slice(int(0.8 * n), n)
    creep = float(-(bottom[tail][-1] - bottom[tail][0]) / (len(bottom[tail]) - 1) * fps)
    w = box_mass * 9.81
    hit = np.flatnonzero(bottom <= depth)
    k_hit = int(hit[0]) if hit.size else None
    # the velocity on the step before contact (the free-fall speed reaching the bed)
    v_impact = float(-vz[k_hit - 1]) if k_hit else (float(-vz[0]) if k_hit == 0 else None)
    k_peak = int(np.argmax(pen))
    sp = voxel / 3.0
    return {"probe": "bearing_fidelity", "mode": mode, "rest": rest, "drop": drop, "fps": fps, "dt": 1.0 / fps,
            "voxel": voxel, "iterations": iterations, "tolerance": tolerance, "basis": basis, "project": project,
            "depth": depth, "friction": friction, "seconds": seconds, "particles": mpm.particles_per_world,
            "peak_penetration_m": float(pen.max()), "final_penetration_m": float(pen[-1]), "creep_m_per_s": creep,
            "reached_floor": bool(bottom.min() < 0.003), "force_over_weight_tail": float(np.mean(fz[tail.start:]) / w),
            "weight_N": w, "impact_speed_m_s": v_impact, "t_contact_s": (k_hit + 1) / fps if k_hit is not None else None,
            "t_peak_s": (k_peak + 1) / fps, "settlement_over_voxel": float(pen[-1] / voxel),
            "overlap": SM.box_overlap(q, pos, quat, box_half),
            "heave_ring_m": SM.ring_heave(q, pos[:2], 1.2 * box_half * 2 ** 0.5, 2.4 * box_half, 2 * sp, depth),
            "max_tilt_deg": float(max(tilt)), "final_tilt_deg": tilt[-1],
            "solves": solve_log.summary(logs, iterations) if log_iterations else None,
            "bottom": bottom.tolist(), "vz": vz, "fz": fz, "tilt_deg": tilt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lagged")
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--rest", action="store_true")
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--basis", default="S2")
    ap.add_argument("--depth", type=float, default=0.08)
    ap.add_argument("--friction", type=float, default=0.7)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--log-iterations", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.mode, a.drop, a.rest, a.fps, a.voxel, a.iterations, a.tolerance, a.basis, a.depth, a.friction,
                a.seconds, project=a.project, log_iterations=a.log_iterations)
    print(json.dumps({k: v for k, v in r.items() if k not in ("bottom", "vz", "fz", "tilt_deg")}))
    if a.json:
        json.dump(r, open(a.json, "w"))


if __name__ == "__main__":
    main()
