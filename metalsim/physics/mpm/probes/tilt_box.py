"""Tilting-box rheology test (benchmark G1 / B1 of research_notes/sand_fidelity/PLAN_2026-10-14.md; sand_fidelity
batch 01, arm 2): a sand layer in a closed, quasi-2D box is tilted slowly until it avalanches. Records the tilt at
failure (theta_fail: first downslope surface speed above 1 mm/s sustained for 0.25 s) and the free-surface angle to the
horizontal once the avalanche has stopped (theta_stop, the post-avalanche angle).

Why it discriminates (plan section 2, R3): in a simple-shear flow the coaxial non-associated Drucker-Prager rule gives
deviatoric stress with zero normal components, so sqrt(J2) = tau and p = sigma_n, and a layer flowing in plane shear
must stop at tan(theta) = mu: theta_stop = atan(mu) if the solver's rheology is right. The lifted-cylinder heap of the
same material stands 7-9 deg lower. theta_stop ~ atan(mu) with a low heap points at the heap (measurement R1, stress
geometry R3); a theta_stop as low as the heap points at the discretization (R2) or the solver (R4).
theta_fail is reported against the plan's criterion (atan(mu) +- 1 deg), but a rigid-perfectly-plastic layer's onset is
not unique: before failure its stress is whatever the solver's warm-started iteration leaves inside the cone, and
statically admissible states exist above atan(mu) (with sigma_xx = sigma_yy free, up to tan(theta) = mu / sqrt(1 - 4
mu^2 / 3): 24.3 deg for mu 0.4, 47.7 deg for 0.68, none for 0.9). So theta_fail > atan(mu) + 1 deg alone does not
falsify the rheology; theta_stop does.

Setup: box 0.30 x 0.04 m (x along the slope), sand 0.05 m deep (``--depth``), rough floor (friction 1.0, a plane),
frictionless side and end walls (static MuJoCo boxes as MPM colliders). The box does not move: the tilt is applied by
rotating gravity in the x-z plane, g = 9.81 (sin theta, 0, -cos theta) (``Model.set_gravity``, a device array the
solver reads every step, so captured graphs see it). For a box rotating at omega = 2 deg/s the fictitious forces
neglected by this are omega^2 r ~ 2e-4 m/s^2 (centrifugal) and 2 omega v (Coriolis, zero before failure): < 2e-5 g.
Protocol: settle 0.3 s level; tilt at 20 deg/s to theta0 = atan(mu) - 6 deg; tilt at 2 deg/s until failure or
theta_max = min(atan(mu) + 14 deg, 60 deg); then hold the tilt for ``--hold`` s and measure. Solve: 250 iterations,
tolerance 1e-6 (converged). Isaac's 1e-4 leaves a creep (bearing3: 0.4 mm/s resting, 16 mm/s at 50 iterations) of the
order of the 1 mm/s failure threshold, which would trigger a false theta_fail; the heap is insensitive to the solve
(repose3: 4x the iterations +0.3 deg), so B1 at 1e-6 and the heap at 1e-4 remain comparable.

Exit code 3 if any particle is non-finite.

    python tilt_box.py [--friction 0.68] [--voxel 0.005] [--json out.json]
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

L, W, WALL_H = 0.30, 0.04, 0.14
G = 9.81


def box_xml(dt):
    t = 0.01
    hl, hw = 0.5 * L, 0.5 * W
    nofric = 'contype="0" conaffinity="0" friction="0 0.005 0.0001"'
    return f"""<mujoco><option timestep="{dt!r}"/><worldbody>
<geom name="floor" type="plane" size="0 0 1" friction="1.0" contype="0" conaffinity="0"/>
<geom name="side_lo" type="box" size="{hl + 2 * t} {t} {0.5 * WALL_H}" pos="0 {-(hw + t)} {0.5 * WALL_H}" {nofric}/>
<geom name="side_hi" type="box" size="{hl + 2 * t} {t} {0.5 * WALL_H}" pos="0 {hw + t} {0.5 * WALL_H}" {nofric}/>
<geom name="end_lo" type="box" size="{t} {hw + 2 * t} {0.5 * WALL_H}" pos="{-(hl + t)} 0 {0.5 * WALL_H}" {nofric}/>
<geom name="end_hi" type="box" size="{t} {hw + 2 * t} {0.5 * WALL_H}" pos="{hl + t} 0 {0.5 * WALL_H}" {nofric}/>
</worldbody></mujoco>"""


def run(friction=0.68, voxel=0.005, depth=0.05, rate=2.0, hold=4.0, fps=120, substeps=2, iterations=250,
        tolerance=1e-6, threshold=1e-3, window=0.25, sample_every=3):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    model = mujoco.MjModel.from_xml_string(box_xml(1.0 / fps / substeps))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, tolerance=tolerance))
    mat = MPMMaterial(density=1500.0, friction=friction)
    mpm.add_emitter(GridEmitter(lower=(-0.5 * L, -0.5 * W, 0.0), upper=(0.5 * L, 0.5 * W, depth), material=mat))
    mpm.set_colliders(geoms=["floor", "side_lo", "side_hi", "end_lo", "end_hi"])
    mpm.finalize()
    sp = voxel / 3.0
    col = 2.0 * sp
    x_lo, x_hi = -0.5 * L + 3 * voxel, 0.5 * L - 3 * voxel     # away from the end walls
    atan_mu = math.degrees(math.atan(friction))
    theta0 = max(atan_mu - 6.0, 0.0)
    theta_max = min(atan_mu + 14.0, 60.0)
    settle, fast = 0.3, 20.0
    t_fast_end = settle + theta0 / fast
    t_slow_end = t_fast_end + (theta_max - theta0) / rate
    n_max = int(math.ceil((t_slow_end + hold) * fps)) + 1
    hist_t, hist_theta, hist_v = [], [], []
    theta_hold = None
    fail_idx = None
    k_slow = None
    n_end = n_max
    theta = 0.0
    for s in range(n_max):                                 # bounded: at most the full tilt plus the hold
        t = s / fps
        if theta_hold is None:
            if t < settle:
                theta = 0.0
            elif t < t_fast_end:
                theta = fast * (t - settle)
            else:
                theta = min(theta0 + rate * (t - t_fast_end), theta_max)
        else:
            theta = theta_hold
        th = math.radians(theta)
        mpm.model.set_gravity((G * math.sin(th), 0.0, -G * math.cos(th)))
        mpm.step()
        if s % sample_every == sample_every - 1:
            q = mpm.particle_q.numpy()
            v = mpm.particle_qd.numpy()
            if not (np.isfinite(q).all() and np.isfinite(v).all()):
                print(json.dumps({"error": "non-finite particles", "step": s, "theta": theta}))
                sys.exit(3)
            hist_t.append(t)
            hist_theta.append(theta)
            hist_v.append(SM.surface_speed(q, v, x_lo, x_hi, col, voxel))
            if theta_hold is None and t >= t_fast_end:
                if k_slow is None:
                    k_slow = len(hist_t) - 1                # first sample of the slow tilt
                k = SM.first_sustained(np.asarray(hist_t[k_slow:]), np.asarray(hist_v[k_slow:]), threshold, window)
                if k is not None:
                    fail_idx = k_slow + k
                    theta_hold = theta                      # stop tilting: the avalanche runs at this tilt
                    n_end = min(n_max, s + 1 + int(round(hold * fps)))
                elif theta >= theta_max and t >= t_slow_end:
                    theta_hold = theta_max                  # no failure up to theta_max: hold and measure anyway
                    n_end = min(n_max, s + 1 + int(round(hold * fps)))
        if s + 1 >= n_end:
            break
    wp.synchronize()
    q = mpm.particle_q.numpy()
    v = mpm.particle_qd.numpy()
    if not (np.isfinite(q).all() and np.isfinite(v).all()):
        print(json.dumps({"error": "non-finite particles at the end"}))
        sys.exit(3)
    final_theta = theta_hold if theta_hold is not None else theta
    ang, a_box, ncol = SM.surface_angle_world(q, final_theta, x_lo, x_hi, col, 2 * voxel)
    speed = np.linalg.norm(v, axis=1)
    k0 = k_slow or 0
    pre = np.asarray(hist_v[k0:fail_idx]) if fail_idx is not None else np.asarray(hist_v[k0:])
    return {"probe": "tilt_box", "friction": friction, "atan_mu_deg": atan_mu, "voxel": voxel, "depth": depth,
            "depth_cells": depth / voxel, "particles": mpm.particles_per_world, "rate_deg_s": rate,
            "iterations": iterations, "tolerance": tolerance, "theta0": theta0, "theta_max": theta_max,
            "failed": fail_idx is not None, "theta_fail_deg": hist_theta[fail_idx] if fail_idx is not None else None,
            "theta_hold_deg": final_theta, "theta_stop_deg": ang, "surface_angle_box_deg": a_box,
            "surface_columns": ncol, "max_speed_end": float(speed.max()),
            "surface_speed_p90_end": SM.surface_speed(q, v, x_lo, x_hi, col, voxel),
            "at_rest": bool(np.quantile(speed, 0.99) < 1e-3), "pre_failure_surface_speed_max": float(pre.max()) if pre.size else None,
            "seconds": n_end / fps, "history": {"t": hist_t, "theta": hist_theta, "surface_speed_p90": hist_v}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", type=float, default=0.68)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--depth", type=float, default=0.05)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.friction, a.voxel, a.depth, a.rate, a.hold, iterations=a.iterations, tolerance=a.tolerance)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
