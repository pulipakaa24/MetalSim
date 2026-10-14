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
Protocol: settle 0.3 s level; tilt at 20 deg/s to theta0 = atan(mu) - 12 deg; tilt at 2 deg/s until failure or
theta_max = min(atan(mu) + 12 deg, 60 deg); then hold the tilt for ``--hold`` s and measure. Solve: 250 iterations,
tolerance 1e-6 (converged). Isaac's 1e-4 leaves a creep (bearing3: 0.4 mm/s resting, 16 mm/s at 50 iterations) of the
order of the 1 mm/s failure threshold, which would trigger a false theta_fail; the heap is insensitive to the solve
(repose3: 4x the iterations +0.3 deg), so B1 at 1e-6 and the heap at 1e-4 remain comparable.

Wall placement (``--on-node``): batch 01's first run (box faces at x = +-0.15, y = +-0.02 m, i.e. exactly on grid
nodes at voxel 0.005 / 0.01) went non-finite at mu 0.4, 0.25 s after its failure detector fired at 16.3 deg. Newton's
draft PR #4660 reports wrong collider distance signs near thin slabs and where collider normals cancel; whether that
or something else is the cause is open. By default the whole scene (walls and sand) is now translated by half a voxel
in x and y so that no wall face lies on a node (the floor stays at z = 0, on nodes, as in every probe that ran
finite); ``--on-node`` keeps the original placement, as the reproduction of the non-finite run.

Exit code 3 if any particle becomes non-finite. Before exiting, the probe writes its JSON (``--json``) with
``"error": "nonfinite"``, the history so far, and where the first non-finite particles were at the last finite sample
(distance to the walls, floor and wall edges), and saves that sample as .npz next to the JSON, so a blow-up is
localised rather than just reported.

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


def box_xml(dt, ox=0.0, oy=0.0):
    t = 0.01
    hl, hw = 0.5 * L, 0.5 * W
    nofric = 'contype="0" conaffinity="0" friction="0 0.005 0.0001"'
    return f"""<mujoco><option timestep="{dt!r}"/><worldbody>
<geom name="floor" type="plane" size="0 0 1" friction="1.0" contype="0" conaffinity="0"/>
<geom name="side_lo" type="box" size="{hl + 2 * t} {t} {0.5 * WALL_H}" pos="{ox} {oy - (hw + t)} {0.5 * WALL_H}" {nofric}/>
<geom name="side_hi" type="box" size="{hl + 2 * t} {t} {0.5 * WALL_H}" pos="{ox} {oy + hw + t} {0.5 * WALL_H}" {nofric}/>
<geom name="end_lo" type="box" size="{t} {hw + 2 * t} {0.5 * WALL_H}" pos="{ox - (hl + t)} {oy} {0.5 * WALL_H}" {nofric}/>
<geom name="end_hi" type="box" size="{t} {hw + 2 * t} {0.5 * WALL_H}" pos="{ox + hl + t} {oy} {0.5 * WALL_H}" {nofric}/>
</worldbody></mujoco>"""


def nonfinite_report(q_prev, v_prev, bad, ox, oy, voxel):
    """Where the particles that went non-finite were at the last finite sample."""
    if q_prev is None or not bad.any():
        return {"n_bad": int(bad.sum())}
    qb = q_prev[bad].astype(np.float64)
    dx = np.minimum(np.abs(qb[:, 0] - (ox - 0.5 * L)), np.abs(qb[:, 0] - (ox + 0.5 * L)))
    dy = np.minimum(np.abs(qb[:, 1] - (oy - 0.5 * W)), np.abs(qb[:, 1] - (oy + 0.5 * W)))
    dz = qb[:, 2]
    near = lambda d: d < voxel                               # noqa: E731
    sp = np.linalg.norm(v_prev, axis=1)
    k = int(np.argmax(sp))
    return {"n_bad": int(bad.sum()), "n_total": int(bad.size),
            "bbox_min": qb.min(0).tolist(), "bbox_max": qb.max(0).tolist(),
            "frac_near_end_wall": float(near(dx).mean()), "frac_near_side_wall": float(near(dy).mean()),
            "frac_near_floor": float(near(dz).mean()), "frac_near_wall_edge": float((near(dx) & near(dy)).mean()),
            "frac_near_floor_edge": float(((near(dx) | near(dy)) & near(dz)).mean()),
            "min_dist_end_wall": float(dx.min()), "min_dist_side_wall": float(dy.min()),
            "speed_bad_prev_max": float(sp[bad].max()), "speed_all_prev_max": float(sp.max()),
            "fastest_prev_position": q_prev[k].astype(float).tolist()}


def run(friction=0.68, voxel=0.005, depth=0.05, rate=2.0, hold=4.0, fps=120, substeps=2, iterations=250,
        tolerance=1e-6, threshold=1e-3, window=0.25, sample_every=3, on_node=False, json_out=None):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial

    ox = oy = 0.0 if on_node else 0.5 * voxel
    model = mujoco.MjModel.from_xml_string(box_xml(1.0 / fps / substeps, ox, oy))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, tolerance=tolerance))
    mat = MPMMaterial(density=1500.0, friction=friction)
    mpm.add_emitter(GridEmitter(lower=(ox - 0.5 * L, oy - 0.5 * W, 0.0), upper=(ox + 0.5 * L, oy + 0.5 * W, depth),
                                material=mat))
    mpm.set_colliders(geoms=["floor", "side_lo", "side_hi", "end_lo", "end_hi"])
    mpm.finalize()
    sp = voxel / 3.0
    col = 2.0 * sp
    x_lo, x_hi = ox - 0.5 * L + 3 * voxel, ox + 0.5 * L - 3 * voxel     # away from the end walls
    atan_mu = math.degrees(math.atan(friction))
    theta0 = max(atan_mu - 12.0, 0.0)
    theta_max = min(atan_mu + 12.0, 60.0)
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
    q_prev = v_prev = None
    meta = {"probe": "tilt_box", "friction": friction, "atan_mu_deg": atan_mu, "voxel": voxel, "depth": depth,
            "depth_cells": depth / voxel, "on_node": on_node, "offset": [ox, oy], "rate_deg_s": rate,
            "iterations": iterations, "tolerance": tolerance, "theta0": theta0, "theta_max": theta_max}
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
                bad = ~(np.isfinite(q).all(1) & np.isfinite(v).all(1))
                rec = dict(meta, error="nonfinite", step=s, t=t, theta=theta, particles=mpm.particles_per_world,
                           failed=fail_idx is not None,
                           theta_fail_deg=hist_theta[fail_idx] if fail_idx is not None else None,
                           where=nonfinite_report(q_prev, v_prev, bad, ox, oy, voxel),
                           history={"t": hist_t, "theta": hist_theta, "surface_speed_p90": hist_v})
                print(json.dumps({k: v for k, v in rec.items() if k != "history"}))
                if json_out:
                    json.dump(rec, open(json_out, "w"), indent=1)
                    if q_prev is not None:
                        np.savez_compressed(str(json_out).replace(".json", "_lastfinite.npz"), q=q_prev, v=v_prev,
                                            bad=bad)
                sys.exit(3)
            q_prev, v_prev = q, v
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
    return {**meta, "particles": mpm.particles_per_world,
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
    ap.add_argument("--on-node", action="store_true", help="wall faces on grid nodes (batch 01's first placement)")
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.friction, a.voxel, a.depth, a.rate, a.hold, iterations=a.iterations, tolerance=a.tolerance,
                on_node=a.on_node, json_out=a.json)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
