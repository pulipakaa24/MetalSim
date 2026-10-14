"""Angle of repose by the lifted-cylinder test, in MetalSim's main simulator (BatchSim + MPMCoupling).

Protocol (research_notes/MPM in main simulator/angle_of_repose.md): sand fills a tube (24 staves on one mocap body,
frictionless towards the sand, a kinematic MPM collider moved by its mocap position each step) standing on a rough
floor (friction 1.0, above every tested material friction); the tube rises slowly (V / sqrt(g H) <= 0.05) and the
sand flows out into a heap. The heap is measured at intervals during a rest period: the 95th-percentile surface per
radial bin in 8 azimuthal sectors, a linear fit between the apex and the toe (window 25-75 % of the height, shrunk by
1.5 voxels at both ends), plus two geometric estimates (cone volume, height over flank length). Expected for the
solver's yield surface sqrt(J2) <= mu p: atan(mu) (21.8 / 34.2 / 42.0 deg for mu = 0.4 / 0.68 / 0.9), approached from
below as the resolution increases.

batch_02's run of an earlier version is void: the tube was driven by a velocity actuator whose gain made the explicit
integration unstable (tube position 1e20 m), so the sand collapsed instead of being lifted.

    python repose_lift.py [--friction 0.68] [--voxel 0.005] [--lift-speed 0.025] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pile_repose import column  # noqa: E402

import numpy as np  # noqa: E402

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
<body name="tube" mocap="true" pos="0 0 0">{''.join(staves)}</body></worldbody></mujoco>"""


def heap_metrics(q, voxel, particle_volume, sectors=8, bins=30):
    """Flank angle per sector (fit), cone-volume angle and height-over-flank angle of a heap centred at the origin."""
    # free surface: the highest particle in each xy column (columns of about two particle spacings), then the column
    # surfaces averaged per radial bin (a percentile over a whole annulus underestimates a sloped surface)
    col = 2.0 * particle_volume ** (1.0 / 3.0)
    ij = np.floor(q[:, :2] / col).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    zmax = np.maximum.reduceat(q[order, 2], starts)
    cxy = (ij[order][starts] + 0.5) * col
    r = np.hypot(cxy[:, 0], cxy[:, 1])
    th = np.arctan2(cxy[:, 1], cxy[:, 0])
    z = zmax
    rmax = np.quantile(r, 0.995)
    edges = np.linspace(0.0, rmax, bins + 1)

    def profile(sel):
        rc, h = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = sel & (r >= a) & (r < b)
            if m.sum() >= 2:
                rc.append(float(np.mean(r[m])))
                h.append(float(np.mean(z[m])))
        return np.asarray(rc), np.asarray(h)

    rc, h = profile(np.ones_like(r, bool))
    hmax = float(h.max())
    r_apex = float(rc[h >= 0.95 * hmax].max())
    r_toe = float(rc[h >= 0.02 * hmax].max())
    angles = []
    for s in range(sectors):
        lo = -math.pi + 2 * math.pi * s / sectors
        sel = (th >= lo) & (th < lo + 2 * math.pi / sectors)
        rs, hs = profile(sel)
        if len(rs) < 4:
            continue
        win = (hs >= 0.25 * hmax) & (hs <= 0.75 * hmax) & (rs >= r_apex + 1.5 * voxel) & (rs <= r_toe - 1.5 * voxel)
        if win.sum() < 3:
            win = (hs >= 0.25 * hmax) & (hs <= 0.75 * hmax)
        if win.sum() >= 3:
            slope, _ = np.polyfit(rs[win], hs[win], 1)
            angles.append(math.degrees(math.atan(-slope)))
    volume = particle_volume * q.shape[0]
    return {"angle_fit_mean_deg": float(np.mean(angles)) if angles else float("nan"),
            "angle_fit_sd_deg": float(np.std(angles)) if angles else float("nan"), "sectors_fitted": len(angles),
            "angle_cone_volume_deg": math.degrees(math.atan(3 * volume / (math.pi * r_toe ** 3))),
            "angle_height_flank_deg": math.degrees(math.atan(hmax / max(r_toe - r_apex, 1e-9))),
            "h_max": hmax, "r_apex": r_apex, "r_toe": r_toe}


def run(friction=0.68, voxel=0.005, radius=0.05, fill=0.05, lift_speed=0.025, settle=0.3, rest=2.0, floor_mu=1.0,
        fps=120, substeps=2, iterations=250, sample_every=0.25):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import MPMConfig, MPMCoupling, MPMMaterial, PointsEmitter

    height = fill + 0.02
    model = mujoco.MjModel.from_xml_string(tube_xml(radius, height, floor_mu, 1.0 / fps / substeps))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    p, sp = column(radius - 0.3 * voxel, fill, voxel)
    mat = MPMMaterial(density=1500.0, friction=friction)
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=voxel, max_iterations=iterations, proxy_mass="kinematic"))
    mpm.add_emitter(PointsEmitter(positions=p, mass=sp ** 3 * mat.density, radius=0.5 * sp, material=mat))
    geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ["floor"] + [f"stave{k}" for k in range(N_STAVES)]]
    mpm.set_colliders(geoms=geoms)
    mpm.finalize()
    lift_time = (height + 0.03) / lift_speed
    n_lift_end = int(round((settle + lift_time) * fps))
    steps = n_lift_end + int(round(rest * fps))
    every = int(round(sample_every * fps))
    samples = []
    for s in range(steps):
        t = s / fps
        zt = lift_speed * min(max(t - settle, 0.0), lift_time)
        sim.d.mocap_pos.assign(np.array([[[0.0, 0.0, zt]]], np.float32))
        mpm.step()
        if s >= n_lift_end and (s - n_lift_end) % every == every - 1:
            q = mpm.particle_q.numpy()
            v = mpm.particle_qd.numpy()
            m = heap_metrics(q, voxel, sp ** 3)
            m.update(t_rest=round((s - n_lift_end + 1) / fps, 3), max_speed=float(np.linalg.norm(v, axis=1).max()))
            samples.append(m)
    wp.synchronize()
    last = samples[-1]
    drift = samples[-1]["angle_fit_mean_deg"] - samples[-3]["angle_fit_mean_deg"] if len(samples) >= 3 else float("nan")
    target = math.degrees(math.atan(friction))
    est = [last["angle_fit_mean_deg"], last["angle_cone_volume_deg"], last["angle_height_flank_deg"]]
    return {"friction": friction, "atan_mu_deg": target, "voxel": voxel, "R_over_dx": radius / voxel,
            "particles": int(p.shape[0]), "lift_speed": lift_speed, "lift_froude": lift_speed / math.sqrt(9.81 * fill),
            "floor_mu": floor_mu, "iterations": iterations, "final": last, "angle_drift_last_0.5s_deg": drift,
            "at_rest": last["max_speed"] < 1e-3, "estimators_spread_deg": float(max(est) - min(est)),
            "deficit_deg": target - last["angle_fit_mean_deg"], "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", type=float, default=0.68)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--lift-speed", type=float, default=0.025)
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--rest", type=float, default=2.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.friction, a.voxel, lift_speed=a.lift_speed, iterations=a.iterations, rest=a.rest)
    print(json.dumps({k: v for k, v in r.items() if k != "samples"}))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
