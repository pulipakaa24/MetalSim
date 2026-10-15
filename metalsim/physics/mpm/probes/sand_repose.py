"""Lifted-cylinder angle of repose with profile and stress export, and discretization variants (sand_fidelity batch 01,
arms 1 and 3; research_notes/sand_fidelity/PLAN_2026-10-14.md sections 2 and 6).

The scene, the lift protocol and the heap measurement are ``repose_lift.py``'s (imported, not copied): with the default
options this probe runs exactly ``repose_lift.run``'s configuration, so its R/dx 10 baseline must reproduce
runs/mpm_main/repose3/mu068_dx005_v025.json (fit 26.33 deg). Added:

* the final per-sector radial profiles h(r) and the flank angle fitted per band of height (``sand_metrics``): a convex
  heap (hypothesis R1) shows a steeper lower flank than the 25-75 % fit;
* at rest, ``mpm:particle_stress``: the yield ratio sqrt(J2) / (mu p) and the Lode parameter of the flank particles by
  depth below the surface (hypothesis R3: the flank at yield below atan(mu), off the pure-shear Lode angle);
  positions and stresses are also saved as .npz (``--npz``) for later analysis;
* ``--fill`` / ``--radius`` (a taller heap), ``--ppc-axis`` (particles per cell = ppc_axis^3; 3 = 27 as Isaac's
  cell_center lattice, 4 = 64), ``--velocity-basis``, ``--strain-basis``, ``--integration-scheme`` (hypothesis R2).
  ``velocity_basis="B2"`` makes the sparse grid non-rebuildable (the solver reallocates it every step), so the coupled
  step must not be captured: capture is switched off for it (a captured graph would replay the first step's grid).

Exit code 3 if any particle position or velocity is non-finite.

    python sand_repose.py [--friction 0.68] [--voxel 0.005] [--fill 0.05] [--radius 0.05] [--ppc-axis 3]
                          [--velocity-basis Q1] [--strain-basis P0] [--integration-scheme pic] [--json out] [--npz out]
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
from pile_repose import column  # noqa: E402
from repose_lift import N_STAVES, heap_metrics, tube_xml  # noqa: E402


def column_n(radius, height, voxel, n_axis):
    """``pile_repose.column`` with ``n_axis`` particles per voxel edge (n_axis = 3 is that function itself)."""
    if n_axis == 3:
        return column(radius, height, voxel)
    sp = voxel / n_axis
    n = int(math.ceil(radius / sp))
    g = (np.arange(-n, n) + 0.5) * sp
    x, y, zz = np.meshgrid(g, g, (np.arange(int(math.ceil(height / sp))) + 0.5) * sp, indexing="ij")
    p = np.stack([x, y, zz], -1).reshape(-1, 3)
    return p[np.hypot(p[:, 0], p[:, 1]) <= radius].astype(np.float32), sp


def run(friction=0.68, voxel=0.005, radius=0.05, fill=0.05, lift_speed=0.025, settle=0.3, rest=2.0, floor_mu=1.0,
        fps=120, substeps=2, iterations=250, sample_every=0.25, ppc_axis=3, velocity_basis="Q1", strain_basis="P0",
        integration_scheme="pic", npz=None, solver_variant="parity"):
    import mujoco
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import MPMConfig, MPMCoupling, MPMMaterial, PointsEmitter

    height = fill + 0.02
    model = mujoco.MjModel.from_xml_string(tube_xml(radius, height, floor_mu, 1.0 / fps / substeps))
    sim = BatchSim(model, 1, options=BatchSimOptions(substeps=substeps))
    p, sp = column_n(radius - 0.3 * voxel, fill, voxel, ppc_axis)
    mat = MPMMaterial(density=1500.0, friction=friction)
    extra = {}
    if velocity_basis != "Q1":
        extra.update(velocity_basis=velocity_basis, capture=False)
    if strain_basis != "P0":
        extra["strain_basis"] = strain_basis
    if integration_scheme != "pic":
        extra["integration_scheme"] = integration_scheme
    cfg = MPMConfig(voxel_size=voxel, max_iterations=iterations, proxy_mass="kinematic", solver_variant=solver_variant,
                    **extra)
    mpm = MPMCoupling(sim, cfg)
    mpm.add_emitter(PointsEmitter(positions=p, mass=sp ** 3 * mat.density, radius=0.5 * sp, material=mat))
    geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ["floor"] + [f"stave{k}" for k in range(N_STAVES)]]
    mpm.set_colliders(geoms=geoms)
    mpm.finalize()
    lift_time = (height + 0.03) / lift_speed
    n_lift_end = int(round((settle + lift_time) * fps))
    steps = n_lift_end + int(round(rest * fps))
    every = int(round(sample_every * fps))
    samples = []
    for s in range(steps):                                 # bounded
        t = s / fps
        zt = lift_speed * min(max(t - settle, 0.0), lift_time)
        sim.d.mocap_pos.assign(np.array([[[0.0, 0.0, zt]]], np.float32))
        mpm.step()
        if s >= n_lift_end and (s - n_lift_end) % every == every - 1:
            q = mpm.particle_q.numpy()
            v = mpm.particle_qd.numpy()
            if not (np.isfinite(q).all() and np.isfinite(v).all()):
                print(json.dumps({"error": "non-finite particles", "step": s}))
                sys.exit(3)
            m = heap_metrics(q, voxel, sp ** 3)
            m.update(t_rest=round((s - n_lift_end + 1) / fps, 3), max_speed=float(np.linalg.norm(v, axis=1).max()))
            samples.append(m)
    wp.synchronize()
    q = mpm.particle_q.numpy()
    stress = mpm.state.mpm.particle_stress.numpy()
    last = samples[-1]
    drift = samples[-1]["angle_fit_mean_deg"] - samples[-3]["angle_fit_mean_deg"] if len(samples) >= 3 else float("nan")
    target = math.degrees(math.atan(friction))
    est = [last["angle_fit_mean_deg"], last["angle_cone_volume_deg"], last["angle_height_flank_deg"]]
    prof = SM.heap_profiles(q, sp ** 3)
    bands = SM.flank_angles_by_height(prof, voxel)
    stress_ok = bool(np.isfinite(stress).all() and np.abs(stress).max() > 0)
    fs = SM.flank_stress(q, stress, friction, voxel, sp ** 3, last["r_apex"], last["r_toe"], last["h_max"]) \
        if stress_ok else {"error": "particle_stress is zero or non-finite (not written by this solver configuration)"}
    if npz:
        np.savez_compressed(npz, q=q.astype(np.float32), stress=stress.astype(np.float32), voxel=voxel, friction=friction)
    return {"probe": "sand_repose", "friction": friction, "atan_mu_deg": target, "voxel": voxel, "radius": radius,
            "fill": fill, "R_over_dx": radius / voxel, "ppc": ppc_axis ** 3, "velocity_basis": velocity_basis,
            "strain_basis": strain_basis, "integration_scheme": integration_scheme, "capture": mpm.capturable,
            "solver_variant": solver_variant,
            "particles": int(p.shape[0]), "lift_speed": lift_speed, "lift_froude": lift_speed / math.sqrt(9.81 * fill),
            "floor_mu": floor_mu, "iterations": iterations, "final": last, "angle_drift_last_0.5s_deg": drift,
            "at_rest": last["max_speed"] < 1e-3, "estimators_spread_deg": float(max(est) - min(est)),
            "deficit_deg": target - last["angle_fit_mean_deg"], "height_bands": bands, "flank_stress": fs,
            "profiles": prof, "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", type=float, default=0.68)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--fill", type=float, default=0.05)
    ap.add_argument("--lift-speed", type=float, default=0.025)
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--rest", type=float, default=2.0)
    ap.add_argument("--ppc-axis", type=int, default=3)
    ap.add_argument("--velocity-basis", default="Q1")
    ap.add_argument("--strain-basis", default="P0")
    ap.add_argument("--integration-scheme", default="pic")
    ap.add_argument("--solver-variant", default="parity")
    ap.add_argument("--json")
    ap.add_argument("--npz")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        r = run(a.friction, a.voxel, a.radius, a.fill, lift_speed=a.lift_speed, iterations=a.iterations, rest=a.rest,
                ppc_axis=a.ppc_axis, velocity_basis=a.velocity_basis, strain_basis=a.strain_basis,
                integration_scheme=a.integration_scheme, npz=a.npz, solver_variant=a.solver_variant)
    print(json.dumps({k: v for k, v in r.items() if k not in ("samples", "profiles", "height_bands", "flank_stress")}))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
