"""Compare MPM recordings (record_mpm.py output) between devices: the tiered suite of docs/research/mpm_2026-10-09.md.

    python -m metalsim.physics.mpm.compare RUN_A RUN_B [--json out.json]

T0  bitwise equality of positions (determinism / identical arithmetic)
T1  per-frame per-particle position RMS and max error, centre-of-mass error, kinetic-energy relative error
T2  column observables per run (runout, final height, deposit profile) for scaling-law fits

Recordings must share their initial state (``init_md5``); per-particle comparisons rely on the particle order.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "meta.json")) as fh:
        meta = json.load(fh)
    traj = dict(np.load(os.path.join(run_dir, "traj.npz")))
    init = dict(np.load(os.path.join(run_dir, "init.npz")))
    solves_path = os.path.join(run_dir, "solves.npz")
    solves = dict(np.load(solves_path)) if os.path.exists(solves_path) else {}
    return {"dir": run_dir, "meta": meta, "traj": traj, "init": init, "solves": solves}


def _common_frames(a: dict, b: dict):
    fa, fb = a["traj"]["frame"], b["traj"]["frame"]
    common = np.intersect1d(fa, fb)
    ia = np.searchsorted(fa, common)
    ib = np.searchsorted(fb, common)
    return common, ia, ib


def trajectory_metrics(a: dict, b: dict) -> dict:
    """T0/T1 between two recordings of the same scene and initial state."""
    if a["meta"]["init_md5"] != b["meta"]["init_md5"]:
        raise ValueError(f"initial states differ: {a['meta']['init_md5']} vs {b['meta']['init_md5']}")
    frames, ia, ib = _common_frames(a, b)
    pa, pb = a["traj"]["pos"][ia].astype(np.float64), b["traj"]["pos"][ib].astype(np.float64)
    va, vb = a["traj"]["vel"][ia].astype(np.float64), b["traj"]["vel"][ib].astype(np.float64)
    mass = a["init"]["mass"].astype(np.float64)
    if mass.size != pa.shape[1]:
        mass = np.ones(pa.shape[1])
    d = np.linalg.norm(pa - pb, axis=-1)  # [T, N]
    w = mass / mass.sum()
    com_a = np.einsum("tnk,n->tk", pa, w)
    com_b = np.einsum("tnk,n->tk", pb, w)
    ke_a = 0.5 * np.einsum("tn,n->t", (va * va).sum(-1), mass)
    ke_b = 0.5 * np.einsum("tn,n->t", (vb * vb).sum(-1), mass)
    ke_scale = max(ke_a.max(), ke_b.max(), 1e-30)
    return {
        "frames": frames.tolist(),
        "bitwise_equal": bool(np.array_equal(a["traj"]["pos"][ia], b["traj"]["pos"][ib])),
        "first_diff_frame": int(frames[np.argmax(d.max(1) > 0)]) if (d.max(1) > 0).any() else None,
        "pos_rms": np.sqrt((d * d).mean(1)).tolist(),
        "pos_max": d.max(1).tolist(),
        "com_err": np.linalg.norm(com_a - com_b, axis=-1).tolist(),
        "ke_a": ke_a.tolist(),
        "ke_b": ke_b.tolist(),
        "ke_rel_err_of_peak": (np.abs(ke_a - ke_b) / ke_scale).tolist(),
    }


def column_observables(run: dict, frame_index: int = -1, bin_width: float | None = None) -> dict:
    """Runout and deposit of a column-collapse recording (scene ``column``).

    Runout ``L_f`` is the 99.5th percentile of particle x (the deposit tip, robust to a few saltating particles);
    the final height ``H_f`` is the maximum height of the deposit profile; the profile ``h(x)`` is the maximum particle
    top (z + radius) per x bin of width ``bin_width`` (default: the particle spacing).
    """
    info = run["meta"]["info"]
    spec = info["spec"]
    L_i, H_i = spec["length"], info["initial_height"]
    pos = run["traj"]["pos"][frame_index].astype(np.float64)
    r = run["init"]["radius"].astype(np.float64)
    if r.size != pos.shape[0]:
        # recordings without radii (Isaac Lab's MPMObject): Isaac Lab's cell_center radius, 0.5 * cbrt(cell volume)
        extent = np.array([L_i, spec["width"], H_i], dtype=np.float32)
        cell = extent / np.ceil(spec["particles_per_cell"] * extent / spec["voxel_size"])
        r = np.full(pos.shape[0], 0.5 * float(np.cbrt(np.prod(cell))))
    x, z = pos[:, 0], pos[:, 2] + r
    spacing = spec["voxel_size"] / spec["particles_per_cell"]
    bw = bin_width or spacing
    edges = np.arange(0.0, max(x.max(), L_i) + bw, bw)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    h = np.zeros(len(edges) - 1)
    np.maximum.at(h, idx, np.maximum(z, 0.0))
    L_f = float(np.percentile(x, 99.5))
    H_f = float(h.max())
    return {
        "aspect": spec["aspect"], "L_i": L_i, "H_i": H_i, "L_f": L_f, "L_max": float(x.max()), "H_f": H_f,
        "runout_norm": (L_f - L_i) / L_i, "height_norm": H_f / L_i,
        "profile_x": (0.5 * (edges[:-1] + edges[1:])).tolist(), "profile_h": h.tolist(),
        "frame": int(run["traj"]["frame"][frame_index]),
    }


def profile_rms(obs_a: dict, obs_b: dict) -> float:
    """RMS difference of two deposit profiles on their common bins."""
    n = min(len(obs_a["profile_h"]), len(obs_b["profile_h"]))
    ha = np.zeros(max(len(obs_a["profile_h"]), len(obs_b["profile_h"])))
    hb = ha.copy()
    ha[: len(obs_a["profile_h"])] = obs_a["profile_h"]
    hb[: len(obs_b["profile_h"])] = obs_b["profile_h"]
    del n
    return float(np.sqrt(np.mean((ha - hb) ** 2)))


def solver_stats(run: dict) -> dict:
    s = run["solves"]
    if not s or s["iterations"].size == 0:
        return {}
    it = s["iterations"]
    return {"solves": int(it.size), "iter_mean": float(it.mean()), "iter_max": int(it.max()),
            "at_cap_fraction": float((it >= it.max()).mean()), "res_l2_max": float(s["res_l2"].max()),
            "res_l2_median": float(np.median(s["res_l2"])), "res_linf_max": float(s["res_linf"].max())}


def throughput(run: dict, skip: int = 2) -> dict:
    wall = np.asarray(run["meta"]["wall_s"][skip:])
    n = run["meta"]["particle_count"]
    if wall.size == 0:
        return {}
    ms = float(np.median(wall) * 1e3)
    return {"frame_ms_median": ms, "particles": n, "particle_steps_per_s": n / (ms * 1e-3),
            "ms_per_Mparticle": ms / (n / 1e6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    out = {"a": args.a, "b": args.b, "solver_a": solver_stats(a), "solver_b": solver_stats(b),
           "throughput_a": throughput(a), "throughput_b": throughput(b)}
    tm = trajectory_metrics(a, b)
    out["trajectory"] = tm
    if a["meta"]["info"].get("scene") == "column":
        oa, ob = column_observables(a), column_observables(b)
        out["column"] = {"a": {k: v for k, v in oa.items() if not k.startswith("profile")},
                         "b": {k: v for k, v in ob.items() if not k.startswith("profile")},
                         "profile_rms": profile_rms(oa, ob)}
    fr = tm["frames"]
    print(f"bitwise equal: {tm['bitwise_equal']}  first differing frame: {tm['first_diff_frame']}")
    for k in sorted({0, min(9, len(fr) - 1), min(59, len(fr) - 1), len(fr) - 1}):
        print(f"frame {fr[k]:4d}: pos rms {tm['pos_rms'][k]:.3e} max {tm['pos_max'][k]:.3e}  com {tm['com_err'][k]:.3e}"
              f"  KE rel {tm['ke_rel_err_of_peak'][k]:.3e}")
    print("solver a:", out["solver_a"])
    print("solver b:", out["solver_b"])
    print("throughput a:", out["throughput_a"])
    print("throughput b:", out["throughput_b"])
    if "column" in out:
        print("column a:", out["column"]["a"])
        print("column b:", out["column"]["b"])
        print("profile rms:", out["column"]["profile_rms"])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)


if __name__ == "__main__":
    main()
