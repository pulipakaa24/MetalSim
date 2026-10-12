"""Metal vs Isaac (L4) parity verdicts for the MPM suite (stage_mpm.sh on the L4, record_metal_suite.sh on Metal).

    python -m metalsim.physics.mpm.parity_report L4_DIR METAL_DIR [--json out.json] [--md out.md]

Criteria (MPM.md section 4; the floors are the L4's own float32-vs-float64 runs, R0_*_f32):
  T1  short horizon (frame 10): per-particle position RMS within 3x the scene's float32 floor, where a floor exists
  T2  column collapse, per aspect ratio: seed-mean runout and final height within 10 % and within the combined 2-sigma
      seed spread (2 * sqrt(sd_l4^2 + sd_metal^2)); fitted runout exponents (a <= 2, a >= 3) within 0.05
  T3  examples: centre-of-mass path error within 2 % of the L4's centre-of-mass travel; robot / rigid bodies: body
      position path error within 10 % of their travel; plate: force-vs-travel RMS within 10 % of the L4 peak force
Every row also reports solver iterations; wall time is reported but is not a timing-class measurement.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from metalsim.physics.mpm import compare

ASPECTS = ["0.5", "1", "2", "3", "5", "8"]
SEEDS = range(10)
EXAMPLES = ["R3_granular", "R3_multi_material", "R3_snow_ball", "R3_viscous", "R3_beam_twist", "R4_twoway",
            "R6_anymal"]
# T1: (Metal run, L4 run with the same float32 accumulation, L4 pairs whose difference is the L4's own noise floor:
# float64 vs float32 accumulation, fast_math vs exact / and sqrt)
T1_CASES = [
    ("R0_a2_host_1", "R0_a2_f32", [("R0_a2_host_1", "R0_a2_f32"), ("R0_a2_f32", "R0_a2_f32_precise")]),
    ("R0_granular_host_1", "R0_granular_f32",
     [("R0_granular_host_1", "R0_granular_f32"), ("R0_granular_f32", "R0_granular_f32_precise")]),
]


def _have(d, run):
    return os.path.exists(os.path.join(d, run, "meta.json"))


def _frame_value(tm, frame, key):
    fr = np.asarray(tm["frames"])
    i = int(np.argmin(np.abs(fr - frame)))
    return float(tm[key][i]), int(fr[i])


def _wall(run):
    t = compare.throughput(run)
    return t.get("frame_ms_median")


def t1_rows(l4, mt):
    rows = []
    for run, ref, floors in T1_CASES:
        if not (_have(l4, ref) and _have(mt, run)):
            continue
        tm = compare.trajectory_metrics(compare.load(os.path.join(l4, ref)), compare.load(os.path.join(mt, run)))
        fms = [compare.trajectory_metrics(compare.load(os.path.join(l4, x)), compare.load(os.path.join(l4, y)))
               for x, y in floors if _have(l4, x) and _have(l4, y)]
        row = {"run": run, "l4_ref": ref, "pass": True}
        for label, frame in (("early", 10 if "a2" in run else 6), ("end", tm["frames"][-1])):
            d, fr = _frame_value(tm, frame, "pos_rms")
            fl = max(_frame_value(fm, frame, "pos_rms")[0] for fm in fms)
            row.update({f"{label}_frame": fr, f"{label}_rms": d, f"{label}_floor": fl, f"{label}_ratio": d / fl})
            row["pass"] &= bool(d <= 3 * fl)
        rows.append(row)
    return rows


def _seed_obs(d, a):
    obs = [compare.column_observables(compare.load(os.path.join(d, f"R1_a{a}_j{s}"))) for s in SEEDS
           if _have(d, f"R1_a{a}_j{s}")]
    return np.array([o["runout_norm"] for o in obs]), np.array([o["height_norm"] for o in obs])


def t2_rows(l4, mt):
    rows, means = [], {"l4": [], "metal": []}
    for a in ASPECTS:
        if not (_have(l4, f"R1_a{a}") and _have(mt, f"R1_a{a}")):
            continue
        la, ma = compare.load(os.path.join(l4, f"R1_a{a}")), compare.load(os.path.join(mt, f"R1_a{a}"))
        lo, mo = compare.column_observables(la), compare.column_observables(ma)
        row = {"a": float(a), "lattice_runout_l4": lo["runout_norm"], "lattice_runout_metal": mo["runout_norm"],
               "lattice_height_l4": lo["height_norm"], "lattice_height_metal": mo["height_norm"],
               "lattice_profile_rms_m": compare.profile_rms(lo, mo),
               "iters_l4": compare.solver_stats(la).get("iter_mean"),
               "iters_metal": compare.solver_stats(ma).get("iter_mean")}
        lr, lh = _seed_obs(l4, a)
        mr, mh = _seed_obs(mt, a)
        if lr.size >= 2 and mr.size >= 2:
            ok = True
            for name, x, y in (("runout", lr, mr), ("height", lh, mh)):
                diff = y.mean() - x.mean()
                sig2 = 2 * np.sqrt(x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2)
                rel = diff / x.mean()
                row.update({f"{name}_l4": x.mean(), f"{name}_l4_sd": x.std(ddof=1), f"{name}_metal": y.mean(),
                            f"{name}_metal_sd": y.std(ddof=1), f"{name}_rel": rel, f"{name}_2sigma": sig2,
                            "seeds": int(min(x.size, y.size))})
                ok &= abs(rel) <= 0.10 and abs(diff) <= sig2
            row["pass"] = bool(ok)
            means["l4"].append((float(a), lr.mean()))
            means["metal"].append((float(a), mr.mean()))
        rows.append(row)
    fits = {}
    for side, pts in means.items():
        p = np.array(pts)
        if len(p) < 4:
            continue
        lo_m, hi_m = p[:, 0] <= 2, p[:, 0] >= 3
        fits[side] = {"exp_low": float(np.polyfit(np.log(p[lo_m, 0]), np.log(p[lo_m, 1]), 1)[0]),
                      "exp_high": float(np.polyfit(np.log(p[hi_m, 0]), np.log(p[hi_m, 1]), 1)[0])}
    if len(fits) == 2:
        fits["pass"] = bool(all(abs(fits["l4"][k] - fits["metal"][k]) <= 0.05 for k in ("exp_low", "exp_high")))
    return rows, fits


def _com(run):
    pos = run["traj"]["pos"].astype(np.float64)
    m = run["init"]["mass"].astype(np.float64)
    m = m if m.size == pos.shape[1] else np.ones(pos.shape[1])
    return np.einsum("tnk,n->tk", pos, m / m.sum())


def t3_rows(l4, mt):
    rows = []
    for run in EXAMPLES:
        if not (_have(l4, run) and _have(mt, run)):
            continue
        a, b = compare.load(os.path.join(l4, run)), compare.load(os.path.join(mt, run))
        if a["meta"]["init_md5"] != b["meta"]["init_md5"]:
            rows.append({"run": run, "error": "initial states differ"})
            continue
        frames, ia, ib = compare._common_frames(a, b)
        ca, cb = _com(a)[ia], _com(b)[ib]
        travel = float(np.linalg.norm(ca - ca[0], axis=-1).max())
        com_err = float(np.linalg.norm(ca - cb, axis=-1).max())
        tm = compare.trajectory_metrics(a, b)
        ka, kb = np.asarray(tm["ke_a"]), np.asarray(tm["ke_b"])
        row = {"run": run, "particles": a["meta"]["particle_count"], "frames": int(frames[-1]),
               "com_travel_m": travel, "com_err_max_m": com_err, "com_err_rel": com_err / max(travel, 1e-12),
               "ke_curve_rel_rms": float(np.sqrt(np.mean((ka - kb) ** 2)) / max(ka.max(), 1e-30)),
               "pos_rms_end_m": tm["pos_rms"][-1],
               "iters_l4": compare.solver_stats(a).get("iter_mean"), "iters_metal": compare.solver_stats(b).get("iter_mean"),
               "ms_l4": _wall(a), "ms_metal": _wall(b)}
        ok = row["com_err_rel"] <= 0.02
        if "body_q" in a["traj"] and a["traj"]["body_q"].size:
            qa, qb = a["traj"]["body_q"][ia][..., :3].astype(np.float64), b["traj"]["body_q"][ib][..., :3].astype(np.float64)
            btravel = float(np.linalg.norm(qa - qa[:1], axis=-1).max())
            berr = float(np.linalg.norm(qa - qb, axis=-1).max())
            row.update({"body_travel_m": btravel, "body_err_max_m": berr, "body_err_rel": berr / max(btravel, 1e-12)})
            ok &= row["body_err_rel"] <= 0.10
        row["pass"] = bool(ok)
        rows.append(row)
    for run in ("R7_plate_intrude", "R7_plate_drag"):
        if not (_have(l4, run) and _have(mt, run)):
            continue
        a, b = compare.load(os.path.join(l4, run)), compare.load(os.path.join(mt, run))
        axis = 2 if "intrude" in run else 0
        fa, fb = np.abs(a["traj"]["x_plate_force"][:, axis]), np.abs(b["traj"]["x_plate_force"][:, axis])
        n = min(fa.size, fb.size)
        # compare on a smoothed curve: the raw force steps by half a voxel as the plate crosses grid planes
        k = np.ones(5) / 5
        sa, sb = np.convolve(fa[:n], k, "same"), np.convolve(fb[:n], k, "same")
        rel = float(np.sqrt(np.mean((sa - sb) ** 2)) / max(sa.max(), 1e-30))
        rows.append({"run": run, "particles": a["meta"]["particle_count"], "force_peak_l4_N": float(fa.max()),
                     "force_peak_metal_N": float(fb.max()), "force_curve_rel_rms": rel,
                     "force_final_l4_N": float(sa[n - 3]), "force_final_metal_N": float(sb[n - 3]),
                     "iters_l4": compare.solver_stats(a).get("iter_mean"),
                     "iters_metal": compare.solver_stats(b).get("iter_mean"),
                     "ms_l4": _wall(a), "ms_metal": _wall(b), "pass": bool(rel <= 0.10)})
    return rows


def _fmt(v):
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _table(rows, keys):
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("l4")
    ap.add_argument("metal")
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args()
    t1 = t1_rows(args.l4, args.metal)
    t2, fits = t2_rows(args.l4, args.metal)
    t3 = t3_rows(args.l4, args.metal)
    md = ["## T1 short horizon vs the L4 float32 floor",
          _table(t1, ["run", "l4_ref", "early_frame", "early_rms", "early_floor", "early_ratio", "end_frame",
                      "end_rms", "end_floor", "end_ratio", "pass"]),
          "", "## T2 column collapse (normalized runout / height; seed means)",
          _table(t2, ["a", "lattice_runout_l4", "lattice_runout_metal", "runout_l4", "runout_l4_sd", "runout_metal",
                      "runout_metal_sd", "runout_rel", "height_l4", "height_metal", "height_rel", "iters_l4",
                      "iters_metal", "pass"]),
          "", f"fitted exponents: {json.dumps(fits)}",
          "", "## T3 examples, coupling, robot, plate",
          _table([r for r in t3 if "com_err_rel" in r or "error" in r],
                 ["run", "particles", "frames", "com_travel_m", "com_err_rel", "ke_curve_rel_rms", "body_travel_m",
                  "body_err_rel", "iters_l4", "iters_metal", "ms_l4", "ms_metal", "pass", "error"]),
          "", _table([r for r in t3 if "force_curve_rel_rms" in r],
                     ["run", "particles", "force_peak_l4_N", "force_peak_metal_N", "force_final_l4_N",
                      "force_final_metal_N", "force_curve_rel_rms", "iters_l4", "iters_metal", "ms_l4", "ms_metal",
                      "pass"])]
    text = "\n".join(md)
    print(text)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(text + "\n")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"t1": t1, "t2": t2, "fits": fits, "t3": t3}, fh, indent=1)


if __name__ == "__main__":
    main()
