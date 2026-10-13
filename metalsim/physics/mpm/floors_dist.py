"""Metal vs the L4's floor *distribution* (MPM.md section 5 follow-up; reports/CUDA and Metal float parity.md).

    python -m metalsim.physics.mpm.floors_dist L4_DIR [L4_DIR ...] --metal METAL_DIR [METAL_DIR ...] [--json o] [--md o]

Every L4_DIR holds stage_mpm_floors.sh / stage_mpm_floors2.sh output (<scene>_<variant>/). The first L4_DIR's
<scene>_base is the reference. All other runs of the scene in any L4_DIR are legitimate CUDA perturbations (compiler
variants, ulp-level initial jitter, combinations); ``<scene>_base2`` (an unperturbed rerun) is only checked for
bitwise equality with base (CUDA run-to-run determinism) and not counted as a perturbation. Each METAL_DIR holds
record_variant_suite.sh "floors" runs (F_<scene>).

Per scene, at the first, middle and last common recorded frame and as the maximum over all recorded frames:
  * floor sample  D_k  = particle-position RMS of perturbed L4 run k against base (k = 1..K)
  * pairwise      P_ij = RMS between any two L4 runs (base included), K+1 choose 2 samples
  * Metal         M    = RMS of Metal against base; also Metal's median RMS against every L4 run
  * M's percentile rank within D and within P, and M / p90, M / p95, M / max of each.
Plate drag: the same with the smoothed force-curve RMS relative to the base peak (floors_report.py's metric).
Verdict: "inside" when M <= max(D) at every picked frame; "inside p95" when M <= p95(P) at every picked frame.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np

from metalsim.physics.mpm import compare

SCENES = ["column_a2", "granular", "multi_material", "twoway", "anymal", "plate_drag"]


def _have(path):
    return os.path.exists(os.path.join(path, "meta.json"))


def _runs(l4_dirs, scene):
    """{variant: path} for every L4 run of ``scene`` (first directory wins on a name clash)."""
    out = {}
    for d in l4_dirs:
        for p in sorted(glob.glob(os.path.join(d, f"{scene}_*"))):
            v = os.path.basename(p)[len(scene) + 1:]
            if os.path.isdir(p) and _have(p) and v not in out:
                # scene names that prefix another (none today) would leak in; guard on the recorded scene
                out[v] = p
    return out


def _load_pos(path):
    meta = json.load(open(os.path.join(path, "meta.json")))
    t = np.load(os.path.join(path, "traj.npz"))
    pos = t["pos"]
    frames = t["frame"]
    force = t["x_plate_force"] if "x_plate_force" in t.files else None
    return {"meta": meta, "pos": pos, "frame": frames, "force": force}


def _rms_frames(a, b, frames):
    ia = np.searchsorted(a["frame"], frames)
    ib = np.searchsorted(b["frame"], frames)
    out = np.empty(len(frames))
    for n, (i, j) in enumerate(zip(ia, ib)):
        d = a["pos"][i].astype(np.float64) - b["pos"][j].astype(np.float64)
        out[n] = np.sqrt((d * d).sum(-1).mean())
    return out


def _force_curve(r):
    f = np.abs(r["force"][:, 0])
    return np.convolve(f, np.ones(5) / 5, "same")


def _force_rel(a, b, peak):
    sa, sb = _force_curve(a), _force_curve(b)
    n = min(sa.size, sb.size)
    return float(np.sqrt(np.mean((sa[:n] - sb[:n]) ** 2)) / peak)


def _pct_rank(x, sample):
    s = np.asarray(sample)
    return float(100.0 * (s < x).mean() + 50.0 * (s == x).mean())


def _stats(sample):
    s = np.asarray(sample)
    return {"n": int(s.size), "min": float(s.min()), "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)), "p95": float(np.percentile(s, 95)), "max": float(s.max())}


def scene_report(l4_dirs, metal_dirs, scene):
    runs = _runs(l4_dirs, scene)
    if "base" not in runs:
        return None
    base = _load_pos(runs["base"])
    det = {}
    for v in [v for v in runs if v.startswith("base") and v != "base"]:
        o = _load_pos(runs[v])
        det[v] = bool(o["pos"].shape == base["pos"].shape and np.array_equal(o["pos"], base["pos"]))
    pert = {v: p for v, p in runs.items() if not v.startswith("base")}
    L = {"base": base}
    for v, p in pert.items():
        r = _load_pos(p)
        if r["meta"]["init_md5"] != base["meta"]["init_md5"]:
            raise ValueError(f"{p}: init differs from base")
        L[v] = r
    metal = {}
    for md in metal_dirs:
        p = os.path.join(md, f"F_{scene}")
        if _have(p):
            metal[md] = _load_pos(p)
    if not metal:
        return None
    frames = base["frame"]
    for r in list(L.values()) + list(metal.values()):
        frames = np.intersect1d(frames, r["frame"])
    pick = np.array([frames[0], frames[len(frames) // 2], frames[-1]])
    row = {"scene": scene, "frames": pick.tolist(), "n_perturbed": len(pert), "variants": sorted(pert),
           "base_reruns_bitwise_equal": det}

    if scene == "plate_drag":
        peak = float(_force_curve(base).max())
        D = {v: _force_rel(base, r, peak) for v, r in L.items() if v != "base"}
        names = list(L)
        P = [_force_rel(L[a], L[b], peak) for a, b in itertools.combinations(names, 2)]
        row["metric"] = "force-curve RMS / base peak"
        row["floor"] = _stats(list(D.values()))
        row["floor_samples"] = D
        row["pairwise"] = _stats(P)
        row["final_force_N"] = {v: float(_force_curve(r)[-3]) for v, r in L.items()}
        row["metal"] = {}
        for md, m in metal.items():
            M = _force_rel(base, m, peak)
            Mall = [_force_rel(r, m, peak) for r in L.values()]
            row["metal"][md] = {"M": M, "M_median_vs_all_l4": float(np.median(Mall)),
                                "rank_in_floor": _pct_rank(M, list(D.values())), "rank_in_pairwise": _pct_rank(M, P),
                                "ratio_max_floor": M / row["floor"]["max"], "ratio_p95_pairwise": M / row["pairwise"]["p95"],
                                "ratio_p90_pairwise": M / row["pairwise"]["p90"],
                                "final_force_N": float(_force_curve(m)[-3]),
                                "inside_max": bool(M <= row["floor"]["max"]),
                                "inside_p95": bool(M <= row["pairwise"]["p95"])}
        return row

    allf = frames
    D = {v: _rms_frames(base, r, allf) for v, r in L.items() if v != "base"}
    names = list(L)
    P = np.array([_rms_frames(L[a], L[b], allf) for a, b in itertools.combinations(names, 2)])  # [pairs, F]
    Dm = np.array(list(D.values()))  # [K, F]
    ip = np.searchsorted(allf, pick)
    row["metric"] = "particle-position RMS [m]"
    row["floor"] = [_stats(Dm[:, i]) for i in ip]
    row["pairwise"] = [_stats(P[:, i]) for i in ip]
    row["floor_samples"] = {v: d[ip].tolist() for v, d in D.items()}
    row["metal"] = {}
    for md, m in metal.items():
        M = _rms_frames(base, m, allf)
        Mall = np.array([_rms_frames(r, m, allf) for r in L.values()])
        mx, p95 = Dm.max(0), np.percentile(P, 95, axis=0)
        ok = mx > 0
        row["metal"][md] = {
            "M": M[ip].tolist(), "M_median_vs_all_l4": np.median(Mall, 0)[ip].tolist(),
            "rank_in_floor": [_pct_rank(M[i], Dm[:, i]) for i in ip],
            "rank_in_pairwise": [_pct_rank(M[i], P[:, i]) for i in ip],
            "ratio_max_floor": (M[ip] / Dm.max(0)[ip]).tolist(),
            "ratio_p95_pairwise": (M[ip] / np.percentile(P[:, ip], 95, axis=0)).tolist(),
            "ratio_p90_pairwise": (M[ip] / np.percentile(P[:, ip], 90, axis=0)).tolist(),
            "worst_frame_ratio_max_floor": float((M[ok] / mx[ok]).max()) if ok.any() else None,
            "worst_frame_ratio_p95_pairwise": float((M[ok] / p95[ok]).max()) if ok.any() else None,
            "inside_max": bool((M[ip] <= Dm.max(0)[ip]).all()),
            "inside_p95": bool((M[ip] <= np.percentile(P[:, ip], 95, axis=0)).all()),
        }
    return row


def _fm(x):
    return f"{x * 1e6:.3g} um" if x < 1e-3 else f"{x * 1e3:.3g} mm"


def to_md(rows):
    md = []
    for r in rows:
        md += [f"#### {r['scene']} ({r['n_perturbed']} perturbed L4 runs; base reruns bitwise: {r['base_reruns_bitwise_equal']})", ""]
        if r["scene"] == "plate_drag":
            f, p = r["floor"], r["pairwise"]
            md += ["| Metal run | M (% of peak) | floor p50 / p90 / max | pairwise p90 / p95 / max | rank in floor / pairwise | M / max floor | M / p95 pairwise | final force (L4 range) | verdict |",
                   "|---|---|---|---|---|---|---|---|---|"]
            ff = list(r["final_force_N"].values())
            for md_, m in r["metal"].items():
                md.append(f"| {os.path.basename(md_.rstrip('/'))} | {m['M'] * 100:.2f} | {f['p50'] * 100:.2f} / {f['p90'] * 100:.2f} / {f['max'] * 100:.2f} | "
                          f"{p['p90'] * 100:.2f} / {p['p95'] * 100:.2f} / {p['max'] * 100:.2f} | {m['rank_in_floor']:.0f} / {m['rank_in_pairwise']:.0f} | "
                          f"{m['ratio_max_floor']:.2f} | {m['ratio_p95_pairwise']:.2f} | {m['final_force_N']:.0f} N ({min(ff):.0f}-{max(ff):.0f}) | "
                          f"{'inside' if m['inside_max'] else 'OUTSIDE'} max, {'inside' if m['inside_p95'] else 'OUTSIDE'} p95 |")
        else:
            md += ["| Metal run | frame | M | floor p50 / p90 / max | pairwise p90 / p95 / max | rank in floor / pairwise (%) | M / max floor | M / p95 pairwise |",
                   "|---|---|---|---|---|---|---|---|"]
            for md_, m in r["metal"].items():
                for i, fr in enumerate(r["frames"]):
                    f, p = r["floor"][i], r["pairwise"][i]
                    md.append(f"| {os.path.basename(md_.rstrip('/'))} | {fr} | {_fm(m['M'][i])} | {_fm(f['p50'])} / {_fm(f['p90'])} / {_fm(f['max'])} | "
                              f"{_fm(p['p90'])} / {_fm(p['p95'])} / {_fm(p['max'])} | {m['rank_in_floor'][i]:.0f} / {m['rank_in_pairwise'][i]:.0f} | "
                              f"{m['ratio_max_floor'][i]:.2f} | {m['ratio_p95_pairwise'][i]:.2f} |")
                md.append(f"| {os.path.basename(md_.rstrip('/'))} | worst frame | | | | | {m['worst_frame_ratio_max_floor']:.2f} | "
                          f"{m['worst_frame_ratio_p95_pairwise']:.2f} |")
            md.append("")
            md.append("Verdicts: " + "; ".join(f"{os.path.basename(k.rstrip('/'))}: {'inside' if m['inside_max'] else 'OUTSIDE'} max floor, "
                                              f"{'inside' if m['inside_p95'] else 'OUTSIDE'} p95 pairwise" for k, m in r["metal"].items()))
        md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("l4", nargs="+")
    ap.add_argument("--metal", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args()
    rows = []
    for s in a.scenes:
        r = scene_report(a.l4, a.metal, s)
        if r:
            rows.append(r)
            print(to_md([r]), flush=True)
    if a.md:
        open(a.md, "w").write(to_md(rows) + "\n")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
