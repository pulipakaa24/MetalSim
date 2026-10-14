#!/usr/bin/env python3
"""Evaluate sand_fidelity batch 02a (research_notes/sand_fidelity/PLAN_2026-10-14.md sections 8 and 9) against the
predictions stated there before the runs (PREDICTIONS below), write ``runs/sand_fidelity/batch_02a/summary.md`` +
``summary.json``, and append (or replace) section 10 of the plan note between the markers
``<!-- sand_fidelity_02a results: start -->`` / ``end``.

    python scripts/sand_fidelity/evaluate_batch_02a.py [--results DIR] [--plan NOTE.md] [--no-plan]
    python scripts/sand_fidelity/evaluate_batch_02a.py --selftest

A run whose JSON is missing is "not run"; ``"error": "nonfinite"`` is a localised blow-up (a finding); any other
``error`` is reported as such. CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metalsim/physics/mpm/probes"))
import sand_metrics as SM  # noqa: E402

START = "<!-- sand_fidelity_02a results: start -->"
END = "<!-- sand_fidelity_02a results: end -->"
G = 9.81

PREDICTIONS = [
    ("Repeat check", "The 1.5 s box drop at dt 1/480 (l_slip) repeats batch 01's a4_dt480 within 0.1 mm at every step (bitwise "
             "reported separately): the probe changes add only reads."),
    ("S sphere (8.6)", "S-bed: walls reduce the final depth by >= 15 mm, and the unwalled ball-free control subsides >= 10 mm "
                       "at the centre within 0.2 s. S-res: depth / Uehara falls >= 30 % from 10 to 20 cells, in both "
                       "20-cell variants. S-coupling: kinematic F(16 mm) >= 13 N while the walled free drop still overshoots "
                       "Uehara by >= 1.5x; F(16 mm) <= 7 N means the bed itself is weak. Steel (7,800 kg/m^3): Uehara predicts "
                       "1.77x the 2,500 kg/m^3 depth; a sim ratio outside 1.77 +- 0.35 points at the coupling."),
    ("L lag (8.7)", "Impact-phase peak (within 0.3 s after free-fall contact t_c = sqrt(2h/g)) = d0 + k dt at each drop "
                    "height, with k within +-30 % of v = sqrt(2gh) (1.72 and 0.99 m/s), so k(0.05)/k(0.15) = 0.58 +- 0.2; "
                    "d0 at voxel 0.02 in 10-25 mm; d0 at voxel 0.01 = d0(0.02) + 5 +- 3 mm. Against: k independent of v."),
    ("C mu 0.4 ladder (8.3)", "H-edge: C1 (plane floor, frictionless side walls, fixed tilt 20.3 deg) goes non-finite "
                              "within the 8 s hold; C2 (box floor) or C3 (rough side walls) stays finite; C4 (no side "
                              "walls) stays finite. H-solver: C1-C4 all non-finite. C5 = C1 with logged iterations."),
    ("T tilt and iterations (8.1, 8.2)", "Runaway (surface speed > 10 mm/s and rising for 0.5 s) within atan(mu) +- 1 deg "
                                         "and the deposit angle within 1 deg below it (mu 0.68, 0.9). >= 95 % of logged "
                                         "solves stop at the iteration cap. At a fixed tilt of atan(mu) - 4 deg (mu 0.68) "
                                         "the creep speed at 1000 iterations is <= half that at 250. Resting box at voxel "
                                         "0.01, 1000 iterations: creep <= half of 250's 0.41 mm/s, settlement 0.45 +- 0.1 "
                                         "voxel."),
]


def load(d, k):
    f = d / f"{k}.json"
    if not f.exists():
        return None
    try:
        return json.load(open(f))
    except Exception as e:                                  # noqa: BLE001
        return {"error": f"unreadable: {e}"}


def ok(r):
    return isinstance(r, dict) and "error" not in r


def f1(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def mm(x):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{1000 * x:.1f}"


def status(r):
    return "not run" if r is None else ("NON-FINITE" if r.get("error") == "nonfinite" else r.get("error", "ok"))


def uehara(r, mu=0.7):
    return SM.uehara_solve(mu, r["diameter"], r["h"], r["ball_density"] / 2500.0)


# ------------------------------------------------------------------------------------------------------------ arms


def arm_s(d, b01):
    notes, res, rows = [], {}, []
    runs = {k: load(d, k) for k in ("s_walled", "s_ctrl_walled", "s_ctrl_unwalled", "s_kin", "s_steel", "s20a", "s20b")}
    base = load(b01, "a6_sphere") if b01 else None
    for k, r in list(runs.items()) + [("a6_sphere (batch 01, unwalled)", base)]:
        if not ok(r):
            rows.append(f"| {k} | {status(r)} ||||||")
            continue
        if r.get("ball") is False:
            rows.append(f"| {k} | control | {r['particles']} | — | — | centre subsidence {mm(r.get('centre_subsidence_0p2s_m'))} "
                        f"mm @0.2 s, {mm(r.get('centre_subsidence_m'))} mm end | |")
        elif r.get("kinematic"):
            rows.append(f"| {k} | kinematic {r['kinematic']} m/s | {r['particles']} | — | — | F(16 mm) "
                        f"{f1(r.get('force_at_16mm_N'))} N, F(30 mm) {f1(r.get('force_at_30mm_N'))} N | |")
        else:
            u = uehara(r)
            rows.append(f"| {k} | D {r['diameter']} m ({f1(r['diameter_cells'], 0)} cells), {r['ball_density']:.0f} kg/m^3"
                        f"{', walls' if r.get('walls') else ''} | {r['particles']} | {mm(r['final_penetration_m'])} | "
                        f"{mm(u)} | ratio {r['final_penetration_m'] / u:.2f}; stopped {r['stopped']} | "
                        f"{mm(r.get('centre_subsidence_m'))} |")
    w, c_w, c_u = runs["s_walled"], runs["s_ctrl_walled"], runs["s_ctrl_unwalled"]
    if ok(w) and ok(base):
        dd = base["final_penetration_m"] - w["final_penetration_m"]
        sub = c_u.get("centre_subsidence_0p2s_m") if ok(c_u) else None
        res["S_bed"] = "supported" if dd >= 0.015 and sub is not None and sub >= 0.010 else (
            "not supported" if dd < 0.005 else "partial")
        notes.append(f"S-bed: walls change the depth by {-1000 * dd:+.1f} mm (unwalled {mm(base['final_penetration_m'])} -> "
                     f"walled {mm(w['final_penetration_m'])}); unwalled control subsides {mm(sub)} mm at 0.2 s, walled "
                     f"{mm(c_w.get('centre_subsidence_0p2s_m') if ok(c_w) else None)} mm -> {res['S_bed']}.")
    ratios = {k: runs[k]["final_penetration_m"] / uehara(runs[k]) for k in ("s_walled", "s20a", "s20b") if ok(runs[k])}
    if "s_walled" in ratios and len(ratios) > 1:
        drops = {k: 1 - ratios[k] / ratios["s_walled"] for k in ratios if k != "s_walled"}
        res["S_res"] = "supported" if all(v >= 0.3 for v in drops.values()) else (
            "not supported" if all(v < 0.1 for v in drops.values()) else "partial")
        notes.append("S-res: depth/Uehara 10 cells " + f"{ratios['s_walled']:.2f}; 20 cells " +
                     ", ".join(f"{k} {ratios[k]:.2f} ({100 * drops[k]:+.0f} %)" for k in drops) + f" -> {res['S_res']}.")
    k = runs["s_kin"]
    if ok(k) and ok(w):
        f16 = k["force_at_16mm_N"]
        over = ratios.get("s_walled", float("nan"))
        res["S_coupling"] = ("coupling (bed strong, drop overshoots)" if f16 >= 13 and over >= 1.5 else
                             "bed weak" if f16 <= 7 else "undecided")
        notes.append(f"S-coupling: kinematic F(16 mm) = {f16:.1f} N (Uehara-implied mean 26 N), walled drop "
                     f"{over:.2f}x Uehara -> {res['S_coupling']}.")
    s = runs["s_steel"]
    if ok(s) and ok(w):
        rr = s["final_penetration_m"] / w["final_penetration_m"]
        res["steel_ratio"] = rr
        notes.append(f"Steel / glass-density depth ratio {rr:.2f} (Uehara 1.77): "
                     f"{'consistent' if abs(rr - 1.77) <= 0.35 else 'off: mass-dependent error, points at coupling'}.")
    head = "| run | case | particles | final mm | Uehara mm (mu 0.7) | | centre subsidence mm |\n|---|---|---|---|---|---|---|"
    return "\n".join([head] + rows), notes, res


def arm_l(d, b01):
    notes, res, rows = [], {}, []
    fits = {}
    for h, tag in ((0.15, "h15"), (0.05, "h05")):
        pts = []
        for fps in (120, 240, 480, 960):
            k = f"l_{tag}_dt{fps}"
            r = load(d, k)
            if not ok(r):
                rows.append(f"| {k} | {status(r)} |||||")
                continue
            ip = SM.impact_phase(r)
            pts.append((1.0 / fps, ip["peak_m"]))
            rows.append(f"| {k} | {h} | 1/{fps} | {r['voxel']} | {mm(ip['peak_m'])} | {f1(r.get('max_tilt_deg'))} |")
        if len(pts) >= 3:
            a, s = SM.linear_fit(*zip(*pts))
            v = math.sqrt(2 * G * h)
            fits[h] = (a, s, v)
            notes.append(f"h {h} m: d0 {1000 * a:.1f} mm, k {s:.2f} m/s vs v {v:.2f} m/s "
                         f"({'within' if abs(s - v) <= 0.3 * v else 'outside'} +-30 %).")
    if 0.15 in fits and 0.05 in fits:
        (a1, k1, v1), (a2, k2, v2) = fits[0.15], fits[0.05]
        ratio = k2 / k1 if k1 else float("nan")
        within = all(abs(k - v) <= 0.3 * v for _, k, v in fits.values())
        res["H_lag"] = "supported" if within and abs(ratio - 0.58) <= 0.2 and 0.010 <= a1 <= 0.025 else (
            "against (k independent of v)" if abs(ratio - 1.0) <= 0.15 else "not supported")
        res.update(d0_h15_mm=1000 * a1, k_ratio=ratio)
        notes.append(f"k(0.05)/k(0.15) = {ratio:.2f} (prediction 0.58 +- 0.2) -> H_lag {res['H_lag']}.")
    v1 = [load(d, f"l_v01_dt{f}") for f in (240, 960)]
    if all(ok(r) for r in v1) and 0.15 in fits:
        a, s = SM.linear_fit([1 / 240, 1 / 960], [SM.impact_phase(r)["peak_m"] for r in v1])
        dd = a - fits[0.15][0]
        res["voxel_d0_shift_mm"] = 1000 * dd
        notes.append(f"Voxel 0.01: d0 {1000 * a:.1f} mm, {1000 * dd:+.1f} mm vs voxel 0.02 (prediction +5 +- 3).")
        for r, f in zip(v1, (240, 960)):
            rows.append(f"| l_v01_dt{f} | 0.15 | 1/{f} | 0.01 | {mm(SM.impact_phase(r)['peak_m'])} | {f1(r.get('max_tilt_deg'))} |")
    r, ref = load(d, "l_slip"), load(b01, "a4_dt480") if b01 else None
    if ok(r) and ok(ref):
        same = r["bottom"] == ref["bottom"]
        n = min(len(r["bottom"]), len(ref["bottom"]))
        diff = float(np.max(np.abs(np.asarray(r["bottom"][:n]) - np.asarray(ref["bottom"][:n])))) if n else float("inf")
        # MuJoCo Warp's default mode is not guaranteed run-to-run bitwise (docs/NOW.md), so the check passes within
        # 0.1 mm and reports bitwise equality separately
        res["gate"] = diff <= 1e-4
        res["bitwise"] = same
        late = r["final_penetration_m"] - SM.impact_phase(r)["peak_m"]
        notes.append(f"Repeat of batch 01 a4_dt480: max |diff| {1000 * diff:.3f} mm ({'bitwise' if same else 'not bitwise'}) "
                     f"-> {'PASS' if res['gate'] else 'FAIL'}; late sink {1000 * late:.1f} mm, "
                     f"max box tilt {f1(r.get('max_tilt_deg'))} deg ({'a rotation' if r.get('max_tilt_deg', 0) > 2 else 'no rotation'}).")
    head = "| run | h m | dt | voxel | impact-phase peak mm | max tilt deg |\n|---|---|---|---|---|---|"
    return "\n".join([head] + rows), notes, res


def arm_c(d):
    notes, res, rows = [], {}, []
    out = {}
    for k, desc in (("c1", "plane floor, frictionless sides"), ("c2", "box floor"), ("c3", "rough side walls"),
                    ("c4", "no side walls (width 0.1)"), ("c5", "C1, iterations logged")):
        r = load(d, k)
        if r is None:
            rows.append(f"| {k} | {desc} | not run |||")
            continue
        nf = r.get("error") == "nonfinite"
        out[k] = nf
        w = r.get("where", {}) if nf else r.get("fast_particles", {})
        sv = r.get("solves") or {}
        rows.append(f"| {k} | {desc} | {'NON-FINITE after ' + f1(r.get('hold_seconds_before_nan'), 2) + ' s of hold' if nf else ('finite' if 'error' not in r else r['error'])} | "
                    f"n {w.get('n_bad')}, near floor edge {f1(w.get('frac_near_floor_edge'), 2)}, below floor "
                    f"{f1(w.get('frac_below_floor'), 2)} | {f1(sv.get('frac_at_cap'), 2) if sv else '—'} |")
    if {"c1", "c2", "c3", "c4"} <= out.keys():
        if out["c1"] and (not out["c2"] or not out["c3"]) and not out["c4"]:
            res["cause"] = "H-edge (floor / side-wall junction)"
        elif all(out[k] for k in ("c1", "c2", "c3", "c4")):
            res["cause"] = "H-solver (blow-up without the edge): minimal repro = C4; candidate for an upstream issue"
        elif not out["c1"]:
            res["cause"] = "not reproduced at a fixed tilt (history-dependent)"
        else:
            res["cause"] = "mixed: " + ", ".join(f"{k} {'NaN' if v else 'finite'}" for k, v in sorted(out.items()))
        notes.append(f"Cause: {res['cause']}.")
    head = "| run | variant | outcome | where (last finite sample / fast particles) | solves at cap |\n|---|---|---|---|---|"
    return "\n".join([head] + rows), notes, res


def creep_speed(r, lo, hi):
    h = r.get("history", {})
    t, th, v = (np.asarray(h.get(x, []), float) for x in ("t", "theta", "surface_speed_p90"))
    hold = r.get("fixed_tilt")
    m = (th >= lo) & (th <= hi) if hold is None else (t >= (0.3 + hold / 20.0 + 0.5))
    return float(np.median(v[m])) if m.any() else float("nan")


def arm_t(d):
    notes, res, rows = [], {}, []
    for k in ("t1", "t2"):
        r = load(d, k)
        if not ok(r):
            rows.append(f"| {k} | {status(r)} ||||||")
            continue
        am, tf, ts = r["atan_mu_deg"], r.get("theta_fail_deg"), r.get("theta_stop_deg")
        pr = tf is not None and abs(tf - am) <= 1 and ts is not None and 0 <= tf - ts <= 1.0
        res[f"{k}_runaway_pass"] = pr
        sv = r.get("solves") or {}
        rows.append(f"| {k} | {r['friction']} | {f1(am)} | creep {f1(r.get('theta_creep_deg'))} / runaway {f1(tf)} | "
                    f"{f1(ts)} | {r['at_rest']} | {f1(sv.get('frac_at_cap'), 2) if sv else '—'} |")
        fp = r.get("fast_particles", {})
        if fp.get("n_bad"):
            notes.append(f"{k}: {fp['n_bad']} particles faster than 0.1 m/s at the end; near the floor edge "
                         f"{f1(fp.get('frac_near_floor_edge'), 2)}, below the floor {f1(fp.get('frac_below_floor'), 2)}.")
    caps = [r["solves"]["frac_at_cap"] for r in (load(d, k) for k in ("t1", "c5", "r250")) if isinstance(r, dict)
            and (r.get("solves") or {}).get("solves")]
    if caps:
        res["frac_at_cap_min"] = min(caps)
        notes.append(f"Solves at the 250 cap: {', '.join(f'{c:.2f}' for c in caps)} (prediction >= 0.95) -> "
                     f"{'cap-bound' if min(caps) >= 0.95 else 'converges in part'}.")
    a, b = load(d, "t3b"), load(d, "t3")
    if ok(a) and ok(b):
        va, vb = creep_speed(a, 0, 90), creep_speed(b, 0, 90)
        res["creep_ratio_250_over_1000"] = va / vb if vb else float("nan")
        notes.append(f"Creep at atan(mu) - 4 deg: 250 iterations {1000 * va:.2f} mm/s, 1000 iterations {1000 * vb:.2f} mm/s, "
                     f"ratio {va / vb:.2f} (prediction >= 2: under-convergence) -> "
                     f"{'supported' if va / vb >= 2 else 'not supported'}.")
    r1, r0 = load(d, "r1000"), load(d, "r250")
    if ok(r1):
        cr = r1["creep_m_per_s"]
        ref = r0["creep_m_per_s"] if ok(r0) else 0.00041
        notes.append(f"Resting box, voxel 0.01, 1000 iterations: settlement {mm(r1['final_penetration_m'])} mm "
                     f"({f1(r1['settlement_over_voxel'], 2)} voxel), creep {1000 * cr:.2f} vs {1000 * ref:.2f} mm/s at 250 -> "
                     f"{'halved' if cr <= 0.5 * ref else 'not halved'}.")
    head = ("| run | mu | atan mu | creep / runaway onset deg | deposit deg | at rest | solves at cap |\n"
            "|---|---|---|---|---|---|---|")
    return "\n".join([head] + rows), notes, res


def evaluate(results, plan, b01):
    parts = [("S. Sphere (8.6)", arm_s(results, b01)), ("L. Lag (8.7)", arm_l(results, b01)),
             ("C. mu 0.4 ladder (8.3)", arm_c(results)), ("T. Tilt and iterations (8.1, 8.2)", arm_t(results))]
    lines = ["# sand_fidelity batch 02a: results", "",
             "Plan: research_notes/sand_fidelity/PLAN_2026-10-14.md sections 8-9. 16-inch, correctness runs.", "",
             "## Predictions stated before the runs", ""]
    lines += [f"* **{n}.** {t}" for n, t in PREDICTIONS] + [""]
    summary = {}
    for title, (table, notes, res) in parts:
        summary[title] = res
        lines += [f"## {title}", "", table, ""] + [f"* {x}" for x in notes] + [""]
    text = "\n".join(lines)
    (results / "summary.md").write_text(text)
    json.dump(summary, open(results / "summary.json", "w"), indent=1, default=str)
    if plan is not None:
        note = plan.read_text() if plan.exists() else ""
        body = "\n".join(["## 10. Results: batch sand_fidelity_02a", "",
                          "Written by scripts/sand_fidelity/evaluate_batch_02a.py; tables in "
                          "runs/sand_fidelity/batch_02a/summary.md.", ""] +
                         [x for x in lines[lines.index("## Predictions stated before the runs"):] if not x.startswith("|")])
        block = f"{START}\n{body}\n{END}\n"
        if START in note and END in note:
            note = note[:note.index(START)] + block + note[note.index(END) + len(END) + 1:]
        else:
            note = note.rstrip("\n") + "\n\n" + block
        plan.write_text(note)
    return summary


def _synthetic(d, b01):
    w = lambda k, r: json.dump(r, open(d / f"{k}.json", "w"))      # noqa: E731
    ball = lambda pen, **kw: {"probe": "sphere_drop", "ball": True, "kinematic": None, "walls": True, "h": 0.2,  # noqa: E731
                              "diameter": 0.05, "diameter_cells": 10, "ball_density": 2500.0, "particles": 307200,
                              "final_penetration_m": pen, "stopped": True, "centre_subsidence_m": 0.001, **kw}
    w("s_walled", ball(0.040))
    w("s_ctrl_walled", {"probe": "sphere_drop", "ball": False, "particles": 1, "centre_subsidence_0p2s_m": 0.001,
                        "centre_subsidence_m": 0.001})
    w("s_ctrl_unwalled", {"probe": "sphere_drop", "ball": False, "particles": 1, "centre_subsidence_0p2s_m": 0.014,
                          "centre_subsidence_m": 0.016})
    w("s_kin", {"probe": "sphere_drop", "ball": True, "kinematic": 0.02, "particles": 1, "force_at_16mm_N": 18.0,
                "force_at_30mm_N": 30.0})
    w("s_steel", ball(0.070, ball_density=7800.0))
    w("s20a", ball(0.026, diameter_cells=20))
    w("s20b", {"error": "rc=1 sort scratch"})
    json.dump(ball(0.0604, walls=False), open(b01 / "a6_sphere.json", "w"))
    bottom = [0.08 - 0.001 * i for i in range(10)]
    json.dump({"bottom": bottom, "final_penetration_m": 0.024, "fps": 480, "depth": 0.08, "drop": 0.15},
              open(b01 / "a4_dt480.json", "w"))
    for h, tag in ((0.15, "h15"), (0.05, "h05")):
        v = math.sqrt(2 * G * h)
        for fps in (120, 240, 480, 960):
            w(f"l_{tag}_dt{fps}", {"fps": fps, "voxel": 0.02, "drop": h, "peak_penetration_m": 0.016 + v / fps,
                                   "max_tilt_deg": 0.1})
    for fps in (240, 960):
        w(f"l_v01_dt{fps}", {"fps": fps, "voxel": 0.01, "drop": 0.15, "peak_penetration_m": 0.021 + 1.72 / fps})
    w("l_slip", {"bottom": bottom, "final_penetration_m": 0.024, "fps": 480, "depth": 0.08, "drop": 0.15,
                 "peak_penetration_m": 0.019, "max_tilt_deg": 0.3})
    for k, nf in (("c1", True), ("c2", False), ("c3", True), ("c4", False), ("c5", True)):
        w(k, {"error": "nonfinite", "hold_seconds_before_nan": 3.1, "where": {"n_bad": 9, "frac_near_floor_edge": 0.8,
                                                                               "frac_below_floor": 0.5}} if nf else
          {"probe": "tilt_box", "fast_particles": {"n_bad": 0}})
    for k, mu, tf, ts in (("t1", 0.68, 34.0, 33.6), ("t2", 0.9, 42.3, 41.6)):
        w(k, {"friction": mu, "atan_mu_deg": math.degrees(math.atan(mu)), "theta_fail_deg": tf, "theta_creep_deg": tf - 1,
              "theta_stop_deg": ts, "at_rest": True, "solves": {"solves": 100, "frac_at_cap": 0.99} if k == "t1" else None,
              "fast_particles": {"n_bad": 3, "frac_near_floor_edge": 1.0, "frac_below_floor": 0.3}})
    hist = lambda v: {"t": [3.0, 3.5, 4.0], "theta": [30.2] * 3, "surface_speed_p90": [v] * 3}   # noqa: E731
    w("t3b", {"fixed_tilt": 30.2, "history": hist(0.001)})
    w("t3", {"fixed_tilt": 30.2, "history": hist(0.0004)})
    w("r250", {"final_penetration_m": 0.0045, "settlement_over_voxel": 0.45, "creep_m_per_s": 0.0004,
               "solves": {"solves": 180, "frac_at_cap": 0.97}})
    w("r1000", {"final_penetration_m": 0.0044, "settlement_over_voxel": 0.44, "creep_m_per_s": 0.0001})


def selftest():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
        d, b01 = pathlib.Path(t) / "r", pathlib.Path(t) / "b01"
        d.mkdir(), b01.mkdir()
        _synthetic(d, b01)
        plan = pathlib.Path(t) / "plan.md"
        plan.write_text("# plan\n")
        s = evaluate(d, plan, b01)
        evaluate(d, plan, b01)
        note = plan.read_text()
        S, L, C, T = (s[k] for k in s)
        checks = {"S_bed": S.get("S_bed") == "supported", "S_res": S.get("S_res") == "supported",
                  "S_coupling": S.get("S_coupling", "").startswith("coupling"), "gate": L.get("gate") is True,
                  "H_lag": L.get("H_lag") == "supported", "cause": C.get("cause", "").startswith("H-edge"),
                  "cap": T.get("frac_at_cap_min", 0) >= 0.95, "creep": T.get("creep_ratio_250_over_1000", 0) >= 2,
                  "plan once": note.count(START) == 1}
        bad = [k for k, v in checks.items() if not v]
        print("selftest:", "PASS" if not bad else f"FAIL {bad}")
        if bad:
            print((d / "summary.md").read_text())
        return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "runs/sand_fidelity/batch_02a"))
    ap.add_argument("--batch01", default=str(ROOT / "runs/sand_fidelity/batch_01"))
    ap.add_argument("--plan", default=str(ROOT / "research_notes/sand_fidelity/PLAN_2026-10-14.md"))
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    s = evaluate(pathlib.Path(a.results), None if a.no_plan else pathlib.Path(a.plan), pathlib.Path(a.batch01))
    print(json.dumps(s, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
