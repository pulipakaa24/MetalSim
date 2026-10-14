#!/usr/bin/env python3
"""Evaluate sand_fidelity batch 01 (research_notes/sand_fidelity/PLAN_2026-10-14.md section 6) against the
predictions stated before the runs (the PREDICTIONS table below, repeated in summary.md), write
``runs/sand_fidelity/batch_01/summary.md`` + ``summary.json``, and append (or replace) a results section in the plan
note between the markers ``<!-- sand_fidelity_01 results: start -->`` / ``end``.

    python scripts/sand_fidelity/evaluate_batch_01.py [--results DIR] [--plan NOTE.md] [--no-plan]
    python scripts/sand_fidelity/evaluate_batch_01.py --selftest      # synthetic data in a temporary directory

Missing runs are reported as "not run"; a run whose JSON has an "error" key or whose job failed (jobs.tsv) is
reported as such. Pure standard library + numpy; CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metalsim/physics/mpm/probes"))
import sand_metrics as SM  # noqa: E402

START = "<!-- sand_fidelity_01 results: start -->"
END = "<!-- sand_fidelity_01 results: end -->"

# reference values from earlier batches (runs/mpm_main/...)
REF_REPOSE_DX10 = 26.33          # repose3/mu068_dx005_v025.json, fit mean
REF_REPOSE_DX20 = 27.75          # repose4/mu068_dx0025_v025.json
REF_REST_V02 = 0.0102            # bearing3/rest_lagged_i250.json (250 it, 1e-6), m
REF_DROP_120 = 0.0298            # bearing4/drop_lagged_i250_t1e6.json, peak, m

PREDICTIONS = [
    ("1 repose profile + stress (R1, R3)",
     "Gate: R/dx 10 baseline reproduces the earlier run (fit 26.33 +- 0.3 deg). "
     "R1 (convex crest, measurement) predicts: lower-flank band angle (5-50 % of h_max) >= fit + 2 deg and within 2 deg "
     "of the cone-volume angle; the taller fill raises the fit by >= 1.5 deg and shrinks the estimator spread by >= 30 %. "
     "Against R1: band angles within 1 deg of each other and the tall fill within +-0.7 deg of the baseline. "
     "R3 (Drucker-Prager stress geometry) predicts: flank surface shells (0-2 cells) at yield (median sqrt(J2)/(mu p) "
     ">= 0.95) while the local angle is >= 3 deg below atan(mu), with a Lode parameter off pure shear (|cos 3theta| > "
     "0.3). Flank surface below yield (median < 0.9) means the heap stopped for a non-rheological reason (R2)."),
    ("2 tilting box (B1, R3)",
     "Correct plane-shear rheology: theta_stop (post-avalanche surface angle) within +-1 deg of atan(mu) at 10 cells "
     "deep, +-2 deg at 5 cells (plan B1, applied to theta_stop). theta_fail is reported against the plan's +-1 deg but "
     "is not decisive (rigid-plastic onset is not unique; see tilt_box.py). Discrimination with arm 1: theta_stop at "
     "atan(mu) while the heap is >= 3 deg low -> deficit is heap-specific (R1/R3); theta_stop as low as the heap "
     "(within 2 deg of the heap's fit, >= 3 deg below atan(mu)) -> general (R2 discretization or R4 solver)."),
    ("3 basis variants (R2)",
     "R2 (weak free-surface skin from P0 strain / Q1 velocity / PIC quadrature / ppc) predicts at least one variant "
     "raises the fit by >= 1 deg over the R/dx 10 baseline. All variants within +-0.5 deg: R2 not supported by these "
     "settings. P1d/Q1d with a sector sd > 1 deg: the partially-filled-cell strain defect fixed upstream in Newton "
     "draft PR #4660, not physics."),
    ("4 drop dt sweep (C1)",
     "One-step exchange lag (H_lag): peak penetration linear in dt with slope within +-30 % of the impact speed (1.72 "
     "m/s) and intercept d0 in 10-25 mm; d(1/480) - d(1/960) < 3 mm. Against: slope < 0.5 v (the lag is not the "
     "cause) or d0 > 25 mm (bed too weak, R3/C4). Voxel 0.01 at 1/480 within 3 mm of voxel 0.02: penetration not set "
     "by grid contact resolution. dt 1/120 reproduces 29.8 mm (+-1 mm)."),
    ("5 resting box (C2)",
     "Gate: voxel 0.02 S2 without projection reproduces 10.2 mm (+-1 mm). H_grid (sub-voxel contact resolution, "
     "Newton #3473): settlement / voxel constant (within +-30 %) over voxel 0.02 / 0.01 / 0.005, and the box overlaps "
     "the particles (deepest particle inside the box >= half the settlement); project_outside then removes the overlap. "
     "H_bed (real bed deformation): settlement in mm constant (+-30 %) over the voxels, small overlap, ring heave > 0. "
     "pic27 (Newton #1251's advice) reduces settlement below S2 at every voxel only under H_grid. "
     "Plan B5 pass: settlement <= 0.25 voxel and <= 3 mm, creep < 0.1 mm/s."),
    ("6 sphere drop smoke (B4)",
     "Smoke pass: finite, the ball stops (|v| < 1 cm/s) within 0.5 s, never reaches the floor, final penetration within "
     "0.5-2x Uehara's law with mu = 0.7 (h = 0.2 m, D = 0.05 m: ~16 mm). Reported, not judged: d - v dt (lag "
     "removed, from arm 4) against Uehara with mu = tan(heap angle of arm 1) (~22 mm) and mu = 0.7 (~16 mm)."),
]


def load(d: pathlib.Path, name: str):
    f = d / f"{name}.json"
    if not f.exists():
        return None
    try:
        return json.load(open(f))
    except Exception as e:                                  # noqa: BLE001
        return {"error": f"unreadable: {e}"}


def jobs(d: pathlib.Path) -> dict:
    out = {}
    f = d / "jobs.tsv"
    if f.exists():
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[0] != "name":
                out[p[0]] = {"rc": p[1], "seconds": p[2], "est_seconds": p[3]}
    return out


def ok(r) -> bool:
    return isinstance(r, dict) and "error" not in r


def f1(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def mm(x):
    return "—" if x is None else f"{1000 * x:.1f}"


def verdict(cond, yes, no, undecided="undecided"):
    return undecided if cond is None else (yes if cond else no)


def _combine(votes, yes, no):
    """All True -> yes; all False -> no; empty -> 'not run'; anything else -> 'mixed / undecided'."""
    if not votes:
        return "not run"
    if all(v is True for v in votes):
        return yes
    if all(v is False for v in votes):
        return no
    return "mixed / undecided"


# ------------------------------------------------------------------------------------------------------------ arms


def arm1(d):
    rows, notes, res = [], [], {}
    runs = {k: load(d, k) for k in ("a1_dx10", "a1_dx20", "a1_tall")}
    for k, r in runs.items():
        if not ok(r):
            rows.append(f"| {k} | {'not run' if r is None else r.get('error')} |||||||||")
            continue
        fin = r["final"]
        bands = r.get("height_bands", {}).get("bands", [])
        bstr = " / ".join(f1(b["angle_deg"]) for b in bands)
        fs = r.get("flank_stress", {})
        sh = fs.get("shells", [])
        s0 = sh[0] if sh else {}
        rows.append(f"| {k} | {f1(r['R_over_dx'], 0)} | {100 * r.get('fill', 0.05):.0f} | {r['particles']} | "
                    f"{f1(fin['angle_fit_mean_deg'])} ± {f1(fin['angle_fit_sd_deg'])} | {f1(fin['angle_cone_volume_deg'])} | "
                    f"{f1(fin['angle_height_flank_deg'])} | {bstr} | {f1(r.get('height_bands', {}).get('max_local_angle_deg'))} | "
                    f"{f1(s0.get('ratio_median'), 2)} / {f1(s0.get('lode_median'), 2)} | {r['at_rest']} |")
    base = runs["a1_dx10"]
    gate = None
    if ok(base):
        gate = abs(base["final"]["angle_fit_mean_deg"] - REF_REPOSE_DX10) <= 0.3
        notes.append(f"Gate (reproduces repose3 {REF_REPOSE_DX10} deg within 0.3): **{'PASS' if gate else 'FAIL'}** "
                     f"({f1(base['final']['angle_fit_mean_deg'], 2)} deg).")
    res["gate"] = gate
    # R1
    r1_votes = []
    for k in ("a1_dx10", "a1_dx20"):
        r = runs[k]
        if not ok(r):
            continue
        b = r.get("height_bands", {}).get("bands", [])
        low = [x["angle_deg"] for x in b[:2] if math.isfinite(x["angle_deg"])]
        if not low:
            continue
        lowa = max(low)
        fit, cone = r["final"]["angle_fit_mean_deg"], r["final"]["angle_cone_volume_deg"]
        c = lowa >= fit + 2 and abs(lowa - cone) <= 2
        r1_votes.append(c)
        notes.append(f"{k}: lower-flank band angle {f1(lowa)} vs fit {f1(fit)} / cone volume {f1(cone)} -> "
                     f"{'convex (R1 prediction met)' if c else 'R1 prediction not met'}.")
    if ok(base) and ok(runs["a1_tall"]):
        t = runs["a1_tall"]
        dfit = t["final"]["angle_fit_mean_deg"] - base["final"]["angle_fit_mean_deg"]
        shrink = 1 - t["estimators_spread_deg"] / max(base["estimators_spread_deg"], 1e-9)
        c = dfit >= 1.5 and shrink >= 0.3
        against = abs(dfit) <= 0.7
        r1_votes.append(c if c or against else None)
        notes.append(f"Tall fill: fit {dfit:+.2f} deg, estimator spread {100 * shrink:+.0f} % smaller -> "
                     f"{'R1 met' if c else ('against R1' if against else 'in between')}.")
    res["R1"] = _combine(r1_votes, "supported", "not supported")
    # R3 (stress)
    r3 = []
    for k in ("a1_dx10", "a1_dx20"):
        r = runs[k]
        if not ok(r) or "shells" not in r.get("flank_stress", {}):
            continue
        sh = r["flank_stress"]["shells"]
        surf = [s for s in sh[:2] if s.get("n")]
        if not surf:
            continue
        ratio = float(sum(s["ratio_median"] * s["n"] for s in surf) / sum(s["n"] for s in surf))
        lode = float(sum(s["lode_median"] * s["n"] for s in surf) / sum(s["n"] for s in surf))
        low = r["atan_mu_deg"] - r["final"]["angle_fit_mean_deg"] >= 3
        if ratio >= 0.95 and low and abs(lode) > 0.3:
            v = "R3 consistent (at yield below atan mu, off pure shear)"
            r3.append(True)
        elif ratio < 0.9:
            v = "flank below yield: stopped for a non-rheological reason (points to R2)"
            r3.append(False)
        else:
            v = "at yield but Lode near pure shear, or in between" if ratio >= 0.95 else "in between (0.9-0.95)"
            r3.append(None)
        notes.append(f"{k}: flank surface (0-2 cells) yield ratio {ratio:.2f}, Lode cos3theta {lode:+.2f}, "
                     f"p_base/(rho g h) {f1(r['flank_stress'].get('p_base_over_rho_g_h'), 2)} -> {v}.")
    res["R3_stress"] = _combine(r3, "consistent", "not supported")
    head = ("| run | R/dx | fill cm | particles | fit (sd) | cone vol. | h/flank | band angles 5-25/25-50/50-75/75-95 % | "
            "max local | flank 0-1 cell ratio / Lode | at rest |\n|---|---|---|---|---|---|---|---|---|---|---|")
    return "\n".join([head] + rows), notes, res


def arm2(d, heap_fit):
    rows, notes, res = [], [], {"theta_stop": {}, "theta_fail": {}}
    for k in ("a2_mu040", "a2_mu068", "a2_mu090", "a2_mu068_v01"):
        r = load(d, k)
        if not ok(r):
            rows.append(f"| {k} | {'not run' if r is None else r.get('error')} |||||||")
            continue
        tol = 1.0 if r["depth_cells"] >= 9.5 else 2.0
        ts, tf, am = r["theta_stop_deg"], r["theta_fail_deg"], r["atan_mu_deg"]
        pass_stop = (ts is not None and math.isfinite(ts) and abs(ts - am) <= tol) if r["at_rest"] else None
        pass_fail = (tf is not None and abs(tf - am) <= tol) if r["failed"] else False
        res["theta_stop"][k] = pass_stop
        res["theta_fail"][k] = pass_fail
        rows.append(f"| {k} | {r['friction']} | {f1(am)} | {f1(r['depth_cells'], 0)} | {f1(tf)} | {f1(ts)} | "
                    f"{r['at_rest']} | {verdict(pass_stop, 'PASS', 'FAIL', 'not at rest')} / "
                    f"{verdict(pass_fail, 'PASS', 'FAIL')} |")
    r = load(d, "a2_mu068")
    if ok(r) and heap_fit is not None and r["theta_stop_deg"] is not None and math.isfinite(r["theta_stop_deg"]):
        ts, am = r["theta_stop_deg"], r["atan_mu_deg"]
        if abs(ts - am) <= 1 and am - heap_fit >= 3:
            v = "theta_stop at atan(mu), heap low: deficit is heap-specific (R1 / R3), plane-shear rheology right"
            res["discrimination"] = "heap-specific"
        elif am - ts >= 3 and abs(ts - heap_fit) <= 2:
            v = "theta_stop as low as the heap: general deficit (R2 discretization or R4 solver)"
            res["discrimination"] = "general"
        else:
            v = "in between: neither prediction met cleanly"
            res["discrimination"] = "undecided"
        notes.append(f"mu 0.68: theta_stop {ts:.1f} vs atan(mu) {am:.1f} vs heap fit (R/dx 20 if run) {heap_fit:.1f} -> {v}.")
    head = ("| run | mu | atan mu | depth cells | theta_fail | theta_stop | at rest | stop / fail vs atan mu |\n"
            "|---|---|---|---|---|---|---|---|")
    return "\n".join([head] + rows), notes, res


def arm3(d):
    rows, notes, res = [], [], {}
    base = load(d, "a1_dx10")
    b = base["final"]["angle_fit_mean_deg"] if ok(base) else REF_REPOSE_DX10
    deltas = {}
    for k in ("a3_b2", "a3_q1d", "a3_p1d", "a3_gimp", "a3_ppc64"):
        r = load(d, k)
        if not ok(r):
            rows.append(f"| {k} | {'not run' if r is None else r.get('error')} ||||||")
            continue
        fin = r["final"]
        dd = fin["angle_fit_mean_deg"] - b
        deltas[k] = dd
        noisy = k in ("a3_q1d", "a3_p1d") and fin["angle_fit_sd_deg"] > 1.0
        rows.append(f"| {k} | {r['velocity_basis']} / {r['strain_basis']} / {r['integration_scheme']} / {r['ppc']} | "
                    f"{f1(fin['angle_fit_mean_deg'])} ± {f1(fin['angle_fit_sd_deg'])} | {dd:+.2f} | "
                    f"{f1(fin['angle_cone_volume_deg'])} | {r['at_rest']} | {'#4660 noise' if noisy else ''} |")
    if deltas:
        if any(v >= 1.0 for v in deltas.values()):
            res["R2"] = "supported by " + ", ".join(k for k, v in deltas.items() if v >= 1.0)
        elif all(abs(v) <= 0.5 for v in deltas.values()):
            res["R2"] = "not supported by these settings"
        else:
            res["R2"] = "undecided (changes between 0.5 and 1 deg)"
        notes.append(f"Baseline fit {b:.2f} deg (a1_dx10). R2: {res['R2']}.")
    head = ("| run | velocity / strain / integration / ppc | fit (sd) | vs baseline | cone vol. | at rest | note |\n"
            "|---|---|---|---|---|---|---|")
    return "\n".join([head] + rows), notes, res


def arm4(d):
    rows, notes, res = [], [], {}
    pts = []
    for k in ("a4_dt120", "a4_dt240", "a4_dt480", "a4_dt960", "a4_dt480_v01"):
        r = load(d, k)
        if not ok(r):
            rows.append(f"| {k} | {'not run' if r is None else r.get('error')} |||||||")
            continue
        rows.append(f"| {k} | 1/{r['fps']} | {r['voxel']} | {f1(r.get('impact_speed_m_s'), 2)} | {mm(r['peak_penetration_m'])} | "
                    f"{mm(r['final_penetration_m'])} | {f1(1000 * r['creep_m_per_s'], 2)} | {r['reached_floor']} |")
        if r["voxel"] == 0.02:
            pts.append((r.get("dt", 1.0 / r["fps"]), r["peak_penetration_m"], r.get("impact_speed_m_s") or 1.716))
    if len(pts) >= 3:
        a, s = SM.linear_fit([p[0] for p in pts], [p[1] for p in pts])
        v = sum(p[2] for p in pts) / len(pts)
        c_slope = abs(s - v) <= 0.3 * v
        c_d0 = 0.010 <= a <= 0.025
        res.update(d0_mm=1000 * a, slope_m_s=s, v_impact=v)
        bydt = {round(1 / p[0]): p[1] for p in pts}
        conv = abs(bydt[480] - bydt[960]) < 0.003 if 480 in bydt and 960 in bydt else None
        h = c_slope and c_d0 and conv is not False
        res["H_lag"] = "supported" if h else ("against (slope < 0.5 v)" if s < 0.5 * v else
                                              ("against (d0 > 25 mm: bed too weak)" if a > 0.025 else "not supported"))
        notes.append(f"Fit peak = d0 + k dt over voxel 0.02: d0 {1000 * a:.1f} mm, k {s:.2f} m/s vs impact speed "
                     f"{v:.2f} m/s; d(1/480)-d(1/960) {'—' if conv is None else f'{1000 * (bydt[480] - bydt[960]):.1f} mm'} "
                     f"-> H_lag {res['H_lag']}.")
    r1, r2 = load(d, "a4_dt480"), load(d, "a4_dt480_v01")
    if ok(r1) and ok(r2):
        dv = r2["peak_penetration_m"] - r1["peak_penetration_m"]
        res["voxel_effect_mm"] = 1000 * dv
        notes.append(f"Voxel 0.01 vs 0.02 at 1/480: {1000 * dv:+.1f} mm -> "
                     f"{'not set by grid contact resolution' if abs(dv) <= 0.003 else 'grid resolution matters'}.")
    r = load(d, "a4_dt120")
    if ok(r):
        notes.append(f"dt 1/120 reproduction of bearing4 ({1000 * REF_DROP_120:.1f} mm): {mm(r['peak_penetration_m'])} mm "
                     f"({'ok' if abs(r['peak_penetration_m'] - REF_DROP_120) <= 0.001 else 'DIFFERS'}).")
    head = ("| run | dt | voxel | impact m/s | peak mm | final mm | creep mm/s | floor |\n|---|---|---|---|---|---|---|---|")
    return "\n".join([head] + rows), notes, res


def arm5(d):
    rows, notes, res = [], [], {}
    tab = {}
    for v in ("v02", "v01", "v005"):
        for b in ("S2", "pic27"):
            for p in ("", "_proj"):
                k = f"a5_{v}_{b}{p}"
                r = load(d, k)
                if not ok(r):
                    rows.append(f"| {k} | {'not run' if r is None else r.get('error')} ||||||||")
                    continue
                tab[k] = r
                o = r.get("overlap", {"inside": None, "max_depth_m": None})
                r.setdefault("settlement_over_voxel", r["final_penetration_m"] / r["voxel"])
                b5 = (r["final_penetration_m"] <= 0.25 * r["voxel"] and r["final_penetration_m"] <= 0.003
                      and abs(r["creep_m_per_s"]) < 1e-4)
                rows.append(f"| {k} | {r['voxel']} | {b} | {bool(p)} | {mm(r['final_penetration_m'])} | "
                            f"{f1(r['settlement_over_voxel'], 2)} | {f1(1000 * r['creep_m_per_s'], 2)} | "
                            f"{o['inside']} / {mm(o['max_depth_m'])} | {mm(r.get('heave_ring_m'))} | {'PASS' if b5 else 'FAIL'} |")
    g = tab.get("a5_v02_S2")
    res["gate"] = None if g is None else abs(g["final_penetration_m"] - REF_REST_V02) <= 0.001
    if g is not None:
        notes.append(f"Gate (reproduces bearing3 rest {1000 * REF_REST_V02:.1f} mm within 1 mm): "
                     f"**{'PASS' if res['gate'] else 'FAIL'}** ({mm(g['final_penetration_m'])} mm).")
    s2 = [tab[k] for k in ("a5_v02_S2", "a5_v01_S2", "a5_v005_S2") if k in tab]
    if len(s2) >= 2:
        ratios = [r["settlement_over_voxel"] for r in s2]
        mms = [r["final_penetration_m"] for r in s2]
        grid = max(ratios) <= 1.3 * min(ratios) if min(ratios) > 0 else False
        bed = max(mms) <= 1.3 * min(mms) if min(mms) > 0 else False
        overl = all((r.get("overlap", {}).get("max_depth_m") or 0.0) >= 0.5 * r["final_penetration_m"] for r in s2)
        res["C2"] = ("H_grid" if grid and overl else "H_bed" if bed and not overl else
                     "mixed (grid scaling without overlap, or overlap without scaling)" if grid or overl else "neither")
        notes.append("S2 without projection: settlement/voxel " + ", ".join(f"{x:.2f}" for x in ratios) +
                     "; settlement mm " + ", ".join(f"{1000 * x:.1f}" for x in mms) +
                     f"; overlap >= half the settlement: {overl} -> {res['C2']}.")
    for v in ("v02", "v01", "v005"):
        a, b = tab.get(f"a5_{v}_S2"), tab.get(f"a5_{v}_pic27")
        c = tab.get(f"a5_{v}_S2_proj")
        parts = []
        if a and b:
            parts.append(f"pic27 - S2 {1000 * (b['final_penetration_m'] - a['final_penetration_m']):+.1f} mm")
        if a and c:
            parts.append(f"project_outside {1000 * (c['final_penetration_m'] - a['final_penetration_m']):+.1f} mm, "
                         f"overlap {a['overlap']['inside']} -> {c['overlap']['inside']} particles")
        if parts:
            notes.append(f"voxel {v}: " + "; ".join(parts) + ".")
    head = ("| run | voxel | collider basis | project | settlement mm | / voxel | creep mm/s | inside box: n / deepest mm | "
            "ring heave mm | B5 |\n|---|---|---|---|---|---|---|---|---|---|")
    return "\n".join([head] + rows), notes, res


def arm6(d, heap_fit, lag_k):
    r = load(d, "a6_sphere")
    res = {}
    if not ok(r):
        return f"a6_sphere: {'not run' if r is None else r.get('error')}", [], res
    u07 = SM.uehara_solve(0.7, r["diameter"], r["h"])
    dfin = r["final_penetration_m"]
    smoke = (r["stopped"] and not r["reached_floor"] and 0.5 * u07 <= dfin <= 2.0 * u07)
    res["smoke"] = smoke
    notes = [f"Smoke: {'PASS' if smoke else 'FAIL'} (stopped {r['stopped']} at {f1(r['t_stop_s'], 3)} s, floor "
             f"{r['reached_floor']}, final {mm(dfin)} mm vs Uehara(mu 0.7) {mm(u07)} mm; peak {mm(r['peak_penetration_m'])} mm; "
             f"{r['particles']} particles, D = {r['diameter_cells']:.0f} cells)."]
    k = lag_k if lag_k is not None else r["v_impact"]
    corr = dfin - k * r["dt"]
    line = f"Lag-corrected d - k dt = {mm(corr)} mm (k {k:.2f} m/s)"
    if heap_fit is not None:
        uh = SM.uehara_solve(math.tan(math.radians(heap_fit)), r["diameter"], r["h"])
        line += f"; Uehara with mu = tan(heap {heap_fit:.1f} deg) {mm(uh)} mm, with mu 0.7 {mm(u07)} mm"
    notes.append(line + " (reported only; the sweep is batch 02).")
    return "", notes, res


# ------------------------------------------------------------------------------------------------------------ main


def _safe(fn, *args):
    """Run one arm's evaluation; an exception becomes a reported evaluator error, never a crash of the whole summary."""
    try:
        return fn(*args)
    except Exception as e:                                  # noqa: BLE001
        return "", [f"evaluator error in {fn.__name__}: {e!r}"], {"error": repr(e)}


def evaluate(results: pathlib.Path, plan: pathlib.Path | None) -> dict:
    J = jobs(results)
    a1t, a1n, a1r = _safe(arm1, results)
    heap = None
    for k in ("a1_dx20", "a1_dx10"):
        r = load(results, k)
        if ok(r):
            heap = r["final"]["angle_fit_mean_deg"]
            break
    a2t, a2n, a2r = _safe(arm2, results, heap)
    a3t, a3n, a3r = _safe(arm3, results)
    a4t, a4n, a4r = _safe(arm4, results)
    a5t, a5n, a5r = _safe(arm5, results)
    a6t, a6n, a6r = _safe(arm6, results, heap, a4r.get("slope_m_s"))
    summary = {"arm1": a1r, "arm2": a2r, "arm3": a3r, "arm4": a4r, "arm5": a5r, "arm6": a6r, "jobs": J}
    lines = ["# sand_fidelity batch 01: results", "",
             "Plan: research_notes/sand_fidelity/PLAN_2026-10-14.md section 6. No solver changes: existing code paths "
             "and new probes (metalsim/physics/mpm/probes/{sand_repose,tilt_box,bearing_fidelity,sphere_drop}.py). "
             "Machine: the 16-inch M4 Max (correctness runs; wall times below are queue-wrapped, not timing-class).", "",
             "## Predictions stated before the runs", ""]
    for name, text in PREDICTIONS:
        lines += [f"* **{name}.** {text}"]
    lines += ["", "## Verdicts", "",
              f"* Arm 1: gate {verdict(a1r.get('gate'), 'PASS', 'FAIL', 'not run')}; R1 {a1r.get('R1')}; "
              f"R3 (stress) {a1r.get('R3_stress')}.",
              f"* Arm 2: theta_stop {a2r.get('theta_stop')}; theta_fail {a2r.get('theta_fail')}; "
              f"heap vs plane shear: {a2r.get('discrimination', 'undecided')}.",
              f"* Arm 3: R2 {a3r.get('R2', 'not run')}.",
              f"* Arm 4: H_lag {a4r.get('H_lag', 'not run')}.",
              f"* Arm 5: gate {verdict(a5r.get('gate'), 'PASS', 'FAIL', 'not run')}; C2 {a5r.get('C2', 'not run')}.",
              f"* Arm 6: smoke {verdict(a6r.get('smoke'), 'PASS', 'FAIL', 'not run')}.", ""]
    for title, t, n in (("1. Repose profile and stress (R1, R3)", a1t, a1n), ("2. Tilting box (B1)", a2t, a2n),
                        ("3. Discretization variants (R2)", a3t, a3n), ("4. Drop dt sweep (C1)", a4t, a4n),
                        ("5. Resting box (C2)", a5t, a5n), ("6. Sphere drop smoke (B4)", a6t, a6n)):
        lines += [f"## {title}", ""]
        if t:
            lines += [t, ""]
        lines += [f"* {x}" for x in n] + [""]
    if J:
        lines += ["## Jobs (wall time, s, gpu_run + watchdog wrapped; estimate from runs/gpu_history.jsonl)", "",
                  "| job | rc | measured | estimate |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['rc']} | {v['seconds']} | {v['est_seconds']} |" for k, v in J.items()]
        lines.append("")
    text = "\n".join(lines)
    (results / "summary.md").write_text(text)
    json.dump(summary, open(results / "summary.json", "w"), indent=1, default=str)
    if plan is not None:
        note = plan.read_text() if plan.exists() else ""
        body = "\n".join(["## 7. Results: batch sand_fidelity_01", "",
                          "Written by scripts/sand_fidelity/evaluate_batch_01.py; full tables in "
                          "runs/sand_fidelity/batch_01/summary.md.", ""] +
                         [x for x in lines[lines.index("## Verdicts") + 2:] if not x.startswith("|")])
        block = f"{START}\n{body}\n{END}\n"
        if START in note and END in note:
            note = note[:note.index(START)] + block + note[note.index(END) + len(END) + 1:]
        else:
            note = note.rstrip("\n") + "\n\n" + block
        plan.write_text(note)
    return summary


# ------------------------------------------------------------------------------------------------------------ selftest


def _synthetic(d: pathlib.Path):
    """Plausible synthetic results for every job (shapes and fields as the probes write them)."""
    def repose(fit, cone, rdx, fill, **kw):
        bands = [{"band": [0.05, 0.25], "cells": [1, 3], "angle_deg": fit + 4, "sd_deg": 0.3, "sectors": 8},
                 {"band": [0.25, 0.5], "cells": [3, 6], "angle_deg": fit + 1, "sd_deg": 0.3, "sectors": 8},
                 {"band": [0.5, 0.75], "cells": [6, 9], "angle_deg": fit - 1, "sd_deg": 0.3, "sectors": 8},
                 {"band": [0.75, 0.95], "cells": [9, 11], "angle_deg": fit - 4, "sd_deg": 0.3, "sectors": 8}]
        shells = [{"depth_cells": [0, 1], "n": 500, "ratio_median": 0.97, "ratio_p90": 1.0, "frac_at_yield": 0.6,
                   "frac_tension": 0.0, "lode_median": 0.45, "lode_mean": 0.4},
                  {"depth_cells": [1, 2], "n": 500, "ratio_median": 0.96, "ratio_p90": 1.0, "frac_at_yield": 0.55,
                   "frac_tension": 0.0, "lode_median": 0.4, "lode_mean": 0.4},
                  {"depth_cells": [2, 4], "n": 900, "ratio_median": 0.8, "ratio_p90": 0.95, "frac_at_yield": 0.2,
                   "frac_tension": 0.0, "lode_median": 0.2, "lode_mean": 0.2},
                  {"depth_cells": [4, None], "n": 3000, "ratio_median": 0.5, "ratio_p90": 0.8, "frac_at_yield": 0.05,
                   "frac_tension": 0.0, "lode_median": 0.1, "lode_mean": 0.1}]
        r = {"probe": "sand_repose", "friction": 0.68, "atan_mu_deg": 34.22, "voxel": 0.05 / rdx, "radius": 0.05,
             "fill": fill, "R_over_dx": rdx, "ppc": 27, "velocity_basis": "Q1", "strain_basis": "P0",
             "integration_scheme": "pic", "particles": 80040, "at_rest": True, "estimators_spread_deg": cone - fit,
             "final": {"angle_fit_mean_deg": fit, "angle_fit_sd_deg": 0.3, "angle_cone_volume_deg": cone,
                       "angle_height_flank_deg": fit + 0.1, "h_max": 0.03, "r_apex": 0.01, "r_toe": 0.08},
             "height_bands": {"h_max": 0.03, "h_max_cells": 6, "bands": bands, "max_local_angle_deg": fit + 4},
             "flank_stress": {"sign": 1.0, "flank_particles": 4900, "shells": shells, "p_base_over_rho_g_h": 0.9}}
        r.update(kw)
        return r
    w = lambda k, r: json.dump(r, open(d / f"{k}.json", "w"))     # noqa: E731
    w("a1_dx10", repose(26.4, 31.5, 10, 0.05))
    w("a1_dx20", repose(27.8, 33.3, 20, 0.05))
    w("a1_tall", repose(28.3, 31.0, 10, 0.10))
    for k, mu, ts, cells in (("a2_mu040", 0.4, 21.5, 10), ("a2_mu068", 0.68, 33.9, 10), ("a2_mu090", 0.9, 41.2, 10),
                             ("a2_mu068_v01", 0.68, 32.5, 5)):
        w(k, {"probe": "tilt_box", "friction": mu, "atan_mu_deg": math.degrees(math.atan(mu)), "depth_cells": cells,
              "theta_fail_deg": ts + 3, "theta_stop_deg": ts, "failed": True, "at_rest": True})
    for k, kw in (("a3_b2", dict(velocity_basis="B2")), ("a3_q1d", dict(strain_basis="Q1d")),
                  ("a3_p1d", dict(strain_basis="P1d")), ("a3_gimp", dict(integration_scheme="gimp")),
                  ("a3_ppc64", dict(ppc=64))):
        w(k, repose(26.6, 31.6, 10, 0.05, **kw))
    w("a3_p1d", {"error": "synthetic failure"})
    for k, fps, vox, pk in (("a4_dt120", 120, 0.02, 0.0298), ("a4_dt240", 240, 0.02, 0.0230),
                            ("a4_dt480", 480, 0.02, 0.0195), ("a4_dt960", 960, 0.02, 0.0178),
                            ("a4_dt480_v01", 480, 0.01, 0.0170)):
        w(k, {"probe": "bearing_fidelity", "fps": fps, "dt": 1 / fps, "voxel": vox, "impact_speed_m_s": 1.70,
              "peak_penetration_m": pk, "final_penetration_m": pk - 0.001, "creep_m_per_s": 0.0005,
              "reached_floor": False})
    for v, vox in (("v02", 0.02), ("v01", 0.01), ("v005", 0.005)):
        for b in ("S2", "pic27"):
            for p in ("", "_proj"):
                s = 0.51 * vox * (0.8 if b == "pic27" else 1.0) * (0.9 if p else 1.0)
                w(f"a5_{v}_{b}{p}", {"probe": "bearing_fidelity", "voxel": vox, "basis": b, "project": bool(p),
                                     "final_penetration_m": s, "settlement_over_voxel": s / vox,
                                     "creep_m_per_s": 5e-5, "overlap": {"inside": 0 if p else 200,
                                                                         "max_depth_m": 0.0 if p else 0.8 * s,
                                                                         "p90_depth_m": 0.0},
                                     "heave_ring_m": 0.0002})
    json.dump({"probe": "sphere_drop", "h": 0.2, "v_impact": 1.98, "dt": 1 / 480, "diameter": 0.05,
               "diameter_cells": 10, "particles": 307200, "peak_penetration_m": 0.022, "final_penetration_m": 0.021,
               "stopped": True, "t_stop_s": 0.06, "reached_floor": False}, open(d / "a6_sphere.json", "w"))
    (d / "jobs.tsv").write_text("name\trc\tseconds\test_seconds\na1_dx10\t0\t80\t80\na3_p1d\t1\t30\t80\n")


def selftest() -> int:
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
        d = pathlib.Path(t)
        _synthetic(d)
        plan = d / "plan.md"
        plan.write_text("# plan\n\nbody\n")
        s = evaluate(d, plan)
        evaluate(d, plan)                                   # idempotent: the block is replaced, not duplicated
        note = plan.read_text()
        md = (d / "summary.md").read_text()
        checks = {
            "gate1": s["arm1"]["gate"] is True, "gate5": s["arm5"]["gate"] is True,
            "R1": s["arm1"]["R1"] == "supported", "R3": s["arm1"]["R3_stress"] == "consistent",
            "arm2": s["arm2"].get("discrimination") == "heap-specific",
            "R2": s["arm3"]["R2"].startswith("not supported"), "p1d error": "synthetic failure" in md,
            "H_lag": s["arm4"]["H_lag"] == "supported", "C2": s["arm5"]["C2"] == "H_grid",
            "smoke": s["arm6"]["smoke"] is True, "plan once": note.count(START) == 1 and note.count(END) == 1,
            "plan body kept": note.startswith("# plan\n\nbody\n"),
        }
        bad = [k for k, v in checks.items() if not v]
        print("selftest:", "PASS" if not bad else f"FAIL {bad}")
        if bad:
            print(md)
            print(json.dumps(s, indent=1, default=str))
        return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "runs/sand_fidelity/batch_01"))
    ap.add_argument("--plan", default=str(ROOT / "research_notes/sand_fidelity/PLAN_2026-10-14.md"))
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    s = evaluate(pathlib.Path(a.results), None if a.no_plan else pathlib.Path(a.plan))
    print(json.dumps({k: v for k, v in s.items() if k != "jobs"}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
