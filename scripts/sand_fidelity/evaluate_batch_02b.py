#!/usr/bin/env python3
"""Evaluate sand_fidelity batch 02b (research_notes/sand_fidelity/PLAN_2026-10-14.md sections 9 and 11) against the
predictions stated before the runs; write ``runs/sand_fidelity/batch_02b/summary.md`` + ``summary.json`` and the plan
note's section 12 between ``<!-- sand_fidelity_02b results: start -->`` / ``end``.

    python scripts/sand_fidelity/evaluate_batch_02b.py [--results DIR] [--plan NOTE.md] [--no-plan] | --selftest
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
sys.path.insert(0, str(ROOT / "scripts/sand_fidelity"))
sys.path.insert(0, str(ROOT / "metalsim/physics/mpm/probes"))
import evaluate_batch_02a as E  # noqa: E402
import sand_metrics as SM  # noqa: E402

START = "<!-- sand_fidelity_02b results: start -->"
END = "<!-- sand_fidelity_02b results: end -->"
WALLED_02A = 0.0213          # batch 02a s_walled final depth (D 0.05, 10 cells, parity)
UEHARA_D05 = 0.0163          # Uehara at mu 0.7, D 0.05, h 0.2

PREDICTIONS = [
    ("P1 tests", "The variant's own tests (Newton PR #4660's, CPU and Metal) pass; the parity suites "
                 "(tests/test_mpm_main.py, tests/test_mpm.py) pass unchanged: the default path is bitwise as before."),
    ("R2 with #4660 (repose)", "With the PR's P1d/Q1d fix the toe band (5-25 % of h_max) is <= atan(mu) + 1 deg (the "
                               "locking is gone). R2 confirmed if the mid band (25-50 %) stays >= 33 deg and, at R/dx 20, "
                               "the 25-75 % fit is >= 31 deg. If the mid band falls back to ~28 deg, batch 01's P1d/Q1d "
                               "gain was the defect, not the strain basis."),
    ("S-weak (sphere, P1d + #4660)", "Depth falls >= 30 % from the parity walled run (21.3 mm), i.e. <= 14.9 mm, "
                                     "and closer to Uehara's 16.3 mm."),
    ("S-res (sphere, 20 cells, P2)", "Depth / Uehara falls >= 30 % from 10 cells (1.31) in both 20-cell variants."),
    ("S-coupling (kinematic, rerun)", "Quasi-static F(16 mm) against the Uehara-implied mean stopping force of 26 N: "
                                      ">= 13 N means the bed is about as strong as the drop needs (the 1.31x residual is "
                                      "not bed weakness); <= 7 N means the bed is weak."),
    ("Edge leak and #4660", "C1 (plane floor + frictionless side walls, mu 0.4, fixed 20.3 deg) with the variant: "
                            "still non-finite -> #4660 does not cover the concave two-collider junction (one collider "
                            "per node); a candidate upstream issue with C1 as the repro. Finite -> covered."),
    ("Tilt runaway, box floor", "Runaway within atan(mu) +- 1 deg and the deposit within 1 deg below it (mu 0.68, 0.9); "
                                "no non-finite state; particles below the floor < 0.2 % of the material."),
    ("Rest, voxel 0.005, S2 (P2)", "Runs (no int32 scratch failure); settlement 0.45-0.5 voxel (2.2-2.5 mm) at 250 "
                                   "iterations, no overlap; project_outside changes it by < 0.5 mm."),
]


def evaluate(d: pathlib.Path, plan):
    L = E.load
    lines = ["# sand_fidelity batch 02b: results", "", "Plan: research_notes/sand_fidelity/PLAN_2026-10-14.md section 11.",
             "", "## Predictions stated before the runs", ""] + [f"* **{n}.** {t}" for n, t in PREDICTIONS] + ["", "## Results", ""]
    res = {}
    st = (d / "STATUS.txt").read_text() if (d / "STATUS.txt").exists() else ""
    for k in ("P1_TESTS_METAL", "PARITY_MAIN", "PARITY_152"):
        res[k] = "PASS" if f"\n{k} PASS" in "\n" + st else ("FAIL" if f"{k} FAIL" in st else "not run")
    lines.append(f"* Tests: variant on Metal {res['P1_TESTS_METAL']}; parity main {res['PARITY_MAIN']}; "
                 f"parity 1.5.2 {res['PARITY_152']}.")
    for k, label in (("rep_p1d_v", "R/dx 10 P1d"), ("rep_q1d_v", "R/dx 10 Q1d"), ("rep_p1d_dx20_v", "R/dx 20 P1d")):
        r = L(d, k)
        if not E.ok(r):
            lines.append(f"* {label} (#4660): {E.status(r)}.")
            continue
        b = r["height_bands"]["bands"]
        toe, mid, fit, am = b[0]["angle_deg"], b[1]["angle_deg"], r["final"]["angle_fit_mean_deg"], r["atan_mu_deg"]
        res[k] = {"toe": toe, "mid": mid, "fit": fit}
        lock_gone = toe <= am + 1
        lines.append(f"* {label} (#4660): fit {fit:.2f} deg (sd {r['final']['angle_fit_sd_deg']:.2f}), toe {toe:.1f}, mid "
                     f"{mid:.1f}, cone volume {r['final']['angle_cone_volume_deg']:.1f}; toe lock "
                     f"{'gone' if lock_gone else 'still there'}; at rest {r['at_rest']}.")
    r20 = res.get("rep_p1d_dx20_v")
    mids = [res[k]["mid"] for k in ("rep_p1d_v", "rep_q1d_v", "rep_p1d_dx20_v") if k in res]
    if r20 and mids:
        res["R2"] = ("confirmed" if min(mids) >= 33 and r20["fit"] >= 31 else
                     "not confirmed (mid band fell back: batch 01's gain was the defect)" if max(mids) <= 30 else "partial")
        lines.append(f"* R2: {res['R2']}.")
    s = L(d, "sph_p1d_v")
    if E.ok(s):
        dd = s["final_penetration_m"]
        res["S_weak"] = "supported" if dd <= 0.7 * WALLED_02A else "not supported"
        lines.append(f"* Sphere with P1d + #4660: {E.mm(dd)} mm vs parity {1000 * WALLED_02A:.1f} mm, Uehara "
                     f"{1000 * UEHARA_D05:.1f} -> S-weak {res['S_weak']}.")
    rat = {k: L(d, k)["final_penetration_m"] / E.uehara(L(d, k)) for k in ("s20a", "s20b") if E.ok(L(d, k))}
    if rat:
        ok = all(v <= 0.7 * WALLED_02A / UEHARA_D05 for v in rat.values())
        res["S_res"] = "supported" if ok else "not supported"
        lines.append("* 20 cells: " + ", ".join(f"{k} {v:.2f}x Uehara" for k, v in rat.items()) +
                     f" (10 cells 1.31x) -> S-res {res['S_res']}.")
    k = L(d, "s_kin")
    if E.ok(k):
        f16 = k["force_at_16mm_N"]
        res["S_coupling"] = "bed strong enough" if f16 >= 13 else ("bed weak" if f16 <= 7 else "in between")
        lines.append(f"* Kinematic sphere: F(16 mm) {f16:.1f} N, F(30 mm) {k['force_at_30mm_N']:.1f} N (Uehara-implied mean "
                     f"26 N) -> {res['S_coupling']}.")
    c = L(d, "c1_v")
    if c is not None:
        nf = c.get("error") == "nonfinite"
        res["edge_leak_pr4660"] = "not covered (upstream issue candidate)" if nf else "covered"
        lines.append(f"* C1 with #4660: {'NON-FINITE after ' + E.f1(c.get('hold_seconds_before_nan'), 2) + ' s' if nf else 'finite'}"
                     f" -> {res['edge_leak_pr4660']}.")
    for k in ("t1b", "t2b", "tilt_p1d_v"):
        r = L(d, k)
        if r is None:
            continue
        if r.get("error") == "nonfinite":
            lines.append(f"* {k}: NON-FINITE at {E.f1(r.get('theta'))} deg (box floor {r.get('floor') == 'box'}).")
            continue
        am, tf, ts = r["atan_mu_deg"], r.get("theta_fail_deg"), r.get("theta_stop_deg")
        fp = r.get("fast_particles", {})
        below = (fp.get("n_bad") or 0) * (fp.get("frac_below_floor") or 0) / max(r.get("particles", 1), 1)
        okk = tf is not None and abs(tf - am) <= 1 and ts is not None and 0 <= tf - ts <= 1
        lines.append(f"* {k}: creep {E.f1(r.get('theta_creep_deg'))}, runaway {E.f1(tf)}, deposit {E.f1(ts)} vs atan(mu) "
                     f"{am:.1f} -> {'PASS' if okk else 'FAIL'}; below floor {100 * below:.3f} %; solves at cap "
                     f"{E.f1((r.get('solves') or {}).get('frac_at_cap'), 2)}.")
    for k in ("rest_v005_S2", "rest_v005_S2_proj"):
        r = L(d, k)
        if not E.ok(r):
            lines.append(f"* {k}: {E.status(r)}.")
            continue
        lines.append(f"* {k}: settlement {E.mm(r['final_penetration_m'])} mm ({r['settlement_over_voxel']:.2f} voxel), "
                     f"overlap {r['overlap']['inside']}, creep {1000 * r['creep_m_per_s']:.2f} mm/s.")
    text = "\n".join(lines) + "\n"
    (d / "summary.md").write_text(text)
    json.dump(res, open(d / "summary.json", "w"), indent=1, default=str)
    if plan is not None:
        note = plan.read_text() if plan.exists() else ""
        body = "\n".join(["## 12. Results: batch sand_fidelity_02b", "",
                          "Written by scripts/sand_fidelity/evaluate_batch_02b.py.", ""] + lines[lines.index("## Results") + 2:])
        block = f"{START}\n{body}\n{END}\n"
        if START in note and END in note:
            note = note[:note.index(START)] + block + note[note.index(END) + len(END) + 1:]
        else:
            note = note.rstrip("\n") + "\n\n" + block
        plan.write_text(note)
    return res


def selftest():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
        d = pathlib.Path(t)
        w = lambda k, r: json.dump(r, open(d / f"{k}.json", "w"))   # noqa: E731
        (d / "STATUS.txt").write_text("P1_TESTS_METAL PASS\nPARITY_MAIN PASS\nPARITY_152 PASS\n")
        bands = lambda toe, mid: {"bands": [{"angle_deg": toe}, {"angle_deg": mid}, {"angle_deg": 30}, {"angle_deg": 22}]}  # noqa: E731
        rep = lambda fit, toe, mid: {"atan_mu_deg": 34.2, "at_rest": True, "height_bands": bands(toe, mid),  # noqa: E731
                                     "final": {"angle_fit_mean_deg": fit, "angle_fit_sd_deg": 0.2, "angle_cone_volume_deg": 33}}
        w("rep_p1d_v", rep(31.5, 34.5, 34.0))
        w("rep_q1d_v", rep(30.8, 35.0, 33.5))
        w("rep_p1d_dx20_v", rep(32.0, 34.8, 34.2))
        sph = {"final_penetration_m": 0.013, "diameter": 0.05, "h": 0.2, "ball_density": 2500.0}
        w("sph_p1d_v", sph)
        w("s20a", dict(sph, final_penetration_m=0.0135))
        w("s_kin", {"force_at_16mm_N": 20.0, "force_at_30mm_N": 35.0})
        w("c1_v", {"error": "nonfinite", "hold_seconds_before_nan": 4.0})
        w("t1b", {"atan_mu_deg": 34.2, "theta_fail_deg": 34.6, "theta_stop_deg": 34.0, "particles": 129600,
                  "fast_particles": {"n_bad": 10, "frac_below_floor": 0.5}})
        w("rest_v005_S2", {"final_penetration_m": 0.0024, "settlement_over_voxel": 0.48, "overlap": {"inside": 0},
                           "creep_m_per_s": 0.0004})
        plan = d / "plan.md"
        plan.write_text("# p\n")
        r = evaluate(d, plan)
        evaluate(d, plan)
        checks = {"R2": r.get("R2") == "confirmed", "S_weak": r.get("S_weak") == "supported",
                  "S_res": r.get("S_res") == "supported", "kin": r.get("S_coupling") == "bed strong enough",
                  "edge": r.get("edge_leak_pr4660", "").startswith("not covered"),
                  "tests": r.get("PARITY_MAIN") == "PASS", "plan": plan.read_text().count(START) == 1}
        bad = [k for k, v in checks.items() if not v]
        print("selftest:", "PASS" if not bad else f"FAIL {bad}")
        if bad:
            print((d / "summary.md").read_text())
        return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "runs/sand_fidelity/batch_02b"))
    ap.add_argument("--plan", default=str(ROOT / "research_notes/sand_fidelity/PLAN_2026-10-14.md"))
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print(json.dumps(evaluate(pathlib.Path(a.results), None if a.no_plan else pathlib.Path(a.plan)), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
