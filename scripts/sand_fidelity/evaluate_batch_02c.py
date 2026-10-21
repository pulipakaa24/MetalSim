#!/usr/bin/env python3
"""Evaluate sand_fidelity batch 02c (research_notes/sand_fidelity/PLAN_2026-10-14.md section 13): the candidate
physically-right preset (P1d strain + Newton #4660 variant + 1000 iterations) against PHYSICAL references (atan(mu),
Uehara's law with mu from the same configuration's own heap, Uehara's mass scaling, B5), not as cuts from parity.
Writes runs/sand_fidelity/batch_02c/summary.{md,json} and the plan note's section 14
(``<!-- sand_fidelity_02c results: start/end -->``).

    python scripts/sand_fidelity/evaluate_batch_02c.py [--results DIR] [--plan NOTE.md] [--no-plan] | --selftest
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

START = "<!-- sand_fidelity_02c results: start -->"
END = "<!-- sand_fidelity_02c results: end -->"
ATAN_068 = math.degrees(math.atan(0.68))
REF = {"rep10_p1d_250": 29.89, "rep20_p1d_250": 30.73, "sph_glass_p1d_250": 0.0163, "drop_v01_parity_d0": 0.0225}

PREDICTIONS = [
    ("Heap (preset)", f"Reference atan(mu) = {ATAN_068:.1f} deg. Under-convergence (H-iter): R/dx 10 rises >= 1.5 deg over "
                      "the 250-iteration P1d value (29.9 -> >= 31.4) and R/dx 20 ends within 2 deg of atan(mu) (>= 32.2). "
                      "Against: both within +-0.5 deg of their 250-iteration values (30.7 at R/dx 20)."),
    ("Tilt (preset, box floor)", "Runaway within atan(mu) +- 1 deg and the deposit angle within 1 deg below the runaway."),
    ("Sphere glass (preset)", "Final depth within +-15 % (Uehara's scatter) of Uehara's law with mu = tan(the preset heap "
                              "angle at the finest resolution run), and of mu = 0.7; reported both ways."),
    ("Sphere mass scaling", "Steel / glass depth ratio 1.77 +- 0.2 (Uehara: d ~ rho_ball^1/2) at 250 iterations with P1d "
                            "and with the preset."),
    ("Sphere resolution (P1d, 20 cells)", "Within +-15 % of the 10-cell P1d depth (16.3 mm): the 10-cell hit is "
                                         "resolution-converged. Deepening like parity P0 (1.31 -> 1.46x Uehara) means "
                                         "the 10-cell match was a resolution coincidence."),
    ("Box drop (preset, dt 1/960, voxel 0.01)", "Lag-free penetration (impact-phase peak minus v dt = 1.8 mm) in the "
                                                "10-25 mm band and below parity's 22.5 mm at the same voxel."),
    ("Rest", "Parity at voxel 0.005 with 8 particles per cell: 0.45-0.5 voxel (grid contact resolution); the voxel-0.01 "
             "bridge at 8 ppc reproduces 27-ppc's 4.5 mm within 0.7 mm; the preset at voxel 0.005 passes B5 "
             "(<= 0.25 voxel, <= 3 mm, creep < 0.1 mm/s)."),
    ("Edge leak", "C1 with the collider-only variant (pr4660c) stays finite with no particle below the floor: the "
                  "collider distance-sign fix is what stops the leak."),
    ("Deposit metric", "Parity t1b rerun (fixed deposit fit): deposit within 1 deg below its runaway."),
]


def evaluate(d, plan):
    L, ok = E.load, E.ok
    out, res = [], {}
    heap = {}
    for k, ref in (("rep10_pre", REF["rep10_p1d_250"]), ("rep20_pre", REF["rep20_p1d_250"])):
        r = L(d, k)
        if not ok(r):
            out.append(f"* {k}: {E.status(r)}.")
            continue
        fit = r["final"]["angle_fit_mean_deg"]
        b = r["height_bands"]["bands"]
        heap[k] = fit
        out.append(f"* {k}: fit {fit:.2f} deg ({fit - ref:+.2f} vs 250 iterations; atan(mu) - fit = {ATAN_068 - fit:.2f}); "
                   f"toe / mid {b[0]['angle_deg']:.1f} / {b[1]['angle_deg']:.1f}; cone volume "
                   f"{r['final']['angle_cone_volume_deg']:.1f}; at rest {r['at_rest']}.")
    if "rep10_pre" in heap:
        d10 = heap["rep10_pre"] - REF["rep10_p1d_250"]
        res["H_iter"] = "supported" if d10 >= 1.5 else ("against" if abs(d10) <= 0.5 else "partial")
        if "rep20_pre" in heap:
            res["preset_heap_within_2deg"] = ATAN_068 - heap["rep20_pre"] <= 2.0
        out.append(f"* H-iter: {res['H_iter']}; preset heap within 2 deg of atan(mu): {res.get('preset_heap_within_2deg')}.")
    mu_ref = math.tan(math.radians(heap.get("rep20_pre", heap.get("rep10_pre", REF["rep20_p1d_250"]))))
    for k in ("tilt_pre", "t1b_fix"):
        r = L(d, k)
        if r is None:
            continue
        if r.get("error") == "nonfinite":
            out.append(f"* {k}: NON-FINITE at {E.f1(r.get('theta'))} deg.")
            continue
        am, tf, ts = r["atan_mu_deg"], r.get("theta_fail_deg"), r.get("theta_stop_deg")
        good = tf is not None and abs(tf - am) <= 1 and ts is not None and 0 <= tf - ts <= 1
        res[k] = good
        out.append(f"* {k}: creep {E.f1(r.get('theta_creep_deg'))}, runaway {E.f1(tf)} (hold {E.f1(r.get('theta_hold_deg'))}), "
                   f"deposit {E.f1(ts)} vs atan(mu) {am:.1f} -> {'PASS' if good else 'FAIL'}.")
    g, s, s250, g250 = L(d, "sph_glass_pre"), L(d, "sph_steel_pre"), L(d, "sph_steel_p1d"), REF["sph_glass_p1d_250"]
    if ok(g):
        dd = g["final_penetration_m"]
        u_ref = SM.uehara_solve(mu_ref, g["diameter"], g["h"])
        u07 = SM.uehara_solve(0.7, g["diameter"], g["h"])
        res["sphere_vs_uehara_ref"] = dd / u_ref
        out.append(f"* Sphere glass (preset): {E.mm(dd)} mm; Uehara with mu = tan(heap) = {mu_ref:.3f}: {E.mm(u_ref)} mm "
                   f"({dd / u_ref:.2f}x), with mu 0.7: {E.mm(u07)} mm ({dd / u07:.2f}x) -> "
                   f"{'within' if abs(dd / u_ref - 1) <= 0.15 else 'outside'} +-15 % of the self-consistent reference.")
    for label, a, b in (("P1d, 250 iterations", s250, g250), ("preset", s, g)):
        if ok(a) and (isinstance(b, float) or ok(b)):
            gd = b if isinstance(b, float) else b["final_penetration_m"]
            rr = a["final_penetration_m"] / gd
            res[f"steel_ratio_{label.split(',')[0]}"] = rr
            out.append(f"* Steel / glass ({label}): {rr:.2f} (Uehara 1.77 +- 0.2) -> {'PASS' if abs(rr - 1.77) <= 0.2 else 'FAIL'}.")
    r = L(d, "sph20_p1d")
    if ok(r):
        rr = r["final_penetration_m"] / g250
        res["sphere_p1d_20cells_ratio"] = rr
        out.append(f"* Sphere P1d at 20 cells: {E.mm(r['final_penetration_m'])} mm, {rr:.2f}x the 10-cell P1d depth -> "
                   f"{'converged' if abs(rr - 1) <= 0.15 else 'not converged (10-cell hit partly resolution)'}.")
    r = L(d, "drop_pre")
    if ok(r):
        ip = SM.impact_phase(r)
        lagfree = ip["peak_m"] - ip["v_impact"] * r["dt"]
        good = 0.010 <= lagfree <= 0.025 and lagfree < REF["drop_v01_parity_d0"]
        out.append(f"* Box drop (preset): impact-phase peak {E.mm(ip['peak_m'])} mm, lag-free {E.mm(lagfree)} mm (parity d0 "
                   f"at voxel 0.01: 22.5) -> {'PASS' if good else 'FAIL'}; late sink "
                   f"{E.mm(r['final_penetration_m'] - ip['peak_m'])} mm, max tilt {E.f1(r.get('max_tilt_deg'))} deg.")
    for k, check in (("rest_v005_ppc2", lambda r: 0.45 <= r["settlement_over_voxel"] <= 0.5),
                     ("rest_v01_ppc2", lambda r: abs(r["final_penetration_m"] - 0.0045) <= 0.0007),
                     ("rest_v005_pre", lambda r: r["final_penetration_m"] <= min(0.25 * r["voxel"], 0.003)
                      and abs(r["creep_m_per_s"]) < 1e-4)):
        r = L(d, k)
        if not ok(r):
            out.append(f"* {k}: {E.status(r)}.")
            continue
        out.append(f"* {k}: settlement {E.mm(r['final_penetration_m'])} mm ({r['settlement_over_voxel']:.2f} voxel), creep "
                   f"{1000 * r['creep_m_per_s']:.2f} mm/s, overlap {r['overlap']['inside']}, {r.get('particles')} particles "
                   f"-> {'PASS' if check(r) else 'FAIL'}.")
    r = L(d, "c1_c")
    if r is not None:
        nf = r.get("error") == "nonfinite"
        below = (r.get("fast_particles") or {}).get("frac_below_floor") or 0
        res["collider_sign_fix_is_cause"] = (not nf) and below == 0
        out.append(f"* C1 with the collider-only variant: {'NON-FINITE' if nf else 'finite'}; "
                   f"-> the collider sign fix {'is' if res['collider_sign_fix_is_cause'] else 'is not (alone)'} the cause.")
    lines = ["# sand_fidelity batch 02c: results", "", "## Predictions stated before the runs", ""] + \
            [f"* **{n}.** {t}" for n, t in PREDICTIONS] + ["", "## Results", ""] + out
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(res, open(d / "summary.json", "w"), indent=1, default=str)
    if plan is not None:
        note = plan.read_text() if plan.exists() else ""
        block = f"{START}\n" + "\n".join(["## 14. Results: batch sand_fidelity_02c", ""] + out) + f"\n{END}\n"
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
        bands = {"bands": [{"angle_deg": 34.5}, {"angle_deg": 33.5}, {"angle_deg": 31}, {"angle_deg": 23}]}
        w("rep10_pre", {"final": {"angle_fit_mean_deg": 31.8, "angle_cone_volume_deg": 32}, "height_bands": bands, "at_rest": True})
        w("rep20_pre", {"final": {"angle_fit_mean_deg": 32.6, "angle_cone_volume_deg": 33}, "height_bands": bands, "at_rest": True})
        w("tilt_pre", {"atan_mu_deg": ATAN_068, "theta_fail_deg": 34.6, "theta_stop_deg": 34.0, "theta_creep_deg": 34.0})
        sph = {"diameter": 0.05, "h": 0.2, "ball_density": 2500.0}
        w("sph_glass_pre", dict(sph, final_penetration_m=0.0175))
        w("sph_steel_pre", dict(sph, ball_density=7800.0, final_penetration_m=0.031))
        w("sph_steel_p1d", dict(sph, ball_density=7800.0, final_penetration_m=0.029))
        w("sph20_p1d", dict(sph, final_penetration_m=0.0170))
        w("c1_c", {"fast_particles": {"n_bad": 0}})
        plan = d / "p.md"
        plan.write_text("# p\n")
        r = evaluate(d, plan)
        evaluate(d, plan)
        checks = {"H_iter": r.get("H_iter") == "supported", "heap2": r.get("preset_heap_within_2deg") is True,
                  "tilt": r.get("tilt_pre") is True, "steel": abs(r.get("steel_ratio_preset", 0) - 1.77) < 0.2,
                  "res": abs(r.get("sphere_p1d_20cells_ratio", 0) - 1) <= 0.15,
                  "c1": r.get("collider_sign_fix_is_cause") is True, "plan": plan.read_text().count(START) == 1}
        bad = [k for k, v in checks.items() if not v]
        print("selftest:", "PASS" if not bad else f"FAIL {bad}")
        if bad:
            print((d / "summary.md").read_text())
        return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "runs/sand_fidelity/batch_02c"))
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
