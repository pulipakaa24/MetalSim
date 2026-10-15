#!/usr/bin/env python3
"""Evaluate sand_fidelity batch 02d (research_notes/sand_fidelity/PLAN_2026-10-14.md section 15.3): particles-per-cell
convergence and the heap / layer angles against mu. Reports the FACTS for the decision that follows (calibrate mu per
material on the heap and document the layer offset, or a Lode-dependent yield surface as a code project); it does not
choose. Writes runs/sand_fidelity/batch_02d/summary.{md,json} and the plan note's section 16
(``<!-- sand_fidelity_02d results: start/end -->``).

    python scripts/sand_fidelity/evaluate_batch_02d.py [--results DIR] [--plan NOTE.md] [--no-plan] | --selftest
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
sys.path.insert(0, str(ROOT / "scripts/sand_fidelity"))
import evaluate_batch_02a as E  # noqa: E402

START = "<!-- sand_fidelity_02d results: start -->"
END = "<!-- sand_fidelity_02d results: end -->"
# earlier batches, same configurations (raw JSONs under runs/sand_fidelity/)
KNOWN = {
    "heap_mu068_r10": 29.89,      # 02b rep_p1d_v: P1d + #4660, R/dx 10, 250 iterations
    "heap_mu068_r20": 30.73,      # 02b rep_p1d_dx20_v: R/dx 20 (the R/dx 10 -> 20 offset is +0.84 deg at mu 0.68)
    "layer_mu068": 36.68,         # 02c tilt_pre: preset deposit, box floor
    "sphere_p1d_8ppc": 0.0163,    # 02b sph_p1d_v: P1d + #4660, 250 iterations, 8 ppc, 10 cells
    "rest_v01_27ppc": 0.0045,     # 02a a5_v01_S2 / r250: parity, voxel 0.01, 27 ppc
    "rest_v01_8ppc": 0.00637,     # 02c rest_v01_ppc2: parity, voxel 0.01, 8 ppc
}
TARGETS = (32.0, 34.0)            # dry-sand repose angles a material preset might target (sand 30-35 deg)

PREDICTIONS = [
    ("ppc, sphere", "P1d + #4660 sphere at 27 ppc (10 cells) within +-10 % of its 8-ppc depth (16.3 mm, 250 iterations)."),
    ("ppc, rest", "Parity resting box at voxel 0.01: settlement converges between 27 ppc (4.5 mm) and 64 ppc within "
                  "0.1 voxel (1 mm); 8 ppc (6.4 mm) is then the unconverged one."),
    ("Preset rest", "The preset (P1d + #4660 + 1000 iterations) resting box at voxel 0.01 and 27 ppc tilts < 0.5 deg "
                    "(the 02c preset box at 8 ppc tilted 3 deg); lowest-corner settlement and creep reported against B5."),
    ("Heap / layer vs mu", "No prediction of the outcome: the gap (layer deposit - heap angle) at mu 0.68 / 0.90 and the "
                           "heap at 0.80 / 0.90 are reported with the calibration facts. Reading agreed beforehand: a gap "
                           ">= 4 deg at every mu means Drucker-Prager's geometry dependence is the limit; < 2 deg at some "
                           "mu means that mu can define the preset."),
]


def evaluate(d, plan):
    L, ok = E.load, E.ok
    out, res = [], {}
    # ---- ppc
    s = L(d, "sph27_p1d")
    if ok(s):
        rr = s["final_penetration_m"] / KNOWN["sphere_p1d_8ppc"]
        res["sphere_27_over_8ppc"] = rr
        out.append(f"* Sphere P1d + #4660 at 27 ppc ({s['particles']} particles): {E.mm(s['final_penetration_m'])} mm, "
                   f"{rr:.2f}x the 8-ppc depth -> {'ppc-converged' if abs(rr - 1) <= 0.10 else 'NOT ppc-converged'}.")
    else:
        out.append(f"* Sphere at 27 ppc: {E.status(s)}.")
    r = L(d, "rest_v01_ppc64")
    if ok(r):
        s64 = r["final_penetration_m"]
        conv = abs(s64 - KNOWN["rest_v01_27ppc"]) <= 0.1 * r["voxel"]
        res["rest_ppc_converged_27_64"] = conv
        out.append(f"* Parity rest, voxel 0.01: 8 / 27 / 64 ppc = {1000 * KNOWN['rest_v01_8ppc']:.1f} / "
                   f"{1000 * KNOWN['rest_v01_27ppc']:.1f} / {E.mm(s64)} mm (lowest corner {E.mm(r.get('final_penetration_lowest_m'))}"
                   f" mm, tilt {E.f1(r.get('final_tilt_deg'), 2)} deg) -> 27-64 {'converged' if conv else 'NOT converged'} "
                   f"within 0.1 voxel.")
    else:
        out.append(f"* Parity rest at 64 ppc: {E.status(r)}.")
    r = L(d, "rest_v01_pre27")
    if ok(r):
        lo = r.get("final_penetration_lowest_m")
        tilt = r.get("max_tilt_deg")
        b5 = (lo is not None and lo <= min(0.25 * r["voxel"], 0.003) and abs(r.get("creep_lowest_m_per_s", 1)) < 1e-4)
        res["preset_rest_tilt_ok"] = tilt is not None and tilt < 0.5
        res["preset_rest_b5"] = b5
        out.append(f"* Preset rest, voxel 0.01, 27 ppc: lowest-corner settlement {E.mm(lo)} mm "
                   f"({(lo or 0) / r['voxel']:.2f} voxel), centre {E.mm(r['final_penetration_m'])} mm, max tilt "
                   f"{E.f1(tilt, 2)} deg, lowest-corner creep {1000 * r.get('creep_lowest_m_per_s', float('nan')):.2f} mm/s "
                   f"-> tilt {'< 0.5 deg' if res['preset_rest_tilt_ok'] else '>= 0.5 deg (still rocking)'}; "
                   f"B5 {'PASS' if b5 else 'FAIL'}.")
    else:
        out.append(f"* Preset rest at 27 ppc: {E.status(r)}.")
    # ---- heap / layer vs mu
    heap = {0.68: KNOWN["heap_mu068_r10"]}
    for mu, k in ((0.80, "heap_mu080"), (0.90, "heap_mu090")):
        r = L(d, k)
        if ok(r):
            heap[mu] = r["final"]["angle_fit_mean_deg"]
        else:
            out.append(f"* {k}: {E.status(r)}.")
    layer = {0.68: KNOWN["layer_mu068"]}
    r = L(d, "tilt_pre_mu090")
    if ok(r) and r.get("theta_stop_deg") is not None and math.isfinite(r["theta_stop_deg"]):
        layer[0.90] = r["theta_stop_deg"]
    elif r is not None:
        out.append(f"* tilt_pre_mu090: {E.status(r) if r.get('error') else 'no deposit angle'}"
                   f"{' (' + str(r.get('n_outside')) + ' particles outside the box)' if r.get('error') == 'ejected' else ''}.")
    rows = ["| mu | atan(mu) | heap R/dx 10 (R/dx 20 est.) | layer deposit | layer - heap | heap - atan | layer - atan |",
            "|---|---|---|---|---|---|---|"]
    gaps = {}
    for mu in sorted(set(heap) | set(layer)):
        am = math.degrees(math.atan(mu))
        h, la = heap.get(mu), layer.get(mu)
        h20 = h + (KNOWN["heap_mu068_r20"] - KNOWN["heap_mu068_r10"]) if h is not None else None
        g = la - h20 if (la is not None and h20 is not None) else None
        if g is not None:
            gaps[mu] = g
        rows.append(f"| {mu:.2f} | {am:.1f} | {E.f1(h)} ({E.f1(h20)}) | {E.f1(la)} | {E.f1(g)} | "
                    f"{E.f1(h20 - am if h20 is not None else None)} | {E.f1(la - am if la is not None else None)} |")
    res["heap"], res["layer"], res["gap"] = heap, layer, gaps
    out += [""] + rows + [""]
    if gaps:
        res["gap_min"], res["gap_max"] = min(gaps.values()), max(gaps.values())
        reading = (">= 4 deg at every mu measured: the agreed reading is a Drucker-Prager geometry limit" if min(gaps.values()) >= 4
                   else "< 2 deg at some mu: that mu could define the preset" if min(gaps.values()) < 2 else "between 2 and 4 deg")
        out.append(f"* Gap (layer - heap at R/dx 20 est.): {', '.join(f'{g:.1f} deg at mu {mu:.2f}' for mu, g in sorted(gaps.items()))}"
                   f" -> {reading}.")
    if len(heap) >= 2:
        mus = np.array(sorted(heap))
        hh = np.array([heap[m] for m in mus]) + (KNOWN["heap_mu068_r20"] - KNOWN["heap_mu068_r10"])
        lm = sorted(layer)
        for tgt in TARGETS:
            mu_cal = float(np.interp(tgt, hh, mus, left=float("nan"), right=float("nan"))) if hh.min() <= tgt <= hh.max() else None
            if mu_cal is None and len(mus) >= 2:            # linear extrapolation, flagged
                b, a = np.polyfit(hh, mus, 1)
                mu_cal = float(a + b * tgt)
                note = " (extrapolated)"
            else:
                note = ""
            lay = None
            if len(lm) >= 2:
                b2, a2 = np.polyfit(lm, [layer[m] for m in lm], 1)
                lay = float(a2 + b2 * mu_cal)
            res[f"mu_for_heap_{tgt:.0f}"] = mu_cal
            out.append(f"* Option A facts, heap target {tgt:.0f} deg: mu = {mu_cal:.3f}{note} (atan {math.degrees(math.atan(mu_cal)):.1f} "
                       f"deg); a plane-shear layer at that mu deposits at ~{E.f1(lay)} deg "
                       f"({E.f1(lay - tgt if lay is not None else None)} deg above the heap); sphere depth scales ~1/mu "
                       f"(Uehara) from 15.1 mm at mu 0.7.")
    out.append("* Option B facts: a Lode-dependent yield surface (Mohr-Coulomb-like / Matsuoka-Nakai) replaces the "
               "second-order-cone projection in the DB16 local solver (plan section 2, R3): a new projection kernel in "
               "rheology_solver_kernels, flag-gated like the #4660 variant, with the heap / layer / sphere / drop arms as "
               "its acceptance tests.")
    lines = ["# sand_fidelity batch 02d: results", "", "## Predictions stated before the runs", ""] + \
            [f"* **{n}.** {t}" for n, t in PREDICTIONS] + ["", "## Results (facts for the decision; no choice made here)", ""] + out
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(res, open(d / "summary.json", "w"), indent=1, default=str)
    if plan is not None:
        note = plan.read_text() if plan.exists() else ""
        block = f"{START}\n" + "\n".join(["## 16. Results: batch sand_fidelity_02d", ""] + out) + f"\n{END}\n"
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
        w("sph27_p1d", {"final_penetration_m": 0.0170, "particles": 1036800})
        w("rest_v01_ppc64", {"final_penetration_m": 0.0042, "voxel": 0.01, "final_penetration_lowest_m": 0.0043,
                             "final_tilt_deg": 0.05})
        w("rest_v01_pre27", {"final_penetration_m": 0.0011, "final_penetration_lowest_m": 0.0013, "voxel": 0.01,
                             "max_tilt_deg": 0.2, "creep_lowest_m_per_s": 0.00005})
        w("heap_mu080", {"final": {"angle_fit_mean_deg": 33.0}})
        w("heap_mu090", {"final": {"angle_fit_mean_deg": 35.6}})
        w("tilt_pre_mu090", {"theta_stop_deg": 44.6})
        plan = d / "p.md"
        plan.write_text("# p\n")
        r = evaluate(d, plan)
        evaluate(d, plan)
        checks = {"ppc_sphere": abs(r["sphere_27_over_8ppc"] - 1.043) < 0.01, "rest_conv": r["rest_ppc_converged_27_64"],
                  "tilt": r["preset_rest_tilt_ok"], "b5": r["preset_rest_b5"], "gap": 4 <= r["gap_min"] <= 6,
                  "cal": r.get("mu_for_heap_34") is not None and 0.8 < r["mu_for_heap_34"] < 0.9,
                  "plan": plan.read_text().count(START) == 1}
        bad = [k for k, v in checks.items() if not v]
        print("selftest:", "PASS" if not bad else f"FAIL {bad}")
        if bad:
            print((d / "summary.md").read_text(), json.dumps(r, default=str))
        return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "runs/sand_fidelity/batch_02d"))
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
