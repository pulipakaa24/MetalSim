"""Static hold band of candidate STS3215 models against the real followers' sag by approach direction (real_dir.json).

A P servo stopping after a move in the direction gravity pulls ("with") settles at the band's low edge, after a move
against gravity ("against") at its high edge (it stops at the first pose where the friction budget holds it). Candidate
models (all at LeRobot's P = 16):
  7v4_real      kp 5.262 (datasheet stall 1.912 N.m / 0.97 x P x 0.166 x 1.0049), Coulomb 0.0515 (BAM m1)
  bam_m1        BAM m1: kp from its kt / R, Coulomb 0.0515
  bam_m3..m6    BAM m3..m6: kp from each file's kt / R / error_gain_ratio, its full static friction budget
  7v4_real+m6f  7v4_real's kp with BAM m6's friction budget (kp from the datasheet, friction from BAM)
  python3 static_band.py   (writes static_band.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bam_m6 as B  # noqa: E402

real = json.load(open(os.path.join(HERE, "real_dir.json")))
P = {k: B.load(k) for k in ("m1", "m3", "m4", "m5", "m6")}
M = {k: B.motor(P[k]) for k in P}
cands = {
    "7v4_real": (5.262, lambda tm, te: 0.0515 + 0 * tm),
    "bam_m1": (M["m1"]["kp"], lambda tm, te: B.budget(P["m1"], tm, te, 0.0)),
    "bam_m3": (M["m3"]["kp"], lambda tm, te: B.budget(P["m3"], tm, te, 0.0)),
    "bam_m4": (M["m4"]["kp"], lambda tm, te: B.budget(P["m4"], tm, te, 0.0)),
    "bam_m5": (M["m5"]["kp"], lambda tm, te: B.budget(P["m5"], tm, te, 0.0)),
    "bam_m6": (M["m6"]["kp"], lambda tm, te: B.budget(P["m6"], tm, te, 0.0)),
    "7v4_real+m6f": (5.262, lambda tm, te: B.budget(P["m6"], tm, te, 0.0)),
}
print("BAM motor constants at P 16: " + "; ".join(f"{k} kp {v['kp']:.2f} stall {v['stall']:.2f} N.m no-load "
                                                  f"{v['noload']:.2f} rad/s" for k, v in M.items()))
out = dict(kp={k: round(v[0], 3) for k, v in cands.items()}, rows=[])
err = {k: [] for k in cands}
for ds, joints in real.items():
    for j, r in joints.items():
        for b in r["bins"]:
            g = b["g"]
            row = dict(ds=ds, joint=j, g=g, real_with=b["with_"][0], n_with=b["with_"][1],
                       real_against=b["against"][0], n_against=b["against"][1])
            for k, (kp, fr) in cands.items():
                lo, hi = B.static_band(kp, g, fr)
                row[k] = (round(lo * 1e3, 1), round(hi * 1e3, 1))
                for side, val, n in ((0, row["real_with"], row["n_with"]), (1, row["real_against"], row["n_against"])):
                    if val is not None and n >= 30 and g >= 0.05:
                        err[k].append((j, side, row[k][side] - val))
            out["rows"].append(row)
print(f"{'ds':9s} {'joint':13s} {'|g|':>5s} {'real with / against':>21s} | " + " | ".join(f"{k:>13s}" for k in cands))
for r in out["rows"]:
    rw = f"{r['real_with']} / {r['real_against']}"
    print(f"{r['ds']:9s} {r['joint']:13s} {r['g']:5.2f} {rw:>21s} | " +
          " | ".join(f"{r[k][0]:5.1f} /{r[k][1]:5.1f}" for k in cands))
print("\nmedian |model edge - real| (mrad), bins with |g| >= 0.05 and >= 30 samples; per joint (with-edge, against-edge):")
summ = {}
for k in cands:
    e = err[k]
    cells = {}
    for j in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        w = [abs(x) for jj, s, x in e if jj == j and s == 0]
        a = [abs(x) for jj, s, x in e if jj == j and s == 1]
        cells[j] = (round(float(np.median(w)), 1) if w else None, round(float(np.median(a)), 1) if a else None)
    allv = [abs(x) for _, _, x in e]
    summ[k] = dict(per_joint=cells, all=round(float(np.median(allv)), 1), n=len(allv))
    print(f"  {k:13s} all {summ[k]['all']:5.1f} (n {len(allv)})  " +
          "  ".join(f"{j} {c[0]} / {c[1]}" for j, c in cells.items()))
out["summary"] = summ
out["motor"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in M.items()}
json.dump(out, open(os.path.join(HERE, "static_band.json"), "w"), indent=1)
