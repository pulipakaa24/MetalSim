"""Bitwise comparison of two sets of MPM recordings (record_mpm.py output with the same run ids).

    python -m metalsim.physics.mpm.bitwise_report DIR_A DIR_B [--md out.md]

For every run id present in both: every array of traj.npz (positions, velocities, body poses, per-frame extras) and,
when both runs logged solves, solves.npz (iterations and residuals per solve) compared byte for byte; the first
frame where positions differ otherwise.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def compare_run(a: str, b: str) -> dict:
    ta, tb = dict(np.load(os.path.join(a, "traj.npz"))), dict(np.load(os.path.join(b, "traj.npz")))
    row = {"arrays": sorted(ta), "equal": True, "differs": []}
    for k in sorted(set(ta) | set(tb)):
        if k not in ta or k not in tb or ta[k].shape != tb[k].shape or ta[k].tobytes() != tb[k].tobytes():
            row["equal"] = False
            row["differs"].append(k)
    if "pos" in row["differs"] and ta["pos"].shape == tb["pos"].shape:
        d = np.abs(ta["pos"].astype(np.float64) - tb["pos"].astype(np.float64)).max(axis=(1, 2))
        row["first_diff_frame"] = int(ta["frame"][int(np.argmax(d > 0))])
    sa, sb = os.path.join(a, "solves.npz"), os.path.join(b, "solves.npz")
    if os.path.exists(sa) and os.path.exists(sb):
        la, lb = dict(np.load(sa)), dict(np.load(sb))
        if la.get("iterations", np.zeros(0)).size and lb.get("iterations", np.zeros(0)).size:
            row["solves_equal"] = all(la[k].tobytes() == lb[k].tobytes() for k in ("iterations", "res_l2", "res_linf"))
            row["equal"] &= row["solves_equal"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--md")
    args = ap.parse_args()
    ids = sorted(d for d in os.listdir(args.a) if os.path.exists(os.path.join(args.a, d, "traj.npz"))
                 and os.path.exists(os.path.join(args.b, d, "traj.npz")))
    lines = [f"bitwise: {args.a} vs {args.b}", "", "| run | arrays | solves | verdict |", "|---|---|---|---|"]
    n_eq = 0
    for i in ids:
        r = compare_run(os.path.join(args.a, i), os.path.join(args.b, i))
        n_eq += r["equal"]
        verdict = "bitwise equal" if r["equal"] else f"DIFFERS ({', '.join(r['differs'])}" + (
            f", first frame {r['first_diff_frame']})" if "first_diff_frame" in r else ")")
        lines.append(f"| {i} | {', '.join(r['arrays'])} | {r.get('solves_equal', 'not logged')} | {verdict} |")
    lines.append("")
    lines.append(f"{n_eq} / {len(ids)} runs bitwise equal")
    text = "\n".join(lines)
    print(text)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
