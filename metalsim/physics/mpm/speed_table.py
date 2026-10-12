"""Table of speed_suite.sh results: median ms/frame (and host waits per frame) per scene and variant.

    python -m metalsim.physics.mpm.speed_table runs/mpm/speed base fastmath ...   [--md out.md]

With several repetitions (``<scene>_r<k>.json``) the value is the median over repetitions of the per-run medians.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

SCENES = ["granular", "multi_material", "twoway", "twoway_capture", "anymal", "anymal_capture", "plate_drag",
          "column_a8", "column_a2"]


def load(root: str, variant: str, scene: str):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, variant, f"{scene}_r*.json"))):
        if p.endswith("_r0.json"):
            continue  # smoke runs
        with open(p) as fh:
            runs.append(json.load(fh))
    if not runs:
        return None
    return {"ms": float(np.median([r["ms_median"] for r in runs])),
            "all": [round(r["ms_median"], 2) for r in runs],
            "waits": float(np.median([r["per_frame_counters"]["host_waits"] for r in runs])),
            "cbs": float(np.median([r["per_frame_counters"]["command_buffers"] for r in runs]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--md")
    args = ap.parse_args()
    base = args.variants[0]
    head = "| scene | " + " | ".join(f"{v} ms (waits, cbs)" for v in args.variants) + " | " + \
        " | ".join(f"{v}/{base}" for v in args.variants[1:]) + " |"
    lines = [head, "|" + "---|" * (2 * len(args.variants))]
    for s in SCENES:
        rows = [load(args.root, v, s) for v in args.variants]
        if rows[0] is None and all(r is None for r in rows):
            continue
        cells = [f"{r['ms']:.2f} {r['all'] if len(r['all']) > 1 else ''} ({r['waits']:g}, {r['cbs']:g})" if r else "-"
                 for r in rows]
        ratios = [f"{r['ms'] / rows[0]['ms']:.3f}" if r and rows[0] else "-" for r in rows[1:]]
        lines.append(f"| {s} | " + " | ".join(cells) + " | " + " | ".join(ratios) + " |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
