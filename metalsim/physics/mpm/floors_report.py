"""Metal vs the L4's own floors (MPM.md section 5, "against CUDA's own floors").

    python -m metalsim.physics.mpm.floors_report L4_FLOORS_DIR METAL_DIR [METAL_DIR ...] [--json out.json] [--md out.md]

L4_FLOORS_DIR holds stage_mpm_floors.sh output (<scene>_base and the perturbed <scene>_{precise,nofma,cubin,sah});
each METAL_DIR holds record_variant_suite.sh "floors" runs (F_<scene>). Per scene: particle-position RMS of Metal
against the L4 baseline at the first, middle and last common recorded frame, and the worst L4 floor (largest RMS of
a perturbed L4 run against the same baseline) at the same frames; "inside" when Metal is at or below the worst floor
at every one of the three frames. Plate drag: smoothed force-vs-travel RMS relative to the L4 peak, and final force.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from metalsim.physics.mpm import compare

SCENES = ["column_a2", "granular", "multi_material", "twoway", "anymal"]
PERTURB = ["precise", "nofma", "cubin", "sah"]


def _have(path):
    return os.path.exists(os.path.join(path, "meta.json"))


def _rms_at(base, other, frames):
    tm = compare.trajectory_metrics(base, other)
    fr = np.asarray(tm["frames"])
    out = []
    for f in frames:
        i = int(np.argmin(np.abs(fr - f)))
        out.append(float(tm["pos_rms"][i]))
    return out, tm


def _force(run):
    f = np.abs(run["traj"]["x_plate_force"][:, 0])
    k = np.ones(5) / 5
    return np.convolve(f, k, "same")


def scene_row(l4, mt, scene):
    base_dir = os.path.join(l4, f"{scene}_base")
    m_dir = os.path.join(mt, f"F_{scene}")
    if not (_have(base_dir) and _have(m_dir)):
        return None
    base, m = compare.load(base_dir), compare.load(m_dir)
    frames = list(compare._common_frames(base, m)[0])
    pick = [frames[0], frames[len(frames) // 2], frames[-1]]
    metal, tm = _rms_at(base, m, pick)
    floors = {}
    for p in PERTURB:
        d = os.path.join(l4, f"{scene}_{p}")
        if _have(d):
            floors[p] = _rms_at(base, compare.load(d), pick)[0]
    worst = [max(v[i] for v in floors.values()) for i in range(3)] if floors else [float("nan")] * 3
    worst_name = [max(floors, key=lambda k: floors[k][i]) for i in range(3)] if floors else [""] * 3
    row = {"scene": scene, "frames": [int(x) for x in pick], "metal_rms_m": metal, "worst_floor_m": worst,
           "worst_floor_from": worst_name, "ratio": [a / b if b > 0 else float("inf") for a, b in zip(metal, worst)],
           "floors_m": floors, "inside": bool(all(a <= b for a, b in zip(metal, worst))),
           "iters_l4": compare.solver_stats(base).get("iter_mean"), "iters_metal": compare.solver_stats(m).get("iter_mean"),
           "bitwise_equal_to_l4": tm["bitwise_equal"]}
    return row


def plate_row(l4, mt):
    base_dir, m_dir = os.path.join(l4, "plate_drag_base"), os.path.join(mt, "F_plate_drag")
    if not (_have(base_dir) and _have(m_dir)):
        return None
    base = compare.load(base_dir)
    sa = _force(base)
    peak = float(sa.max())

    def rel(run):
        sb = _force(run)
        n = min(sa.size, sb.size)
        return float(np.sqrt(np.mean((sa[:n] - sb[:n]) ** 2)) / peak), float(sb[n - 3])

    m_rel, m_final = rel(compare.load(m_dir))
    floors = {}
    for p in PERTURB:
        d = os.path.join(l4, f"plate_drag_{p}")
        if _have(d):
            floors[p] = rel(compare.load(d))
    worst = max(v[0] for v in floors.values()) if floors else float("nan")
    return {"scene": "plate_drag", "force_rel_rms_metal": m_rel, "force_rel_rms_worst_floor": worst,
            "ratio": m_rel / worst if worst > 0 else float("inf"), "final_force_l4_N": float(sa[sa.size - 3]),
            "final_force_metal_N": m_final, "final_force_floors_N": {k: v[1] for k, v in floors.items()},
            "inside": bool(m_rel <= worst)}


def report(l4, mt):
    rows = [r for r in (scene_row(l4, mt, s) for s in SCENES) if r]
    p = plate_row(l4, mt)
    return rows, p


def _fmt_m(x):
    return f"{x * 1e6:.3g} um" if x < 1e-3 else f"{x * 1e3:.3g} mm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("l4")
    ap.add_argument("metal", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args()
    out, md = {}, []
    for mt in args.metal:
        rows, p = report(args.l4, mt)
        out[mt] = {"scenes": rows, "plate": p}
        md += [f"### {mt}", "", "| scene | frames | Metal RMS (first / mid / last) | worst L4 floor | ratio | iters L4 / Metal | verdict |",
               "|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['scene']} | {r['frames']} | {' / '.join(_fmt_m(x) for x in r['metal_rms_m'])} | "
                      f"{' / '.join(_fmt_m(x) for x in r['worst_floor_m'])} ({', '.join(r['worst_floor_from'])}) | "
                      f"{' / '.join(f'{x:.2f}' for x in r['ratio'])} | {r['iters_l4']:.2f} / {r['iters_metal']:.2f} | "
                      f"{'inside' if r['inside'] else 'OUTSIDE'} |")
        if p:
            md.append(f"| plate_drag | force curve | {p['force_rel_rms_metal'] * 100:.2f} % of peak, final "
                      f"{p['final_force_metal_N']:.0f} N | {p['force_rel_rms_worst_floor'] * 100:.2f} %, final "
                      f"{', '.join(f'{v:.0f}' for v in p['final_force_floors_N'].values())} N (base "
                      f"{p['final_force_l4_N']:.0f} N) | {p['ratio']:.2f} | | {'inside' if p['inside'] else 'OUTSIDE'} |")
        md.append("")
    text = "\n".join(md)
    print(text)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(text + "\n")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
