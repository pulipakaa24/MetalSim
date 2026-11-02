"""Old-vs-new rough transfer (PARITY §1.5 protocols A and B) per checkpoint, from the jsonl records.

Protocol A (`runs/g1_rough_transfer.jsonl`, 2026-09-25; re-runs `transfer/*_protocol24.jsonl`): levels 3 and 6, one env per
column (pyramid stairs 0, inverted stairs 4, boxes 9, random rough 14), heightfield collision + grid scan, command
(0.5, 0, 0), 400 control steps, mean action; fall = final pelvis height above the local ground < 0.3 m
(scripts/diagnostics/g1_rough_transfer.py:4,180-191). Protocol B (`runs/terrain_walls/transfer_reps.jsonl`, rows
boxes_local + exact; re-runs `transfer/*_reps8.jsonl`): the same with 8 starts per column (rep 0 the protocol start,
reps 1-7 jittered by up to 5 cm). "In Isaac" columns (isaac_hag_final) come from Isaac's play records and are unchanged
by the MuJoCo Warp fix.

    .venv/bin/python runs/readme_media/scratch/rough-parity/analyze_transfer.py > runs/readme_media/scratch/rough-parity/transfer.md
"""
import json
import numpy as np

R = "runs/rough_rerun_2026-10-31/transfer"
NAMES = {0: "pyr. stairs", 4: "inv. stairs", 9: "boxes", 14: "random rough"}
FALL = 0.3


def load(path, keep=lambda r: True):
    return [r for r in (json.loads(l) for l in open(path)) if keep(r)]


def per_label(rows):
    out = {}
    for r in rows:
        d = out.setdefault(r["label"], {"falls": 0, "n": 0, "x": {c: [] for c in NAMES}, "f": {c: 0 for c in NAMES},
                                        "nc": {c: 0 for c in NAMES}, "isaac_falls": 0, "isaac_n": 0, "levels": set()})
        d["levels"].add(r["level"])
        for i, c in enumerate(r["columns"]):
            fell = r["hag_final"][i] < FALL
            d["falls"] += fell; d["n"] += 1; d["x"][c].append(r["x"][i]); d["f"][c] += fell; d["nc"][c] += 1
        if "isaac_hag_final" in r and r["isaac_hag_final"]:
            d["isaac_falls"] += sum(h < FALL for h in r["isaac_hag_final"]); d["isaac_n"] += len(r["isaac_hag_final"])
    return out


def show(title, rows):
    print(f"\n### {title}")
    print("| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | " + " | ".join(f"{NAMES[c]}: falls, mean x" for c in NAMES)
          + " | falls in Isaac (play records) |")
    print("|---|---|---|" + "---|" * len(NAMES) + "---|")
    tot = [0, 0, 0, 0]
    for lab, d in per_label(rows).items():
        wall = d["f"][4] + d["f"][9]; walln = d["nc"][4] + d["nc"][9]
        tot[0] += d["falls"]; tot[1] += d["n"]; tot[2] += wall; tot[3] += walln
        cells = " | ".join(f"{d['f'][c]} / {d['nc'][c]}, {np.mean(d['x'][c]):.2f} m" for c in NAMES)
        isaac = f"{d['isaac_falls']} / {d['isaac_n']}" if d["isaac_n"] else "-"
        print(f"| {lab} (levels {sorted(d['levels'])}) | {d['falls']} / {d['n']} | {wall} / {walln} | {cells} | {isaac} |")
    print(f"| total | {tot[0]} / {tot[1]} | {tot[2]} / {tot[3]} |" + " |" * len(NAMES) + " |")


def main():
    old_a = load("runs/g1_rough_transfer.jsonl")
    show("Protocol A, 2026-09-25 record (plane-path kernel; `runs/g1_rough_transfer.jsonl`)", old_a)
    for tag, what in (("old_pin", "pinned fork 8eb51fe (plane path), today's code"), ("old_fix", "fixed fork 93e5ce1"),
                      ("new_s0", "fixed fork 93e5ce1, re-trained seed 0")):
        show(f"Protocol A, {what} (`{R}/{tag}_protocol24.jsonl`)", load(f"{R}/{tag}_protocol24.jsonl"))
    old_b = load("runs/terrain_walls/transfer_reps.jsonl",
                 lambda r: r.get("terrain_collision") == "boxes_local" and r.get("scan_surface") == "exact")
    show("Protocol B, 2026-09-25 record (boxes_local + exact; `runs/terrain_walls/transfer_reps.jsonl`)", old_b)
    for tag, what in (("old_pin", "pinned fork 8eb51fe"), ("old_fix", "fixed fork 93e5ce1"), ("new_s0", "fixed fork 93e5ce1, re-trained seed 0")):
        show(f"Protocol B, {what} (`{R}/{tag}_reps8.jsonl`)", load(f"{R}/{tag}_reps8.jsonl"))
    # same checkpoint, pinned vs fixed: per-env x difference (A) as the size of the change
    print("\n### Same checkpoint, fixed minus pinned, protocol A and B (per env)")
    for proto in ("protocol24", "reps8"):
        p = {(r["label"], r["level"]): r for r in load(f"{R}/old_pin_{proto}.jsonl")}
        f = {(r["label"], r["level"]): r for r in load(f"{R}/old_fix_{proto}.jsonl")}
        for lab in sorted({k[0] for k in p}):
            for c in NAMES:
                dx = [f[(lab, lv)]["x"][i] - p[(lab, lv)]["x"][i] for lv in (3, 6) if (lab, lv) in f
                      for i, cc in enumerate(p[(lab, lv)]["columns"]) if cc == c]
                print(f"{proto} {lab} {NAMES[c]}: mean dx {np.mean(dx):+.2f} m (min {min(dx):+.2f}, max {max(dx):+.2f}, n {len(dx)})")


if __name__ == "__main__":
    main()
