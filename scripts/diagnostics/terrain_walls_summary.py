"""Falls and distance per cell type from runs/terrain_walls/transfer.jsonl (and the pre-change runs/g1_rough_transfer.jsonl),
split by policy source: Isaac's checkpoints in MetalSim, and ours (MetalSim-trained). Fall: pelvis above the local ground
< 0.3 m at the end of the 400 steps."""
import json, sys, collections
import numpy as np

NAMES = {0: "pyramid stairs", 4: "inverted stairs", 9: "boxes", 14: "random rough"}
rows = [json.loads(l) for l in open("runs/g1_rough_transfer.jsonl")]
for r in rows: r.setdefault("terrain_collision", "hfield (0b45075)"); r.setdefault("scan_surface", "grid")
rows += [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "runs/terrain_walls/transfer.jsonl")]
g = collections.OrderedDict()
for r in rows:
    g.setdefault((r["terrain_collision"], r.get("scan_surface", "grid"), r["src"]), []).append(r)
print("| surface | scan | policies | " + " | ".join(f"{NAMES[c]} falls / mean x" for c in (0, 4, 9, 14)) + " | total falls |")
print("|---|---|---|---|---|---|---|---|")
for (tc, sc, src), rs in g.items():
    cells, tot = [], 0
    for ci, c in enumerate(rs[0]["columns"]):
        f = sum(r["hag_final"][ci] < 0.3 for r in rs); tot += f
        cells.append(f"{f} / {len(rs)}, {np.mean([r['x'][ci] for r in rs]):.2f} m")
    who = "Isaac's (500/1000/1499)" if src == "rsl_rl" else "ours (500/1000/1500)"
    print(f"| {tc} | {sc} | {who} | " + " | ".join(cells) + f" | {tot} / {len(rs) * 4} |")
# Isaac's own playback (same protocol in Isaac Sim) for reference
rs = [r for r in rows if r["terrain_collision"] == "hfield (0b45075)"]
for src in ("rsl_rl", "metalsim"):
    q = [r for r in rs if r["src"] == src]
    cells = [f"{sum(r['isaac_hag_final'][ci] < 0.3 for r in q)} / {len(q)}, {np.mean([r['isaac_x'][ci] for r in q]):.2f} m" for ci in range(4)]
    print(f"| Isaac Sim (PhysX trimesh) | - | {'Isaac' if src == 'rsl_rl' else 'ours'} | " + " | ".join(cells) + " |")
