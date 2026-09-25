"""Fall rates per cell type and policy source from runs/terrain_walls/transfer_reps.jsonl (8 starts per cell type per
checkpoint and level: rep 0 = the protocol start, reps 1-7 jittered by up to +-5 cm)."""
import json, collections, sys
import numpy as np
NAMES = {0: "pyramid stairs", 4: "inverted stairs", 9: "boxes", 14: "random rough"}
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "runs/terrain_walls/transfer_reps.jsonl")]
g = collections.OrderedDict()
for r in rows:
    g.setdefault((r["terrain_collision"], r.get("scan_surface", "grid"), r["src"]), []).append(r)
print("| surface | scan | policies | " + " | ".join(f"{NAMES[c]}: falls, mean x" for c in (0, 4, 9, 14)) + " | wall cells (inv. stairs + boxes) |")
print("|---|---|---|---|---|---|---|---|")
for (tc, sc, src), rs in g.items():
    cols = np.tile([0, 4, 9, 14], rs[0].get("reps", 1)); cells = []; wall = [0, 0]
    for c in (0, 4, 9, 14):
        idx = np.nonzero(cols == c)[0]
        h = np.array([[r["hag_final"][i] for i in idx] for r in rs]); x = np.array([[r["x"][i] for i in idx] for r in rs])
        f = int((h < 0.3).sum()); cells.append(f"{f} / {h.size}, {x.mean():.2f} m")
        if c in (4, 9): wall[0] += f; wall[1] += h.size
    who = "Isaac's (500/1000/1499)" if src == "rsl_rl" else "ours (500/1000/1500)"
    print(f"| {tc} | {sc} | {who} | " + " | ".join(cells) + f" | {wall[0]} / {wall[1]} ({100 * wall[0] / wall[1]:.0f} %) |")
