"""Aggregate runs/competitors/g1_falls/*.json (g1_falls.py) into one table: falls per preset and seed, timing, direction."""
import glob, json, os, sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runs", "competitors", "g1_falls")
rows = {}
for f in sorted(glob.glob(os.path.join(D, "*.json"))):
    r = json.load(open(f)); key = (os.path.basename(r["ckpt"]).replace("_metalsim.pt", ""), r["contact_cfg"]); rows.setdefault(key, []).append(r)
print("| checkpoint | preset | falls per seed (episodes) | mean ± sd | envs that fell (twice+) | fall time < 1 s / 1-5 / 5-15 / >= 15 s | median s | forward / backward / sideways / upright 40 ms before | mean |cmd_xy| at fall (< 0.1: n) |")
print("|---|---|---|---|---|---|---|---|---|")
for (ck, p), rs in rows.items():
    falls = np.array([r["falls"] for r in rs]); ft = {k: sum(r["fall_time_s"][k] for r in rs) for k in ("lt_1", "1_to_5", "5_to_15", "ge_15")}
    med = np.median([f["ep_len_s"] for r in rs for f in r["falls_detail"]]) if falls.sum() else float("nan")
    dr = {k: sum(r["direction"][k] for r in rs) for k in rs[0]["direction"]}
    cm = np.mean([np.linalg.norm(f["cmd"][:2]) for r in rs for f in r["falls_detail"]]) if falls.sum() else float("nan")
    lo = sum(r["cmd_norm_at_fall_lt_0.1"] for r in rs)
    print(f"| {ck} | {p} | {' / '.join(f'{r[\"falls\"]} ({r[\"episodes\"]})' for r in rs)} | {falls.mean():.0f} ± {falls.std(ddof=1) if len(rs) > 1 else 0:.0f} | "
          f"{' / '.join(f'{r[\"envs_that_fell\"]} ({r[\"envs_fell_twice_or_more\"]})' for r in rs)} | {ft['lt_1']} / {ft['1_to_5']} / {ft['5_to_15']} / {ft['ge_15']} | {med:.1f} | "
          f"{' / '.join(str(v) for v in dr.values())} | {cm:.2f} ({lo}) |")
