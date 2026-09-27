"""Cap-probe table for the penetration sweep (runs/penetration/cap/<preset>_{random,policy}.json, cap_probe_ellip.py)."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print("| preset | states | worlds at cap 20 per step (of 4096) | cap 20: p99 |dqvel| / worlds > 0.01 | floor (cap 100 vs 100): p99 / > 0.01 | mean / max Newton iterations (cap 100) |")
print("|---|---|---|---|---|---|")
for p in sys.argv[1:]:
    for src in ("random", "policy"):
        f = os.path.join(ROOT, f"runs/penetration/cap/{p}_{src}.json")
        if not os.path.exists(f): continue
        d = json.load(open(f)); c = d["summary_cap20"]; fl = d["summary_floor"]
        mx = max(pt["max_niter_cap100"] for pt in d["points"])
        print(f"| {p} | {src} | {c['worlds_hitting_cap_mean_per_step']:.1f} | {c['p99_dqvel_max']:.4f} / {c['worlds_dqvel_gt_0.01_mean']:.1f} "
              f"| {fl['p99_dqvel']:.4f} / {fl['worlds_dqvel_gt_0.01']:.1f} | {d['summary_cap100']['mean_niter']:.2f} / {mx} |")
