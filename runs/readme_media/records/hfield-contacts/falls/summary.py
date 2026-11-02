"""Aggregate falls JSONs: per tag prefix, falls per seed over repeats. Usage: summary.py (in falls/)."""
import glob, json, os, collections
here = os.path.dirname(os.path.abspath(__file__))
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(os.path.join(here, "*_*.json"))):
  r = json.load(open(f)); name = os.path.basename(f)[:-5]
  grp = ("cpu (start perturbed 1e-6 rad)" if "pert" in name else "cpu") if name.startswith("cpu") else ("gpu fix" if "fix" in name else "gpu base (8eb51fe)")
  for s, v in r["seeds"].items():
    groups[grp][int(s)].append(v["falls"])
out = {}
for g, d in groups.items():
  runs = sum(len(v) for v in d.values()); tot = sum(sum(v) for v in d.values())
  out[g] = dict(per_seed={s: d[s] for s in sorted(d)}, robots=16 * runs, falls=tot, share=round(tot / (16 * runs), 3))
  print(f"{g:20s} " + "  ".join(f"seed {s}: {d[s]}" for s in sorted(d)) + f"  | total {tot}/{16 * runs} = {tot / (16 * runs):.3f}")
json.dump(out, open(os.path.join(here, "summary.json"), "w"), indent=1)
