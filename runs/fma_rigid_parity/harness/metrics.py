import sys
files = sys.argv[1:]; sys.argv = sys.argv[:1]
exec(open(__file__.rsplit("/",1)[0] + "/classify.py").read().split("B1, B2, C1, C2 =")[0])
R = [load(p) for p in files]; names = [f.split("/")[-1][:-6] for f in files]
SK = ("t0","dt","host","total","times","tdev","_","build_s","fps","mean_step_ms","h.","ev[","tf")
print("test | metric | " + " | ".join(names))
for t in sorted(R[0]):
    vals = [dict(flat(r[t]["locals"])) if r.get(t) and r[t].get("locals") else {} for r in R]
    for k in vals[0]:
        if k.startswith("@py") or any(k.startswith(s) for s in SK) or not all(k in v for v in vals): continue
        xs = [v[k] for v in vals]
        if isinstance(xs[0], float) and not isinstance(xs[0], bool):
            if len(set(xs)) > 1: print(f"{t.replace('tests/','')} | {k} | " + " | ".join(f"{x:.6g}" for x in xs))
        elif isinstance(xs[0], dict) and "sha1" in xs[0] and any(s in k.lower() for s in ("err", "worst", "diff", "dev", "agg", "e")) and len(k) <= 8:
            if len({x["sha1"] for x in xs}) > 1 and all("max" in x for x in xs):
                print(f"{t.replace('tests/','')} | max({k}) | " + " | ".join(f"{x['max']:.6g}" for x in xs))
