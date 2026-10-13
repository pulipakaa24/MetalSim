import sys
sys.argv, files = sys.argv[:1], sys.argv[1:]
exec(open(__file__.rsplit("/",1)[0] + "/classify.py").read().split("B1, B2, C1, C2 =")[0])
R = [load(p) for p in files]; names = [f.split("/")[-1][:-6] for f in files]
for t in sorted(R[0]):
    vals = [dict(flat(r[t]["locals"])) if r.get(t) and r[t].get("locals") else {} for r in R]
    rows = []
    for k in vals[0]:
        if k.startswith("@py") or not all(k in v for v in vals): continue
        ks = [key(v[k]) for v in vals]
        if len(set(ks)) > 1 and not (ks[0] == ks[1] and ks[2] == ks[3]):
            rows.append((k, [fmt(v[k]) for v in vals]))
    if rows:
        print(f"\n# {t}")
        for k, fs in rows[:30]:
            print(f"  {k}"); [print(f"     {n}: {f}") for n, f in zip(names, fs)]
