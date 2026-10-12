import sys
files = sys.argv[1:]; sys.argv = sys.argv[:1]
exec(open(__file__.rsplit("/",1)[0] + "/classify.py").read().split("B1, B2, C1, C2 =")[0])
SKIP = ("t0", "t1", "dt", "times", "elapsed", "wall", "ms", "build_s", "tdev", "host", "ev[", "png", "src", "scene", "_")
R = [load(p) for p in files]
cat = {}
for t in sorted(R[0]):
    vals = [dict(flat(r[t]["locals"])) if r.get(t) and r[t].get("locals") else {} for r in R]
    for k in vals[0]:
        if k.startswith("@py") or not all(k in v for v in vals): continue
        b1, b2, c1, c2 = (key(v[k]) for v in vals)
        c = ("B-rep" if b1 == b2 else "B-NONDET") + "/" + ("C-rep" if c1 == c2 else "C-NONDET")
        cat.setdefault(c, {}).setdefault(t, []).append(k)
for c in sorted(cat):
    print(f"\n=== {c}: {sum(map(len, cat[c].values()))} locals in {len(cat[c])} tests")
    if c != "B-rep/C-rep":
        for t, ks in cat[c].items(): print(f"  {t}: {', '.join(ks[:12])}{' ...' if len(ks) > 12 else ''}")
