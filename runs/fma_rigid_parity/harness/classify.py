import json, sys
def load(p):
    r = {}
    for line in open(p):
        j = json.loads(line)
        if j["when"] == "call" or j["nodeid"] not in r: r[j["nodeid"]] = j
    return r
def flat(v, p=""):
    if isinstance(v, dict):
        if "sha1" in v: yield p, v; return
        for k, x in v.items(): yield from flat(x, f"{p}.{k}" if p else k)
    elif isinstance(v, list):
        for i, x in enumerate(v): yield from flat(x, f"{p}[{i}]")
    else: yield p, v
def key(v): return (v.get("sha1"), tuple(v.get("shape", ()))) if isinstance(v, dict) and "sha1" in v else json.dumps(v, sort_keys=True, default=str)
def fmt(v):
    if isinstance(v, dict) and "sha1" in v:
        g=lambda x: f"{x:.7g}" if isinstance(x,float) else str(x)
        return f"{v['shape']} max={g(v.get('max'))} mean={g(v.get('mean'))} min={g(v.get('min'))} nf={v.get('nonfinite','')}"
    return (f"{v:.7g}" if isinstance(v,float) else repr(v)[:90])
B1, B2, C1, C2 = [load(p) for p in sys.argv[1:5]]
tree_tests = {}; noise_tests = set(); ntree = nnoise = nsame = 0
for t in sorted(B1):
    vals = []
    for r in (B1, B2, C1, C2):
        j = r.get(t); vals.append(dict(flat(j["locals"])) if j and j.get("locals") else {})
    for k in vals[0]:
        if k.startswith("@py") or not all(k in v for v in vals): continue
        b1, b2, c1, c2 = (key(v[k]) for v in vals)
        if b1 == b2 == c1 == c2: nsame += 1
        elif b1 == b2 and c1 == c2: ntree += 1; tree_tests.setdefault(t, []).append((k, vals[0][k], vals[2][k]))
        else: nnoise += 1; noise_tests.add(t)
print(f"locals: {nsame} identical in all 4 runs, {ntree} TREE-induced (stable within each tree), {nnoise} run-to-run noise")
print(f"tests with TREE-induced differences: {len(tree_tests)}; tests with run-to-run noise: {len(noise_tests)}")
print("noise tests:", *sorted(noise_tests), sep="\n  ")
for t, rows in tree_tests.items():
    print(f"\n# {t}  ({len(rows)} locals)")
    for k, b, c in rows[:60]:
        print(f"   {k:30s} base: {fmt(b)}\n   {'':30s} cand: {fmt(c)}")
