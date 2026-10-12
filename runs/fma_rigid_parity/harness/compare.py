"""compare.py A.jsonl B.jsonl [C.jsonl]: outcome per test and locals that differ (bitwise) between runs."""
import json, sys

def load(p):
    r = {}
    for line in open(p):
        j = json.loads(line); k = j["nodeid"]
        if j["when"] == "call" or k not in r:
            r[k] = j
    return r

def oc(j):
    if j is None: return "-"
    return ("xfail" if j["xfail"] and j["outcome"] == "skipped" else "xpass" if j["xfail"] else j["outcome"])

def flat(v, p=""):
    if isinstance(v, dict):
        if "sha1" in v:
            yield p, v; return
        for k, x in v.items(): yield from flat(x, f"{p}.{k}" if p else k)
    elif isinstance(v, list):
        for i, x in enumerate(v): yield from flat(x, f"{p}[{i}]")
    else:
        yield p, v

def fmt(v):
    if isinstance(v, dict) and "sha1" in v:
        return f"{v['shape']} {v['dtype']} max={v.get('max','')!s:.10} mean={v.get('mean','')!s:.10} min={v.get('min','')!s:.10} nf={v.get('nonfinite','')}"
    return repr(v)[:80]

runs = [load(p) for p in sys.argv[1:]]
names = [p.split("/")[-1].replace(".jsonl", "") for p in sys.argv[1:]]
ids = sorted(set().union(*runs))
print("== outcomes"); nflip = 0
for k in ids:
    o = [oc(r.get(k)) for r in runs]
    if len(set(o)) > 1:
        nflip += 1; print("FLIP", k, dict(zip(names, o)))
print(f"{len(ids)} tests, {nflip} outcome differences")
from collections import Counter
for nm, r in zip(names, runs): print(nm, dict(Counter(oc(j) for j in r.values())))
print("== locals differing (vs first run)")
same = diff = 0
for k in ids:
    a = runs[0].get(k)
    if not a or not a.get("locals"): continue
    fa = dict(flat(a["locals"]))
    rows = []
    for nm, r in zip(names[1:], runs[1:]):
        b = r.get(k)
        if not b or not b.get("locals"): continue
        fb = dict(flat(b["locals"]))
        for key in fa:
            if key.startswith("@py") or key not in fb: continue
            va, vb = fa[key], fb[key]
            eq = (va.get("sha1") == vb.get("sha1") and va.get("shape") == vb.get("shape")) if isinstance(va, dict) and isinstance(vb, dict) and "sha1" in va else va == vb
            if not eq: rows.append((nm, key, va, vb))
    if rows:
        diff += 1; print(f"\n# {k}  ({len(rows)} differing locals)")
        for nm, key, va, vb in rows[:40]:
            print(f"   {key:28s} {names[0]}: {fmt(va)}\n   {'':28s} {nm}: {fmt(vb)}")
    else:
        same += 1
print(f"\n{same} tests with all captured locals bitwise equal, {diff} with differences")
