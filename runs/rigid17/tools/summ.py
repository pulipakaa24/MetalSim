"""summ.py: collide17 JSON results under out/ (ab*/abdet*), one line each, then medians per configuration."""
import glob
import json
import os
import statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../out")
rows = {}
for f in sorted(glob.glob(f"{D}/ab*_r*.json")):
    r = json.load(open(f))
    k = os.path.basename(f)[:-5]
    print(k, round(r["collide_ms"]["median"], 2), r["rigid_contacts"], r["contact_sha"], r["contact_sha_after"])
    rows.setdefault(k.rsplit("_r", 1)[0], []).append(r["collide_ms"]["median"])
for k, v in rows.items():
    print(f"{k}: median {st.median(v):.2f} ms over {len(v)} reps {[round(x, 2) for x in v]}")
for f in sorted(glob.glob(f"{D}/ab*_r*.guard")):
    t = open(f).read()
    if "end code=0" not in t:
        print("NONZERO", os.path.basename(f), [ln for ln in t.splitlines() if "end code" in ln])
