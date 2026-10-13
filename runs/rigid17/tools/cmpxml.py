"""cmpxml.py A.xml B.xml: per-test outcomes of two junit files (pytest-collected free functions excluded), and the
tests whose outcome differs. Outcome: pass | fail | error | skip."""
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def load(p):
    out = {}
    for tc in ET.parse(p).getroot().iter("testcase"):
        name = f"{tc.get('classname')}::{tc.get('name')}"
        kids = {c.tag for c in tc}
        o = "fail" if "failure" in kids else "error" if "error" in kids else "skip" if "skipped" in kids else "pass"
        if o == "error" and any("fixture 'test' not found" in (c.get("message") or "") + (c.text or "") for c in tc):
            continue  # module-level helper functions pytest collects as tests (Newton runs them through unittest)
        out[name.split("upstream.", 1)[-1].split(".", 1)[-1] if "upstream." in name else name] = o
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
for lab, d in (("A", a), ("B", b)):
    c = Counter(d.values())
    metal = Counter(v for k, v in d.items() if "metal" in k)
    print(lab, dict(c), "metal:", dict(metal))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(f"{a.get(k, '-'):>6} -> {b.get(k, '-'):<6} {k}")
