"""Move the Newton 1.5.2 fork pin from 3fd1f052 to NEWSHA in every place a fresh setup reads it.

    python move_pin.py NEWSHA
Idempotent. Historical mentions of 3fd1f052 in MPM.md sections 8f-8h and PARITY.md stay (they describe the runs).
"""
import pathlib
import re
import sys

OLD = "3fd1f052a5ad031578960d9642aaaedc03834e4d"
new = sys.argv[1]
assert re.fullmatch(r"[0-9a-f]{40}", new), new
W = pathlib.Path(__file__).resolve().parents[3]
o8, n8 = OLD[:8], new[:8]


def sub(path, pairs):
    p = W / path
    t = p.read_text()
    t0 = t
    for a, b in pairs:
        t = t.replace(a, b)
    if t != t0:
        p.write_text(t)
        print("updated", path)


sub("scripts/vendor_mpm152.py", [(f'PIN = "{OLD}"', f'PIN = "{new}"')])
sub("scripts/setup_newton152.sh", [
    (f"PIN={OLD}                               # metalsim-1.5.2 head, 2026-10-12 (MPM step at scale)",
     f"PIN={new}                               # metalsim-1.5.2 head, 2026-10-13 (GS coloring with separate worlds)"),
    (f"metalsim-1.5.2-{o8}.diff", f"metalsim-1.5.2-{n8}.diff")])
sub("THIRD_PARTY_NOTICES.md", [(OLD, new), (f"fork commit `{o8}`", f"fork commit `{n8}`")])
mpm = W / "metalsim/physics/mpm/MPM.md"
t = mpm.read_text()
i = t.index("## 10. Reproducing from a fresh clone")
j = t.index("## 11.", i)
sec = t[i:j].replace(f"metalsim-1.5.2 @{o8}", f"metalsim-1.5.2 @{n8}").replace(f"pinned `{o8}`", f"pinned `{n8}`") \
    .replace(f"metalsim-1.5.2-{o8}.diff", f"metalsim-1.5.2-{n8}.diff")
if sec != t[i:j]:
    mpm.write_text(t[:i] + sec + t[j:])
    print("updated MPM.md section 10")
sub("README.md", [("Branch [`metalsim-1.5.2`](https://github.com/pulipakaa24/newton/tree/metalsim-1.5.2) (`c0427fc`)",
                   f"Branch [`metalsim-1.5.2`](https://github.com/pulipakaa24/newton/tree/metalsim-1.5.2) (`{new[:7]}`)")])
