"""Append a row to the commits table of a fork's METALSIM_CHANGES.md (after the file's last table line).

    python3 add_row.py warp|mjw FILE SHORT_SHA
"""
import sys

kind, path, sha = sys.argv[1:4]
ROWS = {
    "warp": (
        f"| `{sha}` | Deterministic mode on Metal: a consumed-return counter call in the execution pass that the counting "
        "pass did not record (or recorded with another destination or count), and a thread that makes fewer counter "
        "calls in the execution pass, set bit 2 of the sticky deterministic flag; launches outside a capture raise "
        "\"Deterministic counter phase mismatch\". The counting pass suppresses stores, so a kernel whose counter calls "
        "depend on its own stores (the deterministic execution guide's \"Side effects in the counting pass\") otherwise "
        "hands slots to the wrong calls silently; MuJoCo Warp's CCD kernel did. Tests in "
        "`warp/tests/test_metal_deterministic.py`. |"),
    "mjw": (
        f"| `{sha}` | Deterministic mode (Warp `RUN_TO_RUN`): non-heightfield CCD runs as a solve kernel without a slot "
        "counter (EPA / multicontact scratch row = thread id, launch width at most `naccdmax`) that stages each "
        "collision's contacts, then a contact-write kernel. The single kernel took its EPA row from a counter, so "
        "Warp's counting pass ran EPA on stale scratch and a contact could land in another world's slot. In "
        "deterministic mode a primitive against a heightfield (GJK/EPA) raises. `MJW_CCD_DET_SPLIT=0` keeps the single "
        "kernel; default mode unchanged. |"),
}
lines = open(path).read().split("\n")
last = max(i for i, l in enumerate(lines) if l.startswith("| "))
if sha in lines[last]:
    print("row already present")
    sys.exit(0)
lines.insert(last + 1, ROWS[kind])
open(path, "w").write("\n".join(lines))
print(f"row added after line {last + 1}")
