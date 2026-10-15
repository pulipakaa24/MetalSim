#!/usr/bin/env python3
"""P2 amendment (research_notes/sand_fidelity/PLAN_2026-10-14.md section 13): keep the 8-byte scratch elements, go
back to doubling growth. The pool keeps every outgrown buffer (captured graphs may still reference them), so the
retained chain sums to ~1/(1 - 1/g) x the largest buffer for a growth factor g: about 5x at g = 1.25 (f8a1d679) and
2x at g = 2. Batch 02b's voxel-0.005 S2 rest run allocated 57.7 GB against a 55.7 GB Metal working set with g = 1.25.

    python amend_growth.py <warp checkout>       # idempotent
"""
import pathlib
import sys

src = pathlib.Path(sys.argv[1]) / "warp/_src/metal_primitives.py"
OLD = '''        Allocated in 8-byte elements: Warp array shapes are int32, so a byte-element buffer could not exceed 2 GiB
        (an S2 collider rasterisation at 1.57M particles needed more). Grown to at most 1.25x the request beyond the
        current capacity, not doubled past it (doubling turned a 1.6 GiB request into a 3 GiB allocation)."""
        capacity = 8 * self.scratch[-1].size if self.scratch else 0
        if capacity < nbytes:
            size = max(nbytes, min(2 * capacity, nbytes + nbytes // 4), 1 << 16)
'''
NEW = '''        Allocated in 8-byte elements: Warp array shapes are int32, so a byte-element buffer could not exceed 2 GiB
        (an S2 collider rasterisation at 1.57M particles needed more). Grown by doubling: outgrown buffers stay alive
        (captured graphs may reference them), so the retained chain is ~2x the largest buffer; a smaller growth step
        retains more (1.25x steps: ~5x)."""
        capacity = 8 * self.scratch[-1].size if self.scratch else 0
        if capacity < nbytes:
            size = max(nbytes, 2 * capacity, 1 << 16)
'''
s = src.read_text()
if NEW in s:
    print("already amended")
elif OLD in s:
    src.write_text(s.replace(OLD, NEW))
    print("amended")
else:
    sys.exit("scratch_ptr does not match f8a1d679's text; not amending")
tst = pathlib.Path(sys.argv[1]) / "warp/tests/test_metal_primitives.py"
t = tst.read_text()
OLDT = "        self.assertLessEqual(nbytes, request + request // 4 + 8)\n"
NEWT = "        self.assertLessEqual(nbytes, request + 8)          # a fresh pool allocates the request (rounded to 8 B)\n"
if OLDT in t:
    tst.write_text(t.replace(OLDT, NEWT))
    print("test amended")
