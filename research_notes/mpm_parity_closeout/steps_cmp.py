"""Step-by-step device comparison of dump_steps.py outputs, per material and per field.

    python steps_cmp.py REF=dir A=dir [B=dir ...]   (multi_material material ranges)
For every run against REF: RMS over the material's particles of the field difference, per step; and the number of
particles whose field differs at all.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm_per_material import C, NAMES  # noqa: E402

args = [a.split("=", 1) for a in sys.argv[1:]]
ref_name, ref = args[0][0], np.load(args[0][1] + "/steps.npz")
fields = [f for f in ref.files]
steps = ref["particle_q"].shape[0]
nmax = int(__import__("os").environ.get("NSTEPS", steps))
for name, d in args[1:]:
    r = np.load(d + "/steps.npz")
    print(f"\n######## {name} vs {ref_name}")
    for f in fields:
        a = ref[f].astype(np.float64)
        b = r[f].astype(np.float64)
        print(f"-- {f}")
        for k in range(1, 4):
            s = slice(C[k], C[k + 1])
            dd = (b[:nmax, s] - a[:nmax, s]).reshape(nmax, C[k + 1] - C[k], -1)
            rms = np.sqrt((dd ** 2).sum(-1).mean(-1))
            nz = (np.abs(dd).max(-1) > 0).sum(-1)
            print(f"   {NAMES[k]:5s} rms " + " ".join(f"{x:8.2e}" for x in rms) + " | nz " + " ".join(f"{x:6d}" for x in nz))
