"""Per-solve iteration counts and residuals, Metal vs L4 runs (first N solves).

    python solves_cmp.py SCENE N [name=dir ...]     (default: L4 base / nofma / precise and Metal fm_off / fm_cuda)
"""
import sys

import numpy as np

F = "runs/parity/isaaclab3_mpm_floors/mpm_floors"
FM = "runs/mpm"
scene, n = sys.argv[1], int(sys.argv[2])
runs = [a.split("=", 1) for a in sys.argv[3:]] or [
    ("base", f"{F}/{scene}_base"), ("nofma", f"{F}/{scene}_nofma"), ("precise", f"{F}/{scene}_precise"),
    ("m_off", f"{FM}/fm_off/F_{scene}"), ("m_cuda", f"{FM}/fm_cuda/F_{scene}")]
S = {name: np.load(d + "/solves.npz") for name, d in runs}
print("fields:", list(S[runs[0][0]].files))
for name, s in S.items():
    print(f"{name:10s} it", s["iterations"][:n].tolist())
for name, s in S.items():
    print(f"{name:10s} res_l2", " ".join(f"{x:.2e}" for x in s["res_l2"][:n]))
for name, s in S.items():
    it = s["iterations"]
    print(f"{name:10s} solves {it.size} mean it {it.mean():.3f}, at cap {(it >= it.max()).sum()}")
