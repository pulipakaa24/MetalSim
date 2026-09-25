"""One row per MuJoCo C setting (substep_forces.py recordings): 5 ms / 20 ms peak of the total vertical contact force in
the drop landing, the drop's torso impact and the hold's torso impact, impulses over the whole event window (N s, steps lo..hi, incl. the resting load), and foot-contact flicker
(transitions of Isaac's 1 N flag in A_hold + C_drop, both feet; at 2.5 ms and on 5 ms averages). Isaac's reference row
uses the bounds of win5_bounds.py and the momentum impulses of momentum_impulse.py."""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from flicker import phases
MI = json.load(open(os.path.join(ROOT, "runs/contact_research/momentum_impulse.json")))
EV = [("C_drop", "landing", 5, 30), ("C_drop", "torso", 55, 90), ("A_hold", "torso", 55, 90)]
print("| setting | " + " | ".join(f"{t} {e}: 5 ms / 20 ms peak N, impulse N s" for t, e, _, _ in EV) + " | flicker (A+C, 2.5 ms / 5 ms avg) |")
print("|---|" + "---|" * (len(EV) + 1))
row = ["Isaac PhysX (momentum-derived; 5 ms = bounds)"]
for t, e, lo, hi in EV:
    J = np.array(MI[t]["isaac"]["J"]); F = np.array(MI[t]["isaac"]["Fs"]); rest = np.clip(J[lo:hi] - 0.005 * F[lo:hi], 0, None)
    row.append(f"{np.maximum(F[lo:hi], rest / 0.015).max():.0f}-{np.maximum(F[lo:hi], rest / 0.005).max():.0f} / {J[lo:hi].max() / 0.02:.0f}, {J[lo:hi].sum():.1f}")
row.append("n/a (5 ms samples not recorded)"); print("| " + " | ".join(row) + " |")
for p in sys.argv[1:]:
    Z = np.load(os.path.join(ROOT, f"runs/contact_research/substep_c_{p}.npz")); b = list(Z["bodies"]); row = [p]
    for t, e, lo, hi in EV:
        fz = Z[f"{t}_force"][:, :, 2].sum(1).reshape(-1, 8)[lo + 1:hi + 1]
        row.append(f"{fz.reshape(-1, 2).mean(1).max():.0f} / {fz.mean(1).max():.0f}, {fz.sum() * 0.0025:.1f}")
    n25 = n5 = 0
    for t in ("A_hold", "C_drop"):
        for nm in ("left_ankle_roll_link", "right_ankle_roll_link"):
            f = np.linalg.norm(Z[f"{t}_force"][:, b.index(nm)], axis=-1)
            n25 += phases(f > 1.0, 0.0025)[0]; n5 += phases(f.reshape(-1, 2).mean(1) > 1.0, 0.005)[0]
    row.append(f"{n25} / {n5}"); print("| " + " | ".join(row) + " |")
