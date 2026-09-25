"""Bounds on Isaac's (PhysX) largest 5 ms-averaged contact force during each impact, from its own recording:
per control step, the momentum-derived impulse J (momentum_impulse.py) and the sensor's last-5 ms sample F_last
give the impulse of the first three physics steps, J - 5 ms * F_last, hence
   lower bound (spread evenly over the three steps) = (J - 5 ms F_last) / 15 ms,
   upper bound (all in one step)                    = (J - 5 ms F_last) / 5 ms,
and max(F_last, those) bounds the step's largest 5 ms force. Compared with MuJoCo's max 5 ms force (aligned
substep pairs, sum over bodies) from substep_forces.py. Forces are totals over all bodies (vertical)."""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MI = json.load(open(os.path.join(ROOT, "runs/contact_research/momentum_impulse.json")))
presets = sys.argv[1:] or ["default", "tau10_impact_hardlimits", "tau5_imp99_hardlimits"]
for tag, events in (("A_hold", {"torso_impact": (55, 90)}), ("C_drop", {"landing": (5, 30), "torso_impact": (55, 90)})):
    J = np.array(MI[tag]["isaac"]["J"]); F = np.array(MI[tag]["isaac"]["Fs"])
    for ev, (lo, hi) in events.items():
        rest = np.clip(J[lo:hi] - 0.005 * F[lo:hi], 0, None)
        lb = np.maximum(F[lo:hi], rest / 0.015).max(); ub = np.maximum(F[lo:hi], rest / 0.005).max()
        line = f"{tag:7s} {ev:13s} Isaac max 5 ms force in [{lb:6.0f}, {ub:6.0f}] N (sample max {F[lo:hi].max():5.0f}, 20 ms max {J[lo:hi].max()/0.02:5.0f})"
        for p in presets:
            Z = np.load(os.path.join(ROOT, f"runs/contact_research/substep_c_{p}.npz"))
            fz = Z[f"{tag}_force"][:, :, 2].sum(1).reshape(-1, 8)[lo + 1:hi + 1]     # control steps t+1 (J index t = step t+1)
            w5 = fz.reshape(-1, 2).mean(1).max(); w20 = fz.mean(1).max(); inst = fz.max()
            line += f" | {p}: 5 ms {w5:5.0f}, 20 ms {w20:5.0f}, 2.5 ms {inst:5.0f}"
        print(line)
