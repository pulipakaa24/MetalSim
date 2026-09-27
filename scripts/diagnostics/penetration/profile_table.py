"""Landing time profile, like for like with PhysX's recording: deepest foot contact per 5 ms tick (pose time t), depth in mm
(positive = penetration, negative = the separation of a contact inside the contact offset / gap), G1 1 m drop.
PhysX: runs/parity3/isaac/penetration_il2 (step k reports the pose after step k-1, t = 5 (k-1) ms). Ours: record_g1's
sub_dist, substep s reports the pose at t = 2.5 s ms (contacts of the substep's start)."""
import os, sys, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
T = np.arange(225, 300, 5)
z = np.load(os.path.join(ROOT, "runs/parity3/isaac/penetration_il2/C_drop_penetration.npz"))
feet = np.isin(z["body_names"][z["tensor_body"]], ["left_ankle_roll_link", "right_ankle_roll_link"])
row = {}
for t in T:
    k = t // 5 + 1; m = (z["tensor_step"] == k) & feet
    row[t] = -z["tensor_separation"][m].min() * 1e3 if m.any() else None
out = [("Isaac 5.1 PhysX (GPU)", row)]
for dr in sys.argv[1:]:
    Z = np.load(os.path.join(ROOT, dr, "C_drop.npz")); d = -Z["sub_dist"].reshape(-1, Z["sub_dist"].shape[2], 3)[:, 0, :2]
    import json; dt = json.load(open(os.path.join(ROOT, dr, "meta.json")))["physics_dt"]
    r = {}
    for t in T:
        s = int(round(t * 1e-3 / dt)); v = d[s]
        r[t] = None if np.isnan(v).all() else float(np.nanmax(v)) * 1e3
    out.append((os.path.basename(dr), r))
print("| t [ms] | " + " | ".join(str(t) for t in T) + " |"); print("|---" * (len(T) + 1) + "|")
for nm, r in out:
    print(f"| {nm} | " + " | ".join("–" if r[t] is None else f"{r[t]:.2f}" for t in T) + " |")
