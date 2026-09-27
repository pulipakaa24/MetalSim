"""Landing normal-force trace of the G1 1 m drop, like for like with PhysX's per-5 ms-step recording: summed foot normal force
applied in each 5 ms window [5 (k-1), 5 k) ms (PhysX step k: runs/parity3/isaac/penetration_il2 tensor_normal_force; ours:
record_g1's per-substep |net normal force| of both feet, averaged over the two 2.5 ms substeps of the window), env 0, plus the
per-substep values and the landing impulse over 225-300 ms.
    python scripts/diagnostics/speculative/force_trace.py REC_DIR [REC_DIR...]"""
import os, sys, json, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
T0, T1 = 225, 300
z = np.load(os.path.join(ROOT, "runs/parity3/isaac/penetration_il2/C_drop_penetration.npz"))
feet = np.isin(z["body_names"][z["tensor_body"]], ["left_ankle_roll_link", "right_ankle_roll_link"])
W = list(range(T0, T1, 5))
px = {t: float(z["tensor_normal_force"][(z["tensor_step"] == t // 5 + 1) & feet].sum()) for t in W}
rows = [("Isaac 5.1 PhysX (GPU), N per 5 ms step", px, None)]
for dr in sys.argv[1:]:
    Z = np.load(os.path.join(ROOT, dr, "C_drop.npz")); dt = json.load(open(os.path.join(ROOT, dr, "meta.json")))["physics_dt"]
    F = Z["sub_force"].reshape(-1, Z["sub_force"].shape[2], 3)[:, 0, :2].sum(-1)       # (substeps,) env 0, both feet
    st = int(round(0.005 / dt))
    r = {t: float(F[int(round(t * 1e-3 / dt)): int(round(t * 1e-3 / dt)) + st].mean()) for t in W}
    sub = F[int(round(T0 * 1e-3 / dt)): int(round(T1 * 1e-3 / dt))]
    rows.append((os.path.basename(dr), r, sub))
print("| window start [ms] | " + " | ".join(str(t) for t in W) + " | impulse 225-300 ms [N s] |"); print("|---" * (len(W) + 2) + "|")
for nm, r, sub in rows:
    print(f"| {nm} | " + " | ".join(f"{r[t]:.0f}" for t in W) + f" | {sum(r.values()) * 0.005:.1f} |")
for nm, r, sub in rows[1:]:
    print(f"\n{nm} per 2.5 ms substep from {T0} ms: " + " ".join(f"{v:.0f}" for v in sub[:16]))
