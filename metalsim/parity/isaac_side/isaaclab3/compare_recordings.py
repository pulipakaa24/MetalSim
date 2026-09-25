"""Compare two Isaac-side fidelity recordings (record_g1.py output dirs) protocol by protocol, joints matched by name.

    python -m metalsim.parity.isaac_side.isaaclab3.compare_recordings REF_DIR OTHER_DIR [--json out.json]

Metrics per protocol (env 0): joint RMSE over the 37 joints at 0.5 / 1 / 2 s and at the end, max joint error, first step
with any joint > 0.1 rad off (divergence), final root z of each, root-z RMSE, torque RMS of each, contact force:
mean over steps of the summed per-body force magnitude and its peak (both recordings' `contact` arrays, whatever
the backend reports there), and solver iterations when recorded (Newton).
"""
import argparse, json, os
import numpy as np

PROTOCOLS = ("A_hold", "B_random", "C_drop")


def load(d, tag):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, f"{tag}.npz"))
    return meta, {k: z[k] for k in z.files}


def compare(ref_dir, oth_dir):
    out = {}
    for tag in PROTOCOLS:
        if not (os.path.exists(os.path.join(ref_dir, f"{tag}.npz")) and os.path.exists(os.path.join(oth_dir, f"{tag}.npz"))):
            continue
        mr, r = load(ref_dir, tag); mo, o = load(oth_dir, tag)
        idx = [mo["joint_names"].index(n) for n in mr["joint_names"]]
        dt = float(mr["control_dt"])
        qr = r["joint_pos"][:, 0]; qo = o["joint_pos"][:, 0][:, idx]
        n = min(len(qr), len(qo)); qr, qo = qr[:n], qo[:n]
        err = qo - qr
        rmse_t = np.sqrt((err ** 2).mean(1))
        at = lambda s: float(rmse_t[min(n - 1, int(round(s / dt)) - 1)])  # noqa: E731
        div = np.where(np.abs(err).max(1) > 0.1)[0]
        zr = r["root_pos"][:n, 0, 2]; zo = o["root_pos"][:n, 0, 2]
        tr = r["torque"][:n, 0]; to = o["torque"][:n, 0][:, idx]
        cr = np.linalg.norm(r["contact"][:n, 0], axis=-1).sum(-1) if "contact" in r else None
        co = np.linalg.norm(o["contact"][:n, 0], axis=-1).sum(-1) if "contact" in o else None
        row = {"steps": n, "joint_rmse_0.5s": at(0.5), "joint_rmse_1s": at(1.0), "joint_rmse_2s": at(2.0), "joint_rmse_end": float(rmse_t[-1]),
               "joint_err_max": float(np.abs(err).max()), "divergence_step": int(div[0]) if len(div) else None,
               "divergence_s": float((div[0] + 1) * dt) if len(div) else None,
               "root_z_end": [float(zr[-1]), float(zo[-1])], "root_z_rmse": float(np.sqrt(((zo - zr) ** 2).mean())),
               "torque_rms": [float(np.sqrt((tr ** 2).mean())), float(np.sqrt((to ** 2).mean()))],
               "peak_joint_speed": [float(np.abs(r["joint_vel"][:n]).max()), float(np.abs(o["joint_vel"][:n]).max())]}
        if cr is not None and co is not None:
            row["contact_mean_total"] = [float(cr.mean()), float(co.mean())]; row["contact_peak"] = [float(cr.max()), float(co.max())]
        for name, rec in (("ref", r), ("other", o)):
            if "solver_niter" in rec:
                s = rec["solver_niter"]
                row[f"solver_niter_{name}"] = {"mean": float(s.mean()), "max": int(s.max()), "min": int(s.min())}
        out[tag] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("other"); ap.add_argument("--json")
    a = ap.parse_args()
    res = compare(a.ref, a.other)
    for tag, row in res.items():
        print(tag, json.dumps(row))
    if a.json:
        json.dump({"ref": a.ref, "other": a.other, "protocols": res}, open(a.json, "w"), indent=1)
