"""Isaac Lab 3.0 deformable recordings (record_deformables_il3.py): bulk metrics per backend and mesh resolution.
Rope: tip (end-face centroid) relative to the held face; rest drop, oscillation period and log decrement from the tip
signal about its final value (x for swings, z for cantilever bounces: whichever moves more after the first 0.3 s).
Cube: impact minimum, bounce peak, rest centroid, settling time, penetration. Cloth: height on the box, mean height,
extent, settling. usage: il3_analysis.py REC_ROOT [--json OUT]"""
import sys, os, json, numpy as np

HZ = 200.0


def osc(sig, t0=0.3):
    dt = 1 / HZ; k0 = int(t0 * HZ)
    s = sig[k0:] - sig[-int(0.5 * HZ):].mean()
    zc = np.where(np.diff(np.sign(s)) != 0)[0]
    if len(zc) < 3:
        return float("nan"), float("nan"), []
    period = float(2 * np.mean(np.diff(zc)) * dt)
    peaks = [float(np.abs(s[a:b]).max()) for a, b in zip(zc[:-1], zc[1:])]
    p = np.array(peaks[:6])
    decr = float(np.mean(np.log(p[:-1] / p[1:]))) if len(p) > 2 and (p > 0).all() else float("nan")
    return period, decr, [round(x, 4) for x in peaks[:6]]


def rope(pos, pinned):
    pin = pos[:, pinned].mean(1)
    x0 = pos[0]; far = x0[:, 0] > x0[:, 0].max() - 1e-4
    rel = pos[:, far].mean(1) - pin
    L = float(np.linalg.norm(rel[0]))
    ax = 0 if np.ptp(rel[int(0.3 * HZ):, 0]) > np.ptp(rel[int(0.3 * HZ):, 2]) else 2
    period, decr, peaks = osc(rel[:, ax])
    return {"length0": L, "rest_tip_drop": float(-rel[-1, 2]), "rest_tip_x": float(rel[-1, 0]), "min_tip_z": float(rel[:, 2].min()),
            "osc_axis": "xz"[ax // 2], "period": period, "log_decrement": decr, "peaks": peaks, "finite": bool(np.isfinite(pos).all())}


def cube(pos, vel, size=0.2):
    cz = pos[:, :, 2].mean(1); dt = 1 / HZ
    kmin = int(np.argmin(cz[: int(1.0 * HZ)])); peak = kmin + int(np.argmax(cz[kmin: kmin + int(0.5 * HZ)]))
    ke = 0.5 * (vel ** 2).sum((1, 2)); thr = 0.01 * ke.max(); above = np.where(ke > thr)[0]
    return {"t_impact_min": kmin * dt, "min_centroid_z": float(cz[kmin]), "bounce_peak_z": float(cz[peak]), "rest_centroid_z": float(cz[-1]),
            "rest_height": float(np.ptp(pos[-1, :, 2])), "settle_time": float((above[-1] + 1) * dt) if len(above) else 0.0,
            "min_node_z": float(pos[:, :, 2].min()), "finite": bool(np.isfinite(pos).all())}


def cloth(pos, vel, box=0.4):
    x = pos[-1]; on = (np.abs(x[:, 0]) < box / 2) & (np.abs(x[:, 1]) < box / 2)
    ke = 0.5 * (vel ** 2).sum((1, 2)); thr = 0.01 * ke.max(); above = np.where(ke > thr)[0]
    return {"rest_top_z": float(x[on, 2].mean()) if on.any() else float("nan"), "rest_mean_z": float(x[:, 2].mean()), "rest_min_z": float(x[:, 2].min()),
            "xy_extent": float(0.5 * (np.ptp(x[:, 0]) + np.ptp(x[:, 1]))), "settle_time": float((above[-1] + 1) / HZ) if len(above) else 0.0,
            "finite": bool(np.isfinite(pos).all())}


def analyse(d):
    m = json.load(open(os.path.join(d, "meta.json"))); r = np.load(os.path.join(d, "record.npz"))
    out = {"cloth": cloth(r["cloth_pos"], r["cloth_vel"]) if "cloth_pos" in r.files else None, "rope": {}, "cube": {}}
    for k in r.files:
        if k.startswith("rod_") and k.endswith("_pos"):
            n = k[len("rod_"):-len("_pos")]
            out["rope"][n] = rope(r[k], r[f"rod_{n}_pinned"].astype(bool))
        if k.startswith("cube_") and k.endswith("_pos"):
            n = k[len("cube_"):-len("_pos")]
            out["cube"][n] = cube(r[k], r[f"cube_{n}_vel"])
    out["material"] = m["scenes"].get("rod_n1", {}).get("material")
    return out


if __name__ == "__main__":
    root = sys.argv[1]; res = {}
    for B in ("isaacsim_physx", "newton_vbd"):
        if os.path.isdir(os.path.join(root, B)):
            res[B] = analyse(os.path.join(root, B))
            print("==", B)
            for n, v in res[B]["rope"].items():
                print(f"  rope {n}: drop {v['rest_tip_drop']:.3f} tip_x {v['rest_tip_x']:+.3f} osc({v['osc_axis']}) period {v['period']:.3f} decr {v['log_decrement']:.3f} peaks {v['peaks']}")
            for n, v in res[B]["cube"].items():
                print(f"  cube {n}: " + " ".join(f"{a} {b:.4f}" for a, b in v.items() if isinstance(b, float)))
            if res[B]["cloth"]:
                print("  cloth: " + " ".join(f"{a} {b:.3f}" for a, b in res[B]["cloth"].items() if isinstance(b, float)))
    if len(sys.argv) > 3 and sys.argv[2] == "--json":
        json.dump(res, open(sys.argv[3], "w"), indent=1)
