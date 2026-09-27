"""CPU-only (numpy) analysis of the 2026-09-26 Isaac-side recordings in runs/parity3/isaac/ (no GPU, no MetalSim):
  dome   : per-pixel marker directions from the six cube-face PNGs -> Kit's latlong (u, v) -> world mapping
  pen    : PhysX per-contact separation during the G1 C_drop -> peak / time profile / settle
  fourbar: PhysX four-bar vs the exact planar reference (runs/closed_loops/ref/fourbar_none.npz): angle error,
           energy drift, loop-closure gap
    .venv/bin/python metalsim/parity/isaac_side/analyze_session_0926.py [dome|pen|fourbar ...]
"""
import sys, os, json, math, glob
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RUN = os.path.join(ROOT, "runs", "parity3", "isaac")


def dome(tag="dome_il2"):
    import imageio.v2 as iio
    d = os.path.join(RUN, tag); J = json.load(open(os.path.join(d, "dome_results.json")))
    Rs = {k: np.array(v) for k, v in J["cube_cams"]["R_cols_right_up_back"].items()}
    M = J["map"]["markers_uv"]; col = {"red": 0, "green": 1, "blue": 2}
    out = {"kit_version": J.get("kit_version")}
    for var in [v for v in ("none", "rotZ90", "rotX90", "none_repeat") if os.path.exists(os.path.join(d, f"{v}_px.png"))]:
        out[var] = {}
        for k, ci in col.items():
            acc = np.zeros(3); n = 0
            for fn, R in Rs.items():
                a = iio.imread(os.path.join(d, f"{var}_{fn}.png"))[..., :3].astype(float)
                o = [i for i in range(3) if i != ci]
                m = (a[..., ci] > 128) & (a[..., o[0]] < 0.5 * a[..., ci]) & (a[..., o[1]] < 0.5 * a[..., ci])
                ys, xs = np.nonzero(m)
                X = 2 * (xs + .5) / 256 - 1; Y = 1 - 2 * (ys + .5) / 256
                v = (R @ np.stack([X, Y, -np.ones_like(X)])).T
                w = 1 / (1 + X ** 2 + Y ** 2) ** 1.5           # solid angle of a cube-face pixel
                acc += (v / np.linalg.norm(v, axis=1, keepdims=True) * w[:, None]).sum(0); n += len(xs)
            if n == 0:
                out[var][k] = None; continue
            v = acc / np.linalg.norm(acc); u, vv = M[k]
            yaw = 90.0 if var == "rotZ90" else 0.0
            # our adopted convention: atan2(x, -y) = (0.5 - u) 360 + yaw
            ours = ((0.5 - u) * 360 + yaw + 180) % 360 - 180
            out[var][k] = {"pixels": n, "dir": np.round(v, 4).tolist(), "elev_deg": round(math.degrees(math.asin(v[2])), 2),
                           "expected_elev_deg_pole_z": 90 - 180 * vv, "az_atan2_y_x_deg": round(math.degrees(math.atan2(v[1], v[0])), 2),
                           "atan2_x_negy_deg": round(math.degrees(math.atan2(v[0], -v[1])), 2), "ours_atan2_x_negy_deg": round(ours, 2),
                           "kit_minus_ours_deg": round(((math.degrees(math.atan2(v[0], -v[1])) - ours + 180) % 360) - 180, 2)}
    print(json.dumps(out, indent=1)); return out


def pen(tag=None):
    if tag is None:
        for t in ("penetration_il2", "penetration_il2_cpu", "penetration_il3"):
            print(f"== {t}"); pen(t)
        return
    d = os.path.join(RUN, tag); z = np.load(os.path.join(d, "C_drop_penetration.npz")); meta = json.load(open(os.path.join(d, "meta.json")))
    dt = meta["physics_dt"]; st, b, s, f = z["tensor_step"], z["tensor_body"], z["tensor_separation"], z["tensor_normal_force"]
    names = z["body_names"]
    print("bodies in contact:", {str(names[i]): int((b == i).sum()) for i in np.unique(b)})
    for i in np.unique(b):
        m = b == i; j = np.argmin(s[m])
        late = m & (st > meta["n_phys_steps"] - 100)
        print(f"  {names[i]}: first contact step {st[m].min()}, peak penetration {-s[m][j] * 1e3:.3f} mm at step {st[m][j]} (t = {st[m][j] * dt:.3f} s), "
              f"last-0.5 s min sep {s[late].min() * 1e3 if late.any() else float('nan'):+.4f} mm")
    steps = np.arange(1, meta["n_phys_steps"] + 1)
    mn = np.full(len(steps), np.nan); mnl = np.full(len(steps), np.nan); nc = np.zeros(len(steps), int); fz = np.zeros(len(steps))
    for i, k in enumerate(steps):
        m = st == k
        if m.any():
            mn[i] = s[m].min(); nc[i] = m.sum(); fz[i] = f[m].sum()
            ml = m & (f > 0)
            if ml.any(): mnl[i] = s[ml].min()
    i0 = int(np.nanargmin(mn))
    print(f"contacts first at step {steps[np.argmax(nc > 0)]} (t = {steps[np.argmax(nc > 0)] * dt:.3f} s); peak penetration {-mn[i0] * 1e3:.3f} mm at step {steps[i0]} (t = {steps[i0] * dt:.3f} s)")
    for a, bnd in [(0, 60), (60, 100), (100, 200), (200, 400), (400, 601)]:
        sel = (steps > a) & (steps <= bnd) & np.isfinite(mn)
        if sel.any(): print(f"  steps {a+1}-{bnd}: min sep {np.nanmin(mn[sel]) * 1e3:+.4f} mm, mean of per-step minima {np.nanmean(mn[sel]) * 1e3:+.4f} mm, contacts/step {nc[sel].mean():.1f}, sum normal force {fz[sel].mean():.1f} N")
    print("profile around the peak (step, t, min sep mm, n contacts, sum force N):")
    for i in range(max(0, i0 - 4), min(len(steps), i0 + 12)):
        print(f"  {steps[i]:4d} {steps[i] * dt:.3f} {mn[i] * 1e3:+.4f} {nc[i]:3d} {fz[i]:9.1f}")
    print("final root z (control step 150):", meta["final_root_z_ctrl"], "report rows:", meta["report_rows"], "offsets:", json.dumps(meta["collider_offsets"]))
    if meta.get("report_rows"):
        rs, rsep = z["report_step"], z["report_separation"]
        same = all(abs(np.min(rsep[rs == k]) - np.min(s[st == k])) < 1e-7 for k in np.unique(st))
        print("full contact report per-step minimum separation identical to the tensor API's (1e-7 m):", same)
    return dict(steps=steps, min_sep=mn)


def fourbar():
    sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "closed_loops"))
    import mechanisms as M
    ref = np.load(os.path.join(ROOT, "runs", "closed_loops", "ref", "fourbar_none.npz"))
    tr, ar = ref["t"], ref["ang"]
    E0 = M.energy("fourbar", ar[:5], tr[1] - tr[0])[0]
    rows = []
    for d in sorted(glob.glob(os.path.join(RUN, "fourbar_il*", "*"))):
        if not os.path.exists(os.path.join(d, "fourbar_none.npz")): continue
        z = np.load(os.path.join(d, "fourbar_none.npz")); meta = json.load(open(os.path.join(d, "meta.json")))
        t, a = z["t"], z["ang"]
        ari = np.stack([np.interp(t, tr, np.unwrap(ar[:, j])) for j in range(3)], 1)
        err = np.abs(np.unwrap(a, axis=0) - ari).max(1)
        E = M.energy("fourbar", a, meta["dt"])
        e_at = lambda T: float(err[np.searchsorted(t, T - 1e-9)])
        cm = np.maximum.accumulate(err); c_at = lambda T: float(cm[np.searchsorted(t, T - 1e-9)])   # c_summary.md's metric (max over [0, T])
        rows.append(dict(run=os.path.relpath(d, RUN), dt=meta["dt"], device=meta["device"], solver=meta["solver"], iters=f'{meta["pos_iters"]}/{meta["vel_iters"]}',
                         cummax_err_1s=c_at(1.0), cummax_err_2s=c_at(2.0), cummax_err_5s=c_at(5.0), err_at_5s=e_at(5.0), dE_5s_pct=float((E[-1] - E0) / abs(E0) * 100) if E0 else None,
                         E_end_minus_start_J=float(E[-1] - E[0]), gap_max_mm=meta["closure_gap_max_m"] * 1e3, gap_mean_mm=meta["closure_gap_mean_m"] * 1e3,
                         wall_s=meta["wall_s"]))
    E_ref = M.energy("fourbar", ar, tr[1] - tr[0])
    print(f"reference energy start {E_ref[0]:.6f} J end {E_ref[-1]:.6f} J")
    for r in rows: print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
    return rows


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["dome", "pen", "fourbar"]):
        globals()[w]()
