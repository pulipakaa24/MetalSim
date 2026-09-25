"""Fidelity metrics between an Isaac Lab recording and the MetalSim replay of the same protocol.

Physics (same asset, same initial state, same open-loop actions):
  per-joint RMSE over time, root height / orientation error, time to divergence (first step where
  the max joint error exceeds 0.1 rad), foot contact-force statistics, peak joint speed.
Rendering (same camera, same lights, same state per frame; Isaac RTX vs MetalSim tier 2 / tier 0):
  PSNR, SSIM, LPIPS (AlexNet), FLIP (from metalsim.bench.render_fidelity), on raw images and after
  matching the mean brightness (the two engines' light units differ), silhouette IoU from depth, depth RMSE on
  the robot silhouette and on the ground plane (sky / far plane masked, z-depth in both engines), and
  the same image metrics on the robot crop alone (outside the union silhouette set to grey).
Writes a JSON report and side-by-side composites (Isaac | MetalSim tier 2 | MetalSim tier 0).

    python -m metalsim.parity.compare --isaac runs/parity/isaac/rt --metalsim runs/parity/metalsim --out runs/parity/report
"""
import argparse, glob, json, os
import numpy as np
import imageio.v2 as iio


def quat_angle(q1, q2):
    d = np.abs(np.sum(q1 * q2, axis=-1)).clip(0, 1)
    return 2 * np.arccos(d)


def physics_metrics(isaac, ours, tag):
    A = np.load(os.path.join(isaac, f"{tag}.npz")); B = np.load(os.path.join(ours, f"{tag}.npz"))
    T = min(len(A["joint_pos"]), len(B["joint_pos"]))
    jp_a, jp_b = A["joint_pos"][:T, 0], B["joint_pos"][:T, 0]          # env 0 (identical protocol in every env)
    err = np.abs(jp_a - jp_b)
    rmse_t = np.sqrt((err ** 2).mean(1))
    div = int(np.argmax(err.max(1) > 0.1)) if (err.max(1) > 0.1).any() else T
    zr_a, zr_b = A["root_pos"][:T, 0, 2], B["root_pos"][:T, 0, 2]
    ang = quat_angle(A["root_quat"][:T, 0], B["root_quat"][:T, 0])
    cf_a = np.linalg.norm(A["contact"][:T, 0], axis=-1); cf_b = np.linalg.norm(B["contact"][:T, 0], axis=-1)
    return {"steps": T,
            "joint_rmse_rad": {"t0.5s": float(rmse_t[min(24, T - 1)]), "t1s": float(rmse_t[min(49, T - 1)]), "t2s": float(rmse_t[min(99, T - 1)]), "end": float(rmse_t[-1]), "max": float(rmse_t.max())},
            "divergence_step_0.1rad": div, "divergence_time_s": div * 0.02,
            "root_height": {"isaac_end": float(zr_a[-1]), "metalsim_end": float(zr_b[-1]), "rmse": float(np.sqrt(((zr_a - zr_b) ** 2).mean()))},
            "root_orientation_err_rad": {"mean": float(ang.mean()), "max": float(ang.max())},
            "peak_joint_speed_rad_s": {"isaac": float(np.abs(A["joint_vel"][:T, 0]).max()), "metalsim": float(np.abs(B["joint_vel"][:T, 0]).max())},
            "contact_force_N": {"isaac_mean_total": float(cf_a.sum(1).mean()), "metalsim_mean_total": float(cf_b.sum(1).mean()),
                                 "isaac_peak": float(cf_a.max()), "metalsim_peak": float(cf_b.max())},
            "torque_rms_Nm": {"isaac": float(np.sqrt((A["torque"][:T, 0] ** 2).mean())), "metalsim": float(np.sqrt((B["torque"][:T, 0] ** 2).mean()))},
            **_contact_limit_metrics(A, B, ours, T),
            "contact_momentum": _momentum_metrics(A, B, ours, T, tag),
            "_note_contact_force_N": "sampled peaks (isaac_peak / metalsim_peak) depend on each sensor's reporting window "
                                     "(Isaac: last 5 ms PhysX step; ours: last 2.5 ms substep); rank on contact_momentum"}


def _momentum_metrics(A, B, ours, T, tag):
    """Impulse per event and largest 20 ms mean contact force from each engine's recorded states (momentum balance,
    metalsim.parity.momentum), independent of the sensors' reporting windows."""
    try:
        from metalsim.parity import momentum
        mp = os.path.join(ours, "meta.json")
        joints = json.load(open(mp))["isaac_joints"] if os.path.exists(mp) else None
        if joints is None:
            return None
        Ai = {k: A[k][:T] for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_b", "root_ang_vel_b")}
        Bi = {k: B[k][:T] for k in Ai}
        Ji = momentum.contact_impulse(Ai, joints, root_vel_is_com=True)      # Isaac Lab 2.3.2: root COM velocity
        Jo = momentum.contact_impulse(Bi, joints, root_vel_is_com=False)     # MuJoCo free joint: frame origin
        return momentum.event_metrics(Ji, Jo, tag)
    except Exception as ex:                                                 # keep the rest of the report
        return {"error": repr(ex)}


def _contact_limit_metrics(A, B, ours, T):
    """Penetration (MetalSim only: max over substeps, recorded per control step; Isaac's PhysX recording has
    no contact distances) and joint-limit excursion: both engines from joint_pos at control steps against
    the asset's limits (MetalSim's meta.json), plus MetalSim's max over substeps when recorded."""
    out = {}
    if "penetration" in B.files:
        p = B["penetration"][:T, 0]
        out["penetration_m"] = {"metalsim_max": float(p.max()), "metalsim_mean_when_in_contact": float(p[p > 0].mean()) if (p > 0).any() else 0.0,
                                "isaac": None}
    mp = os.path.join(ours, "meta.json")
    rng = json.load(open(mp)).get("joint_range_isaac_order") if os.path.exists(mp) else None
    if rng is not None:
        lim = np.array([r is not None for r in rng]); lo = np.array([r[0] if r else -np.inf for r in rng]); hi = np.array([r[1] if r else np.inf for r in rng])
        def exc(jp):
            e = np.maximum(lo - jp, jp - hi)[:, lim]
            return np.maximum(e, 0)
        ea, eb = exc(A["joint_pos"][:T, 0]), exc(B["joint_pos"][:T, 0])
        out["limit_excursion_rad"] = {"isaac_max_ctrl": float(ea.max()), "metalsim_max_ctrl": float(eb.max()),
                                      "isaac_steps_over_0.01": int((ea.max(1) > 0.01).sum()), "metalsim_steps_over_0.01": int((eb.max(1) > 0.01).sum())}
        if "limit_excursion" in B.files:
            out["limit_excursion_rad"]["metalsim_max_substep"] = float(B["limit_excursion"][:T, 0].max())
    return out


def robot_mask(d, thresh=0.05):
    """Pixels off the ground plane (robot), and pixels on it, from a z-depth image: fit 1/z = a u + b v + c
    on the outer lower image (no robot there in the protocol camera), residual > thresh m = robot.
    thresh 5 cm: MetalSim tier-2 depth carries ~4 mm RMS (1.5 cm max) noise on the plane; Isaac's is exact."""
    d = d.astype(np.float64); H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    valid = np.isfinite(d) & (d > 0) & (d < 50.0)
    sel = valid & (v > 0.55 * H) & ((u < 0.3 * W) | (u > 0.7 * W))
    A = np.stack([u[sel], v[sel], np.ones(sel.sum())], 1); c = np.linalg.lstsq(A, 1.0 / d[sel], rcond=None)[0]
    den = c[0] * u + c[1] * v + c[2]; plane = np.where(den > 1e-9, 1.0 / np.maximum(den, 1e-9), np.inf)   # above the horizon -> inf
    robot = valid & (plane - d > thresh)
    ground = valid & (np.abs(plane - d) <= thresh)
    return robot, ground


def masked_stats(x, y, m, lp=None):
    """PSNR, SSIM and FLIP averaged over the mask pixels only (the grey fill outside the silhouette would
    otherwise inflate them); LPIPS is over the whole crop (no masked variant) and says so."""
    from skimage.metrics import structural_similarity
    from metalsim.bench.render_fidelity import flip
    mse = float(((x - y)[m] ** 2).mean()); out = {"psnr_db": 10 * np.log10(1.0 / max(mse, 1e-10))}
    _, smap = structural_similarity(x, y, channel_axis=2, data_range=1.0, full=True); out["ssim"] = float(smap[m].mean())
    fmap = np.asarray(flip((x * 255).astype(np.uint8), (y * 255).astype(np.uint8))); out["flip"] = float(fmap[m].mean() if fmap.shape == m.shape else fmap.mean())
    if lp is not None:
        import torch
        t = lambda z: torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1))[None] * 2 - 1).float()
        out["lpips_alex_crop"] = float(lp(t(x), t(y)).item())
    return out


def image_metrics(a, b, da, db, lp=None):
    from skimage.metrics import structural_similarity
    from metalsim.bench.render_fidelity import flip
    a = a.astype(np.float32) / 255; b = b.astype(np.float32) / 255
    def stats(x, y):
        mse = float(((x - y) ** 2).mean()); psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        ssim = float(structural_similarity(x, y, channel_axis=2, data_range=1.0))
        f = float(np.mean(flip((x * 255).astype(np.uint8), (y * 255).astype(np.uint8))))
        out = {"psnr_db": psnr, "ssim": ssim, "flip": f}
        if lp is not None:
            import torch
            t = lambda z: torch.from_numpy(z.transpose(2, 0, 1)[None] * 2 - 1).float()
            out["lpips_alex"] = float(lp(t(x), t(y)).item())
        return out
    raw = stats(a, b)
    gain = a.mean() / max(b.mean(), 1e-6)
    norm = stats(a, np.clip(b * gain, 0, 1))
    # robot-only metrics: everything outside the union silhouette set to mid-grey, cropped to its bounding
    # box (+16 px), brightness matched on the robot pixels. Independent of the ground / sky in each scene.
    robot = None
    # Depth. Both engines write z-depth (a ground plane fits 1/z affine in pixel coordinates to < 1 mm in
    # Isaac, < 5 mm in MetalSim); Isaac writes inf for the sky and clips at the 100 m far plane, MetalSim
    # writes 0 for a miss. The robot silhouette is every pixel that departs from the fitted ground plane,
    # so the comparison covers (i) silhouette IoU, (ii) depth RMSE on the robot (both silhouettes),
    # (iii) depth RMSE on the ground both engines see.
    sil_a, ground_a = robot_mask(da); sil_b, ground_b = robot_mask(db)
    both = sil_a & sil_b; inter = float(both.sum()); union = float((sil_a | sil_b).sum())
    g = ground_a & ground_b
    uni = sil_a | sil_b
    if uni.sum() > 100:
        ys, xs = np.nonzero(uni); y0, y1 = max(0, ys.min() - 16), min(a.shape[0], ys.max() + 17); x0, x1 = max(0, xs.min() - 16), min(a.shape[1], xs.max() + 17)
        m = uni[y0:y1, x0:x1]; ca = np.where(m[..., None], a[y0:y1, x0:x1], 0.5); cb = np.where(m[..., None], b[y0:y1, x0:x1], 0.5)
        rg = a[uni].mean() / max(b[uni].mean(), 1e-6)
        if min(ca.shape[:2]) >= 32:
            robot = {"raw": masked_stats(ca, cb, m, lp), "brightness_matched": masked_stats(ca, np.clip(np.where(m[..., None], cb * rg, 0.5), 0, 1), m, lp),
                     "gain": float(rg), "crop_hw": [int(y1 - y0), int(x1 - x0)], "robot_fraction_of_crop": float(m.mean())}
    return {"raw": raw, "brightness_matched": norm, "brightness_gain_applied": float(gain),
            "robot_crop": robot,
            "silhouette_iou": inter / union if union else None,
            "silhouette_px": {"isaac": int(sil_a.sum()), "metalsim": int(sil_b.sum())},
            "robot_depth_rmse_m": float(np.sqrt(((da - db)[both] ** 2).mean())) if both.any() else None,
            "robot_depth_agree_1cm": float((np.abs(da - db)[both] < 0.01).mean()) if both.any() else None,
            "ground_depth_rmse_m": float(np.sqrt(((da - db)[g] ** 2).mean())) if g.any() else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--isaac", required=True); ap.add_argument("--metalsim", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_render", action="store_true", help="physics rows only")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    report = {"physics": {}, "render": {}}
    for tag in ("A_hold", "B_random", "C_drop"):
        if os.path.exists(os.path.join(a.isaac, f"{tag}.npz")) and os.path.exists(os.path.join(a.metalsim, f"{tag}.npz")):
            report["physics"][tag] = physics_metrics(a.isaac, a.metalsim, tag)
    if a.no_render:
        json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
        print(json.dumps(report, indent=1))
        return
    try:
        import lpips, torch
        lp = lpips.LPIPS(net="alex", verbose=False)
    except Exception:
        lp = None
    frames = sorted(glob.glob(os.path.join(a.isaac, "*_rgb.png")))
    per_frame = {"tier2": [], "tier0": []}
    for f in frames:
        base = os.path.basename(f).replace("_rgb.png", "")
        g2 = os.path.join(a.metalsim, base + "_rgb.png"); g0 = os.path.join(a.metalsim, base + "_rgb_tier0.png")
        if not os.path.exists(g2): continue
        ia = iio.imread(f)[..., :3]; i2 = iio.imread(g2)[..., :3]; da = np.load(f.replace("_rgb.png", "_depth.npy")); d2 = np.load(g2.replace("_rgb.png", "_depth.npy"))
        if ia.shape != i2.shape: continue
        per_frame["tier2"].append({"frame": base, **image_metrics(ia, i2, da, d2, lp)})
        if os.path.exists(g0):
            per_frame["tier0"].append({"frame": base, **image_metrics(ia, iio.imread(g0)[..., :3], da, d2, lp)})
        strip = np.concatenate([ia, i2] + ([iio.imread(g0)[..., :3]] if os.path.exists(g0) else []), axis=1)
        iio.imwrite(os.path.join(a.out, base + "_side_by_side.png"), strip)
    for tier, rows in per_frame.items():
        if rows:
            keys = ["psnr_db", "ssim", "flip"] + (["lpips_alex"] if lp else [])
            report["render"][tier] = {"frames": len(rows),
                                      "raw_mean": {k: float(np.mean([r["raw"][k] for r in rows])) for k in keys},
                                      "brightness_matched_mean": {k: float(np.mean([r["brightness_matched"][k] for r in rows])) for k in keys},
                                      **{k + "_mean": float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ("silhouette_iou", "robot_depth_rmse_m", "robot_depth_agree_1cm", "ground_depth_rmse_m")},
                                      "robot_crop_raw_mean": {k: float(np.mean([r["robot_crop"]["raw"][k] for r in rows if r["robot_crop"]])) for k in ["psnr_db", "ssim", "flip"] + (["lpips_alex_crop"] if lp else [])},
                                      "robot_crop_brightness_matched_mean": {k: float(np.mean([r["robot_crop"]["brightness_matched"][k] for r in rows if r["robot_crop"]])) for k in ["psnr_db", "ssim", "flip"] + (["lpips_alex_crop"] if lp else [])},
                                      "robot_fraction_of_crop_mean": float(np.mean([r["robot_crop"]["robot_fraction_of_crop"] for r in rows if r["robot_crop"]])),
                                      "per_frame": rows}
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: (v if k == "physics" else {t: {kk: vv for kk, vv in r.items() if kk != "per_frame"} for t, r in v.items()}) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
