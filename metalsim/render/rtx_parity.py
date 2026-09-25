"""Rendering-model parity against Isaac Sim's RTX frames on identical states (kinematic replay).

The protocol replay (``metalsim.parity.record_g1``) re-simulates the G1 and so compares shading only
on frames where the two physics engines happen to agree. This tool removes physics from the
comparison: every Isaac frame is re-rendered by tier 2 from Isaac's own recorded root pose and joint
angles (``{A_hold,B_random,C_drop}.npz`` next to the frames), so all frames compare the same pose,
camera, lights and asset, and the difference is the renderer's material / light / tone model alone.

    python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/rt --out runs/render_parity/rt_legacy --preset legacy

Reports (``summary.json``): whole-frame PSNR / SSIM / LPIPS / FLIP (``metalsim.parity.compare.image_metrics``)
and robot-only metrics (union silhouette, grey outside, raw and brightness-matched), plus the sky
pixel value and render time per frame. Presets are the measured steps of the RTX-parity work
(docs/research/rendering_vs_rtx_2026-09-25.md).
"""
from __future__ import annotations

import argparse, glob, json, os, time

import imageio.v2 as iio
import mujoco
import numpy as np


def build_scene(meta):
    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.parity.record_g1 import hero_spec
    m, info = build_g1_model("flat", visuals=True, physics_dt=0.0025)
    spec = hero_spec(info["spec"], meta)
    return spec.compile()


def isaac_states(isaac_dir, m, joint_names):
    """qpos per (protocol, step) from Isaac's recorded root pose and joint angles (env 0)."""
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    adr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names])
    out = {}
    for tag in ("A_hold", "B_random", "C_drop"):
        p = os.path.join(isaac_dir, f"{tag}.npz")
        if not os.path.exists(p):
            continue
        A = np.load(p)
        T = len(A["joint_pos"]); q = np.tile(m.qpos0, (T, 1))
        q[:, 0:3] = A["root_pos"][:, 0]; q[:, 3:7] = A["root_quat"][:, 0]; q[:, adr] = A["joint_pos"][:, 0]
        out[tag] = q
    assert len(ours) == len(joint_names)
    return out


# robot materials are OmniPBR (Isaac's G1 asset), the ground a UsdPreviewSurface (the recording's grey plane)
MATERIALS = {None: "omnipbr", "ground": "usd_preview"}
MATERIALS_V1 = {None: "omnipbr_curve", "ground": "usd_preview_storm"}
MATERIALS_FV = {None: "omnipbr_fv", "ground": "usd_preview_fv"}
KIT_EXPOSURE = 1.0 / (50.0 * 5.0 ** 2)      # Kit's default camera: ISO 100, shutter 1/50, f/5 -> 8.0e-4
CAL_EXPOSURE = 8.725e-4                     # least-squares fit of sRGB(ACES(k L)) to RTX's sky + sunlit ground (§1.3 of the note)


ISAAC_CLIP_FAR = 100.0                      # isaac_side/record_g1.py: PinholeCameraCfg clipping_range=(0.05, 100.0)
RTX_DEFAULT_PRESET = "oidn_cal_fvg"            # the most faithful measured configuration (see the note, §3)


def preset_kwargs(name, meta):
    """The cumulative steps of the RTX-parity work (each measured on the same frames)."""
    if name == "legacy":            # the renderer as it was: MuJoCo-unit lights, headlight, linear clamp
        return {"center_sample": True}                                # (annotators at pixel centres for the masks)
    from metalsim.render.tier2 import usd_scene_kwargs
    mats = MATERIALS
    if name.endswith("_v1"):          # archived first BRDF variant: Storm's UsdPreviewSurface, curve-weighted OmniPBR
        mats = MATERIALS_V1; name = name[:-3]
    elif name.endswith("_fv"):        # base weighted by the Fresnel curve at N.V (ground and robot)
        mats = MATERIALS_FV; name = name[:-3]
    elif name.endswith("_fvg"):       # ... ground only
        mats = {None: "omnipbr", "ground": "usd_preview_fv"}; name = name[:-4]
    kw = {"material_model": mats, "center_sample": True}
    if name == "brdf":              # step 1: OmniPBR / UsdPreviewSurface BRDFs, lights and display as before
        return kw
    kw.update(usd_scene_kwargs(meta, tonemap=False))
    kw["clip_far"] = ISAAC_CLIP_FAR
    if name == "lights":            # step 2: + USD light units (sun E = pi*3000*cos, 0.53 deg disk, dome 400 nits, no
        kw["exposure"] = 1.0 / meta["lights"]["dome"]["intensity"]   # headlight), shown at the old absolute level (linear clamp)
        return kw
    kw.update(tonemap="rtx", exposure=KIT_EXPOSURE)
    if name.endswith("_cal"):          # one global constant fitted to RTX's sky and lit ground (not per image)
        kw["exposure"] = CAL_EXPOSURE; name = name[:-4]
    if name == "tonemap":              # step 3: + Kit's default exposure, ACES, sRGB
        return kw
    if name == "oidn":                 # step 4: + Open Image Denoise (Metal device)
        kw.update(denoise="oidn"); return kw
    if name == "atrous":               # step 4b: + in-command-buffer a-trous filter
        kw.update(denoise="atrous"); return kw
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="legacy"); ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--passes", type=int, default=8); ap.add_argument("--bounces", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="use every k-th Isaac frame")
    ap.add_argument("--no_metrics", action="store_true"); ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--save_hdr", action="store_true", help="also save the linear radiance (float16, 3.5 MB per frame)")
    a = ap.parse_args()
    import warp as wp; wp.config.quiet = True
    from metalsim.render import tier2
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.isaac, "meta.json")))
    m = build_scene(meta)
    names = [str(s) for s in np.load(os.path.join(a.isaac, "action_sequence_B.npz"))["joint_names"]]
    states = isaac_states(a.isaac, m, names)
    frames = sorted(glob.glob(os.path.join(a.isaac, "*_rgb.png")))[::a.every]
    if a.limit: frames = frames[:a.limit]
    cam = meta["camera"]
    kw = preset_kwargs(a.preset, meta)
    if a.exposure: kw["exposure"] = a.exposure
    rend = tier2.Tier2Renderer(m, a.batch, width=cam["width"], height=cam["height"], camera="hero", spp=a.spp,
                               max_bounces=a.bounces, **kw)
    datas = [mujoco.MjData(m) for _ in range(a.batch)]
    times = []
    for i0 in range(0, len(frames), a.batch):
        chunk = frames[i0:i0 + a.batch]
        for k in range(a.batch):
            f = chunk[min(k, len(chunk) - 1)]
            tag, t = os.path.basename(f).rsplit("_", 2)[0], int(os.path.basename(f).split("_")[2])
            datas[k].qpos[:] = states[tag][t]; mujoco.mj_forward(m, datas[k])
        t0 = time.perf_counter()
        out = rend.render_host(datas, passes=a.passes)
        times.append((time.perf_counter() - t0) / a.batch)
        for k, f in enumerate(chunk):
            base = os.path.basename(f).replace("_rgb.png", "")
            iio.imwrite(os.path.join(a.out, base + "_rgb.png"), out["rgb"][k])
            np.save(os.path.join(a.out, base + "_depth.npy"), out["depth"][k])
            if a.save_hdr:
                np.save(os.path.join(a.out, base + "_hdr.npy"), out["hdr"][k].astype(np.float16))
        print(f"[rtx_parity] {i0 + len(chunk)}/{len(frames)} {times[-1] * 1e3:.0f} ms/frame", flush=True)
    summary = {"preset": a.preset, "kwargs": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else repr(v)) for k, v in kw.items()},
               "spp": a.spp, "passes": a.passes, "bounces": a.bounces, "frames": len(frames),
               "render_ms_per_frame_host_path": float(np.median(times[1:] if len(times) > 1 else times) * 1e3)}
    if not a.no_metrics:
        summary.update(evaluate(a.isaac, a.out, [os.path.basename(f).replace("_rgb.png", "") for f in frames]))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=1))


def evaluate(isaac_dir, ours_dir, bases, lp="auto"):
    from metalsim.parity.compare import image_metrics
    if lp == "auto":
        try:
            import lpips; lp = lpips.LPIPS(net="alex", verbose=False)
        except Exception:
            lp = None
    rows = []
    for b in bases:
        ia = iio.imread(os.path.join(isaac_dir, b + "_rgb.png"))[..., :3]; ib = iio.imread(os.path.join(ours_dir, b + "_rgb.png"))[..., :3]
        da = np.load(os.path.join(isaac_dir, b + "_depth.npy")); db = np.load(os.path.join(ours_dir, b + "_depth.npy"))
        with np.errstate(invalid="ignore"):
            r = image_metrics(ia, ib, da, db, lp)
        sky = ~np.isfinite(da) | (da > 90)
        r["sky_isaac"] = ia[sky].mean(0).tolist() if sky.any() else None
        r["sky_ours"] = ib[sky].mean(0).tolist() if sky.any() else None
        rows.append({"frame": b, **r})
    keys = ["psnr_db", "ssim", "flip"] + (["lpips_alex"] if lp is not None else [])
    ck = ["psnr_db", "ssim", "flip"] + (["lpips_alex_crop"] if lp is not None else [])
    agree = [r for r in rows if r["robot_crop"] and (r["silhouette_iou"] or 0) > 0.7]
    mean = lambda rs, f: float(np.mean([f(r) for r in rs])) if rs else None
    return {"whole_frame_raw": {k: mean(rows, lambda r: r["raw"][k]) for k in keys},
            "robot_raw": {k: mean(agree, lambda r: r["robot_crop"]["raw"][k]) for k in ck},
            "robot_brightness_matched": {k: mean(agree, lambda r: r["robot_crop"]["brightness_matched"][k]) for k in ck},
            "robot_gain_mean": mean(agree, lambda r: r["robot_crop"]["gain"]),
            "whole_frame_gain_mean": mean(rows, lambda r: r["brightness_gain_applied"]),
            "robot_frames": len(agree), "silhouette_iou_mean": mean(rows, lambda r: r["silhouette_iou"] or 0),
            "robot_depth_rmse_m": mean([r for r in rows if r["robot_depth_rmse_m"] is not None], lambda r: r["robot_depth_rmse_m"]),
            "sky_isaac": np.mean([r["sky_isaac"] for r in rows if r["sky_isaac"]], 0).round(1).tolist(),
            "sky_ours": np.mean([r["sky_ours"] for r in rows if r["sky_ours"]], 0).round(1).tolist(),
            "per_frame": rows}


if __name__ == "__main__":
    main()


def gallery(isaac_rt, isaac_pt, ours_new, ours_old, out_dir, frames=("A_hold_0025", "C_drop_0050"), crop_frame="A_hold_0025"):
    """docs/gallery composites: full frames [Isaac RTX real-time | MetalSim tier 2 (RTX-parity mode) | tier 2 before],
    and a robot crop [RTX real-time | RTX path traced | tier 2 now | tier 2 before]."""
    from metalsim.parity.compare import robot_mask
    os.makedirs(out_dir, exist_ok=True)
    rd = lambda d, b: iio.imread(os.path.join(d, b + "_rgb.png"))[..., :3]
    for b in frames:
        iio.imwrite(os.path.join(out_dir, f"fidelity_{b}.png"), np.concatenate([rd(isaac_rt, b), rd(ours_new, b), rd(ours_old, b)], 1))
    with np.errstate(invalid="ignore"):
        r, _ = robot_mask(np.load(os.path.join(isaac_rt, crop_frame + "_depth.npy")))
    ys, xs = np.nonzero(r); cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = max(0, min(ys.min() - 10, 576 - 250)); x0 = max(0, min(cx - 140, 1024 - 280))
    tiles = [rd(d, crop_frame)[y0:y0 + 250, x0:x0 + 280] for d in (isaac_rt, isaac_pt, ours_new, ours_old)]
    iio.imwrite(os.path.join(out_dir, f"fidelity_crop_{crop_frame}.png"), np.concatenate(tiles, 1))
