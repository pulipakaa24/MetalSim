#!/usr/bin/env python3
"""Splat renderer measurements on the public reference scene (SPLAT.md §4, §5; data: scripts/fetch_splat_scenes.sh).

    python scripts/splat_bench.py verify [--views 8] [--out runs/splat/verify.json]
        Metal vs the independent float64 reference (render/splat/reference.py) at robot-camera views (640x360, HFOV 88.9,
        0.83 m, 22 deg down) placed on the room-polygon grid of InteriorGS 0204_840158: PSNR, SSIM, pixel level
        differences, median-depth and alpha errors. Correctness: run on any Mac.
    python scripts/splat_bench.py bench [--ns 1,8,32,64] [--iters 20] [--out runs/splat/bench.json]
        Throughput: SplatRenderer.bench (frames encoded back to back, one sync, no output copy) best of 3, stage times
        from profile_frame. Timing: only through scripts/gpu_run.sh NAME timing, on the 16-inch.
    python scripts/splat_bench.py options [--out runs/splat/options.json]
        The archived options of SPLAT.md §6 on the same scene: PSNR vs the default renderer over 16 views and GPU ms per
        frame for those 16 cameras (depth key 16 bit, no subtile masks, blend 2 px, SH 0 / 1, antialiased forced on,
        crop none / tight). Timing: through gpu_run.sh timing, on the 16-inch.
    python scripts/splat_bench.py md VERIFY.json BENCH.json [--options OPTIONS.json] --update metalsim/render/SPLAT.md
        Writes the tables between the <!-- splat-bench:begin/end --> (and splat-options) markers of SPLAT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metalsim.render.splat.views import INTERIORGS_REF, room_views  # noqa: E402

W, H, HFOV = 640, 360, 88.9
SCENE = "InteriorGS 0204_840158"


def _scene(cloud=False):
    from metalsim.render.splat import load, prepare
    ply = os.path.join(INTERIORGS_REF, "3dgs_compressed.ply")
    if not os.path.exists(ply):
        sys.exit(f"{ply} absent: run scripts/fetch_splat_scenes.sh interiorgs")
    t0 = time.perf_counter(); c = load(ply); load_s = time.perf_counter() - t0
    s = prepare(c, log=lambda *a: None)
    pos, mats = room_views(os.path.join(INTERIORGS_REF, "structure.json"))
    info = dict(scene=SCENE, n_splats=s.n, load_s=round(load_s, 2), prep_s=s.info["prep_s"], n_views=len(pos))
    return (s, pos, mats, info, c) if cloud else (s, pos, mats, info)


def _machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0) / 2 ** 30
    return f"{chip}, {mem:.0f} GB, macOS {platform.mac_ver()[0]}"


def _ssim(a, b):
    """Mean SSIM over 8x8 windows (grey, data range 255), Wang et al. 2004 constants."""
    a = a.astype(np.float64).mean(-1); b = b.astype(np.float64).mean(-1)
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    A = a[:h, :w].reshape(h // 8, 8, w // 8, 8); B = b[:h, :w].reshape(h // 8, 8, w // 8, 8)
    ma, mb = A.mean((1, 3)), B.mean((1, 3))
    va, vb = A.var((1, 3)), B.var((1, 3))
    cov = ((A - ma[:, None, :, None]) * (B - mb[:, None, :, None])).mean((1, 3))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def verify(a):
    from metalsim.render.splat import reference as ref
    from metalsim.render.splat.renderer import SplatRenderer
    s, pos, mats, info = _scene()
    pick = np.linspace(0, len(pos) - 1, a.views).astype(int)
    r = SplatRenderer(s, len(pick), width=W, height=H, hfov_deg=HFOV, outputs=("rgb", "depth", "depth_expected", "alpha"))
    out = r.render_host(pos[pick], cam_mat=mats[pick])
    rows = []
    for k, i in enumerate(pick):
        t0 = time.perf_counter()
        rr = ref.render(s, pos[i], mats[i], r.intrinsics, W, H)
        ref_u8 = (np.clip(rr["rgb"], 0, 1) * 255 + 0.5).astype(np.uint8)
        lv = np.abs(out["rgb"][k].astype(int) - ref_u8.astype(int)).max(-1)
        mse = np.mean((out["rgb"][k].astype(np.float64) - np.clip(rr["rgb"], 0, 1) * 255) ** 2)
        both = (out["depth"][k] > 0) & (rr["depth"] > 0)
        dz = np.abs(out["depth"][k] - rr["depth"])[both]
        de = (out["depth_expected"][k] > 0) & (rr["depth_expected"] > 0)
        rel = (np.abs(out["depth_expected"][k] - rr["depth_expected"]) / np.maximum(rr["depth_expected"], 1e-6))[de]
        rows.append(dict(view=int(i), psnr=round(10 * np.log10(255 ** 2 / max(mse, 1e-12)), 2), ssim=round(_ssim(out["rgb"][k], ref_u8), 6),
                         px_gt1=round(float((lv > 1).mean()) * 100, 4), max_level=int(lv.max()),
                         depth_med_abs_m=float(np.median(dz)) if dz.size else None, depth_p99_abs_m=float(np.quantile(dz, 0.99)) if dz.size else None,
                         depth_valid_agree=round(float(((out["depth"][k] > 0) == (rr["depth"] > 0)).mean()) * 100, 3),
                         dexp_p99_rel=float(np.quantile(rel, 0.99)) if rel.size else None,
                         alpha_max_err=round(float(np.abs(out["alpha"][k] - rr["alpha"]).max()), 5), ref_s=round(time.perf_counter() - t0, 1)))
        print(rows[-1], flush=True)
    res = dict(kind="verify", time=time.strftime("%F %T"), machine=_machine(), size=[W, H], hfov=HFOV, **info, rows=rows)
    _dump(res, a.out)


def bench(a):
    import warnings
    from metalsim.render.splat.renderer import SplatRenderer
    s, pos, mats, info = _scene()
    rng = np.random.default_rng(0)
    rows = []
    for n in [int(x) for x in a.ns.split(",")]:
        idx = rng.choice(len(pos), n, replace=n > len(pos))
        p, m = pos[idx], mats[idx]
        r = SplatRenderer(s, n, width=W, height=H, hfov_deg=HFOV, outputs=("rgb", "depth"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.render_host(p, cam_mat=m); r.render_host(p, cam_mat=m)          # capacities settle
        st = r.stats(False)
        ms = min(r.bench(p, m, a.iters) for _ in range(3))
        prof = r.profile_frame(p, m)
        rows.append(dict(n=n, ms_per_frame=round(ms, 2), ms_per_cam=round(ms / n, 3), cams_per_s=round(1000 * n / ms, 1), sub_batch=r.batch,
                         records=st["records"], keys=st["keys"], stages_ms={k: round(v, 2) for k, v in prof.items()},
                         scene_mb=round(r.scene_bytes / 1e6), scratch_mb=round(r.scratch_bytes() / 1e6)))
        print(rows[-1], flush=True)
        del r
    res = dict(kind="bench", time=time.strftime("%F %T"), machine=_machine(), size=[W, H], hfov=HFOV, iters=a.iters, **info, rows=rows)
    _dump(res, a.out)


def options(a):
    import warnings
    from metalsim.render.splat import auto_crop_box, prepare
    from metalsim.render.splat.renderer import SplatRenderer
    s, pos, mats, info, c = _scene(cloud=True)
    pick = np.linspace(0, len(pos) - 1, 16).astype(int)
    p, m = pos[pick], mats[pick]
    q = lambda **kw: prepare(c, log=lambda *a_: None, **kw)
    vis = c.means[c.opacity >= 1 / 255]
    configs = [("default", s, {}), ("depth key 16 bit", s, dict(depth_bits=16)), ("no subtile masks", s, dict(defines={"EXP_NO_MASK": 1})),
               ("blend 2 px/thread", s, dict(blend="2px")), ("SH degree 0", q(sh_degree=0), {}), ("SH degree 1", q(sh_degree=1), {}),
               ("antialiased forced on", s, dict(antialiased=True)), ("crop: none", q(crop=None), {}),
               ("crop: tight (q1-99 box + 10 %)", q(crop=auto_crop_box(vis, grow=0.1)), {})]
    rows, ref_rgb = [], None
    for name, sc, kw in configs:
        r = SplatRenderer(sc, len(pick), width=W, height=H, hfov_deg=HFOV, outputs=("rgb", "depth"), **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.render_host(p, cam_mat=m)
            rgb = r.render_host(p, cam_mat=m)["rgb"].astype(np.float64)
        ms = min(r.bench(p, m, a.iters) for _ in range(3))
        if ref_rgb is None:
            ref_rgb = rgb
        mse = ((rgb - ref_rgb) ** 2).reshape(len(pick), -1).mean(1)
        psnr = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12))
        rows.append(dict(option=name, n_splats=sc.n, psnr_mean=round(float(np.minimum(psnr, 120).mean()), 1),
                         psnr_min=round(float(np.minimum(psnr, 120).min()), 1), identical=bool((mse == 0).all()), ms_per_frame=round(ms, 2)))
        print(rows[-1], flush=True)
        del r
    res = dict(kind="options", time=time.strftime("%F %T"), machine=_machine(), size=[W, H], hfov=HFOV, iters=a.iters, n_cams=len(pick),
               **info, rows=rows)
    _dump(res, a.out)


def _dump(res, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


def md(a):
    v = json.load(open(a.verify)); b = json.load(open(a.bench))
    L = [f"**Verification** ({v['scene']}, {v['n_splats']:,} splats; {len(v['rows'])} views at {v['size'][0]}x{v['size'][1]}, "
         f"HFOV {v['hfov']}; {v['machine']}; {v['time']}; `scripts/splat_bench.py verify`)", "",
         "| | PSNR dB | SSIM | pixels > 1 level | max level diff | median depth abs err | median-depth valid agree | expected depth p99 rel err | alpha max err |",
         "|---|---|---|---|---|---|---|---|---|"]
    R = v["rows"]
    f = lambda k: [r[k] for r in R if r[k] is not None]
    L.append(f"| worst over views | **{min(f('psnr')):.1f}** | **{min(f('ssim')):.5f}** | {max(f('px_gt1')):.4f} % | {max(f('max_level'))} | "
             f"{max(f('depth_med_abs_m')):.1e} m (p99 {max(f('depth_p99_abs_m')):.1e} m) | {min(f('depth_valid_agree')):.2f} % | "
             f"{max(f('dexp_p99_rel')):.1e} | {max(f('alpha_max_err')):.4f} |")
    L.append(f"| mean | {np.mean(f('psnr')):.1f} | {np.mean(f('ssim')):.5f} | {np.mean(f('px_gt1')):.4f} % | — | — | — | — | — |")
    L += ["", f"**Throughput** ({b['scene']}, {b['n_splats']:,} splats; {b['size'][0]}x{b['size'][1]}, HFOV {b['hfov']}; {b['machine']}; "
          f"{b['time']}; `scripts/gpu_run.sh splat_bench timing -- python scripts/splat_bench.py bench`; {b['iters']} frames back to back, best of 3)", "",
          "| cameras | ms / frame | ms / camera | cameras / s | sub-batch | records | keys | stage ms (one profiled frame) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        st = ", ".join(f"{k} {x}" for k, x in r["stages_ms"].items() if k != "total")
        L.append(f"| {r['n']} | {r['ms_per_frame']} | {r['ms_per_cam']} | **{r['cams_per_s']}** | {r['sub_batch']} | {r['records']:,} | {r['keys']:,} | {st} |")
    blocks = {"splat-bench": "\n".join(L)}
    if a.options:
        o = json.load(open(a.options)); d0 = o["rows"][0]["ms_per_frame"]
        O = [f"({o['scene']}, {o['n_cams']} cameras at {o['size'][0]}x{o['size'][1]}; {o['machine']}; {o['time']}; "
             f"`scripts/splat_bench.py options`, {o['iters']} frames back to back, best of 3)", "",
             "| option | splats | PSNR vs default, mean (min) dB | GPU ms / frame (16 cameras) |", "|---|---|---|---|"]
        for r in o["rows"]:
            qual = "—" if r["option"] == "default" else ("identical" if r["identical"] else f"{r['psnr_mean']} ({r['psnr_min']})")
            dt = "" if r["option"] == "default" else f" ({100 * (r['ms_per_frame'] / d0 - 1):+.0f} %)"
            O.append(f"| {r['option']} | {r['n_splats']:,} | {qual} | {r['ms_per_frame']}{dt} |")
        blocks["splat-options"] = "\n".join(O)
    if a.update:
        s = open(a.update).read()
        for tag, block in blocks.items():
            i, j = s.index(f"<!-- {tag}:begin -->"), s.index(f"<!-- {tag}:end -->")
            s = s[:i] + f"<!-- {tag}:begin -->\n" + block + "\n" + s[j:]
        open(a.update, "w").write(s)
        print("updated", a.update)
    else:
        print("\n\n".join(blocks.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("verify"); p.add_argument("--views", type=int, default=8); p.add_argument("--out", default="runs/splat/verify.json")
    p.set_defaults(fn=verify)
    p = sp.add_parser("bench"); p.add_argument("--ns", default="1,8,32,64"); p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default="runs/splat/bench.json"); p.set_defaults(fn=bench)
    p = sp.add_parser("options"); p.add_argument("--iters", type=int, default=20); p.add_argument("--out", default="runs/splat/options.json")
    p.set_defaults(fn=options)
    p = sp.add_parser("md"); p.add_argument("verify"); p.add_argument("bench"); p.add_argument("--options", default=None)
    p.add_argument("--update", default=None); p.set_defaults(fn=md)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
