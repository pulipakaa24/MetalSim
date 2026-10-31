"""Encode stage: PNG frames -> H.264 mp4 (a full-resolution master next to the frames and a smaller copy for the
docs) and JPEG stills. ffmpeg from ``imageio_ffmpeg`` (the main venv) or ``FFMPEG``.

README clips (``encode_webp``, ``encode_mp4_clip``) cut a window from a frames directory or a video:

    python -m metalsim.render.film.encode webp SRC OUT.webp --start 3 --seconds 5 [--crop-x 0.3]
    python -m metalsim.render.film.encode mp4  SRC OUT.mp4  --start 3 --seconds 5 [--crop-x 0.3]

The WebP default is MuJoCo Warp's README recipe (mujoco_warp/record.py): 320x240, 30 fps, 5 s, Pillow
``quality=70``, ``loop=0``, ``duration=round(1000 / fps)`` and Pillow's other defaults, no size budget (``--budget MB``
adds a search). A 16:9 source is cropped to 4:3 (centred, or at ``--crop-x``) and then Lanczos-scaled; the mp4 export
uses the same crop at 640x480. ffmpeg decodes, trims, resamples the frame rate, crops and scales to RGB; Pillow
(libwebp) writes the WebP, so its RGB -> YUV step is libwebp's own. Each output gets a ``<out>.json`` with the
settings, the crop and a colour check against the scaled source frames."""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from metalsim.render.film import provenance


def ffmpeg_exe() -> str:
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(shot_dir: str, out_mp4: str, fps: float, crf: int = 22, width: int | None = 1280, stills=(),
           stills_prefix: str | None = None) -> dict:
    frames = sorted(glob.glob(os.path.join(shot_dir, "frames", "frame_*.png")))
    if not frames:
        raise FileNotFoundError(f"no frames in {shot_dir}/frames")
    first = int(os.path.basename(frames[0])[6:10])
    ff = ffmpeg_exe()
    res = {}
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    master = os.path.join(shot_dir, "master.mp4")
    for path, vf, q in ((master, None, 16), (out_mp4, f"scale={width}:-2:flags=lanczos" if width else None, crf)):
        cmd = [ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-start_number", str(first), "-i",
               os.path.join(shot_dir, "frames", "frame_%04d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", str(q),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if vf:
            cmd += ["-vf", vf]
        t0 = time.perf_counter()
        subprocess.run(cmd + [path], check=True)
        res[os.path.basename(path)] = dict(path=provenance.relpath(path), seconds=time.perf_counter() - t0, crf=q,
                                           bytes=os.path.getsize(path), width=width if vf else None)
    out_stills = []
    if stills:
        import imageio.v3 as iio
        import numpy as np

        prefix = stills_prefix or os.path.splitext(out_mp4)[0]
        for i, frac in enumerate(stills):
            src = frames[min(int(round(frac * (len(frames) - 1))), len(frames) - 1)]
            img = iio.imread(src)
            dst = f"{prefix}_still{i + 1}.jpg"
            iio.imwrite(dst, np.asarray(img)[..., :3], quality=90)
            out_stills.append(dict(path=provenance.relpath(dst), frame=int(os.path.basename(src)[6:10]),
                                   bytes=os.path.getsize(dst)))
    out = dict(frames=len(frames), fps=fps, encodes=res, stills=out_stills,
               encode_s=sum(v["seconds"] for v in res.values()))
    with open(os.path.join(shot_dir, "encode.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[encode] {len(frames)} frames -> {provenance.relpath(out_mp4)} ({res[os.path.basename(out_mp4)]['bytes'] / 1e6:.1f} MB),"
          f" {len(out_stills)} stills, {out['encode_s']:.1f} s", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# README clips: animated WebP (autoplays in a GitHub <img>) and a small H.264 mp4 for linking


def _display_path(p: str) -> str:
    """Repo-relative inside the checkout, ``~/...`` under the home directory, else the basename (no machine paths)."""
    ap = os.path.abspath(p)
    if ap.startswith(provenance.ROOT + os.sep):
        return os.path.relpath(ap, provenance.ROOT)
    home = os.path.expanduser("~")
    if ap.startswith(home + os.sep):
        return "~/" + os.path.relpath(ap, home)
    return os.path.basename(ap)


def probe_source(src: str, src_fps: float | None = None) -> dict:
    """A frames directory (``frame_%04d.png``, or a shot directory holding ``frames/``) or a video file."""
    if os.path.isdir(src):
        d = os.path.join(src, "frames") if os.path.isdir(os.path.join(src, "frames")) else src
        frames = sorted(glob.glob(os.path.join(d, "frame_*.png")))
        if not frames:
            raise FileNotFoundError(f"no frame_*.png in {d}")
        nums = [int(os.path.basename(f)[6:10]) for f in frames]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            raise ValueError(f"{d}: frame numbers are not contiguous ({nums[0]}..{nums[-1]}, {len(nums)} files)")
        from PIL import Image

        with Image.open(frames[0]) as im:
            w, h = im.size
        fps = float(src_fps or 30.0)
        return dict(kind="frames", path=d, first=nums[0], frames=len(nums), fps=fps, width=w, height=h,
                    seconds=len(nums) / fps)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    import imageio_ffmpeg

    n, secs = imageio_ffmpeg.count_frames_and_secs(src)
    meta = next(imageio_ffmpeg.read_frames(src))
    fps = float(src_fps or meta["fps"])
    w, h = meta["source_size"]
    return dict(kind="video", path=src, first=0, frames=int(n), fps=fps, width=int(w), height=int(h),
                seconds=int(n) / fps)


def parse_crop_x(v) -> float | int | None:
    """``--crop-x``: a fraction of the spare width in [0, 1] (0 = left edge, 0.5 = centred, 1 = right edge), or a
    pixel offset of the crop's left edge in source pixels (a number > 1, or ``<n>px``). None/empty = centred."""
    if v is None or v == "":
        return None
    if isinstance(v, str) and v.endswith("px"):
        return int(v[:-2])
    x = float(v)
    if x < 0:
        raise ValueError(f"crop-x {v!r} is negative")
    return x if x <= 1 else int(round(x))


def crop_box(src_w: int, src_h: int, out_w: int, out_h: int, crop_x=None) -> dict:
    """The largest ``out_w:out_h`` box in the source: full height and ``crop_x`` (see ``parse_crop_x``) when the
    source is wider (16:9 -> 4:3), else full width, centred vertically. ``slack`` is the spare width/height."""
    if src_w * out_h >= src_h * out_w:
        cw, ch = int(round(src_h * out_w / out_h)), src_h
        slack = src_w - cw
        if crop_x is None:
            x = slack // 2
        elif isinstance(crop_x, float):
            x = int(round(crop_x * slack))
        else:
            x = int(crop_x)
        if not 0 <= x <= slack:
            raise ValueError(f"crop x offset {x} px is outside 0..{slack} (source {src_w}x{src_h}, box {cw}x{ch})")
        return dict(x=x, y=0, w=cw, h=ch, slack=slack)
    cw, ch = src_w, int(round(src_w * out_h / out_w))
    return dict(x=0, y=(src_h - ch) // 2, w=cw, h=ch, slack=0)


def read_clip(info: dict, start: float, seconds: float | None, width: int, fps: float | None,
              height: int | None = None, crop_x=None):
    """RGB uint8 frames (N, H, W, 3) of ``[start, start + seconds)``, resampled to ``fps``, cropped to the
    ``width:height`` aspect (``crop_box``) and Lanczos-scaled to ``width`` x ``height``. ``height`` None keeps the
    source aspect (no crop; height rounded to even). The window is cut by source frame index, so it is exact."""
    import numpy as np

    f0 = int(round(start * info["fps"]))
    if f0 < 0 or f0 >= info["frames"]:
        raise ValueError(f"start {start:g} s is outside the source's {info['seconds']:.3f} s "
                         f"({info['frames']} frames at {info['fps']:g} fps)")
    n = info["frames"] - f0 if seconds is None else int(round(seconds * info["fps"]))
    if f0 + n > info["frames"]:
        print(f"[clip] window {start:g} s + {seconds:g} s runs past the source's {info['seconds']:.3f} s; "
              f"using the {(info['frames'] - f0) / info['fps']:.3f} s to the end", flush=True)
        n = info["frames"] - f0
    out_fps = float(fps or info["fps"])
    if height:
        box = crop_box(info["width"], info["height"], width, height, crop_x)
    else:
        height = int(round(info["height"] * width / info["width"] / 2)) * 2
        box = dict(x=0, y=0, w=info["width"], h=info["height"], slack=0)
    ff = ffmpeg_exe()
    if info["kind"] == "frames":
        inp = ["-framerate", f"{info['fps']:g}", "-start_number", str(info["first"]), "-i",
               os.path.join(info["path"], "frame_%04d.png")]
    else:
        inp = ["-r", f"{info['fps']:g}", "-i", info["path"]]  # untagged yuv decodes as BT.601, as it was encoded
    vf = [f"trim=start_frame={f0}:end_frame={f0 + n}", "setpts=PTS-STARTPTS"]
    if abs(out_fps - info["fps"]) > 1e-6:
        vf.append(f"fps={out_fps:g}")
    vf.append("format=rgb24")
    if (box["w"], box["h"]) != (info["width"], info["height"]):
        vf.append(f"crop={box['w']}:{box['h']}:{box['x']}:{box['y']}:exact=1")
    vf.append(f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int")
    # -fps_mode passthrough: without it the rawvideo output defaults to 25 fps CFR after setpts and drops frames
    cmd = [ff, "-loglevel", "error", *inp, "-vf", ",".join(vf), "-fps_mode", "passthrough", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8)
    arr = arr.reshape(-1, height, width, 3)
    want = n * out_fps / info["fps"]
    if abs(len(arr) - want) > 1:
        raise RuntimeError(f"ffmpeg returned {len(arr)} frames for {n} source frames at {info['fps']:g} -> "
                           f"{out_fps:g} fps (expected {want:.1f})")
    return arr, dict(start_frame=f0 + info["first"], source_frames=n, start_s=f0 / info["fps"],
                     seconds=n / info["fps"], fps=out_fps, width=width, height=height,
                     crop=dict(x=box["x"], y=box["y"], w=box["w"], h=box["h"], spare_px=box["slack"]))


def _colour_check(ref, decoded, idx) -> dict:
    """Per-channel mean bias and PSNR of decoded frames against the scaled source frames (sampled indices)."""
    import numpy as np

    a = ref[idx].astype(np.float64)
    b = decoded.astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    return dict(frames=[int(i) for i in idx], mean_rgb_source=[round(float(x), 2) for x in a.mean((0, 1, 2))],
                mean_rgb_output=[round(float(x), 2) for x in b.mean((0, 1, 2))],
                psnr_db=round(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)), 2))


def _webp_bytes(frames, frame_ms: int, quality: int) -> bytes:
    import io

    from PIL import Image

    ims = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    # MuJoCo Warp's recipe (mujoco_warp/record.py): save_all, integer duration, loop=0, quality, and Pillow's other
    # defaults. Those are lossy, method 0 and key frames every 3 to 5 frames (kmin=3/kmax=5, gif2webp's defaults;
    # PIL/WebPImagePlugin.py _save_all). Do not pass kmax <= 0: libwebp then turns key frames off for the whole clip
    # and carries near-identical 8x8 blocks over from older frames (anim_encode.c FlattenSimilarBlocks), which shows
    # as stale grey squares on slowly shaded white surfaces.
    ims[0].save(buf, format="WEBP", save_all=True, append_images=ims[1:], duration=frame_ms, loop=0, quality=quality)
    return buf.getvalue()


def _ladder(top, floor, step):
    return list(range(top, floor - 1, -step)) or [top]


def encode_webp(src: str, out: str, start: float = 0.0, seconds: float | None = 5.0, width: int = 320,
                height: int | None = 240, fps: float | None = 30.0, crop_x=None, max_bytes: int | None = None,
                src_fps: float | None = None, quality: int = 70, min_quality: int = 50, quality_step: int = 10,
                fps_ladder=(24, 20, 15), width_ladder=(0.75, 0.5), command: str | None = None) -> dict:
    """Animated WebP of ``[start, start + seconds)`` of ``src``, by MuJoCo Warp's README recipe: ``width`` x
    ``height`` (default 320x240; a wider source is cropped to that aspect at ``crop_x``, centred by default, then
    Lanczos-scaled), ``fps`` (30), ``duration=round(1000 / fps)`` ms per frame, ``loop=0``, Pillow ``quality`` (70)
    and Pillow's other defaults. No size budget unless ``max_bytes`` is given.

    With ``max_bytes``: quality from ``quality`` down to ``min_quality`` in ``quality_step``; then the lower frame
    rates in ``fps_ladder``; then smaller sizes (``width_ladder`` fractions). The first fit is kept; if nothing fits,
    the smallest attempt is written and ``fits`` is false. ``<out>.json`` records settings, crop and readback."""
    import numpy as np

    info = probe_source(src, src_fps)
    fps0 = float(fps or info["fps"])
    fpss = [fps0]
    widths = [int(width)]
    qualities = [quality]
    if max_bytes:
        fpss += [float(f) for f in fps_ladder if f < fps0]
        widths += [int(round(width * k / 2)) * 2 for k in width_ladder]
        qualities = _ladder(quality, min_quality, quality_step)
    fits = (lambda b: True) if not max_bytes else (lambda b: b <= max_bytes)
    tries, best, t0 = [], None, time.perf_counter()
    for w in widths:
        h = int(round(height * w / width / 2)) * 2 if (height and w != width) else height
        for f in fpss:
            frames, window = read_clip(info, start, seconds, w, f, h, crop_x)
            frame_ms = int(round(1000.0 / f))
            for q in qualities:
                data = _webp_bytes(frames, frame_ms, q)
                tries.append(dict(width=w, height=frames.shape[1], fps=f, quality=q, bytes=len(data)))
                if best is None or len(data) < len(best[0]) or fits(len(data)):
                    best = (data, frames, window, q, frame_ms)
                if fits(len(data)):
                    break
            if fits(tries[-1]["bytes"]):
                break
        if fits(tries[-1]["bytes"]):
            break
    data, frames, window, q, frame_ms = best
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(data)
    # read it back: frame count, total duration, size, and colours against the scaled source
    # (libwebp merges a frame into the previous one, with a longer duration, when it is the same within the lossy
    # tolerance, so the output can have fewer frames than the window; timestamps map them back to source frames)
    from PIL import Image

    with Image.open(out) as im:
        n_out, size = im.n_frames, im.size
        stamps, dur_ms, dec = [], 0, {}
        idx = set(np.linspace(0, n_out - 1, 4).round().astype(int).tolist())
        for i in range(n_out):
            im.seek(i)
            im.load()  # Pillow sets timestamp/duration on load
            stamps.append(im.info["timestamp"])
            dur_ms += im.info["duration"]
            if i in idx:
                dec[i] = np.asarray(im.convert("RGB"))
    src_idx = [min(int(round(stamps[i] / frame_ms)), len(frames) - 1) for i in sorted(dec)]
    res = dict(output=_display_path(out), bytes=len(data), fits=fits(len(data)), max_bytes=max_bytes,
               width=size[0], height=size[1], frames=n_out, frames_in=len(frames), fps=window["fps"],
               frame_ms=frame_ms, duration_s=dur_ms / 1000.0, quality=q, loop=0,
               encoder=f"Pillow {__import__('PIL').__version__} (libwebp, lossy; Pillow defaults: method 0, kmin 3, "
                       f"kmax 5)",
               recipe="mujoco_warp/record.py: save_all, duration=int ms, loop=0, quality=70, 320x240, 30 fps",
               scaler="ffmpeg crop + lanczos (RGB)", crop=window["crop"],
               source=dict(path=_display_path(info["path"]), kind=info["kind"], fps=info["fps"],
                           width=info["width"], height=info["height"], frames=info["frames"],
                           window_start_s=window["start_s"], window_s=window["seconds"],
                           first_frame=window["start_frame"], frames_used=window["source_frames"]),
               colour_check=_colour_check(frames, np.stack([dec[i] for i in sorted(dec)]), src_idx), tries=tries,
               encode_s=round(time.perf_counter() - t0, 2), command=command)
    with open(out + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    c = window["crop"]
    print(f"[webp] {res['output']}: {size[0]}x{size[1]} (crop {c['w']}x{c['h']}+{c['x']}+{c['y']}), {n_out} frames "
          f"({len(frames)} in) at {frame_ms} ms, {res['duration_s']:.3f} s, q{q}, {len(data) / 1e6:.2f} MB"
          f"{'' if res['fits'] else ' (OVER BUDGET)'}, PSNR {res['colour_check']['psnr_db']} dB", flush=True)
    return res


def encode_mp4_clip(src: str, out: str, start: float = 0.0, seconds: float | None = 5.0, width: int = 640,
                    height: int | None = 480, fps: float | None = None, crop_x=None, max_bytes: int = 3_000_000,
                    src_fps: float | None = None, crf: int = 20, max_crf: int = 32, crf_step: int = 2,
                    command: str | None = None) -> dict:
    """H.264 mp4 of the same window and crop (for linking; default 640x480): yuv420p, BT.709 tagged,
    ``+faststart``, no audio; CRF rises in ``crf_step`` until the file fits ``max_bytes``."""
    import numpy as np

    info = probe_source(src, src_fps)
    frames, window = read_clip(info, start, seconds, width, fps or info["fps"], height, crop_x)
    h, w = frames.shape[1:3]
    ff = ffmpeg_exe()
    tries, t0 = [], time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for c in _ladder(max_crf, crf, crf_step)[::-1]:
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", f"{window['fps']:g}", "-i", "-",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p,"
               "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
               "-c:v", "libx264", "-preset", "slow", "-crf", str(c), "-colorspace", "bt709", "-color_primaries",
               "bt709", "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart", "-an", out]
        subprocess.run(cmd, input=frames.tobytes(), check=True)
        tries.append(dict(crf=c, bytes=os.path.getsize(out)))
        if tries[-1]["bytes"] <= max_bytes:
            break
    # decode back as a browser does (BT.709, limited range, exact chroma upsampling) for the colour check; swscale's
    # default fast path reads about 2 levels dark here and is not what a viewer sees
    idx = np.unique(np.linspace(0, len(frames) - 1, 4).round().astype(int))
    raw = subprocess.run([ff, "-loglevel", "error", "-i", out, "-vf",
                          "scale=in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    dec = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    res = dict(output=_display_path(out), bytes=tries[-1]["bytes"], fits=tries[-1]["bytes"] <= max_bytes,
               max_bytes=max_bytes, width=w, height=h, frames=int(len(dec)), fps=window["fps"],
               duration_s=len(dec) / window["fps"], crf=tries[-1]["crf"], codec="libx264 yuv420p bt709 +faststart",
               scaler="ffmpeg crop + lanczos (RGB)", crop=window["crop"],
               source=dict(path=_display_path(info["path"]), kind=info["kind"], fps=info["fps"],
                           width=info["width"], height=info["height"], frames=info["frames"],
                           window_start_s=window["start_s"], window_s=window["seconds"],
                           first_frame=window["start_frame"], frames_used=window["source_frames"]),
               colour_check=_colour_check(frames, dec[idx], idx), tries=tries,
               encode_s=round(time.perf_counter() - t0, 2), command=command)
    with open(out + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[mp4] {res['output']}: {w}x{h}, {len(dec)} frames, crf {res['crf']}, {res['bytes'] / 1e6:.2f} MB, "
          f"PSNR {res['colour_check']['psnr_db']} dB", flush=True)
    return res


def _clip_main(kind: str, argv) -> int:
    ap = argparse.ArgumentParser(prog=f"python -m metalsim.render.film.encode {kind}",
                                 description="Cut a README clip from a frames directory or a video.")
    ap.add_argument("src", help="frames directory (frame_%%04d.png, or a shot directory with frames/) or a video")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0, help="window start [s] of the source")
    ap.add_argument("--seconds", type=float, default=5.0, help="window length [s]; 0 = to the end")
    ap.add_argument("--width", type=int, default=320 if kind == "webp" else 640)
    ap.add_argument("--height", type=int, default=240 if kind == "webp" else 480,
                    help="output height; the source is cropped to width:height first. 0 = keep the source aspect")
    ap.add_argument("--crop-x", default=None,
                    help="horizontal crop position: a fraction of the spare width in [0, 1] (0 left, 0.5 centre, "
                         "1 right) or a left-edge offset in source pixels (> 1, or e.g. 120px). Default: centred")
    ap.add_argument("--fps", type=float, default=30.0 if kind == "webp" else None,
                    help="output frame rate (resampled from the source rate when different)")
    ap.add_argument("--src-fps", type=float, default=None, help="source frame rate (frames directory; default 30)")
    if kind == "webp":
        ap.add_argument("--quality", type=int, default=70, help="Pillow WebP quality (with --budget: the first tried)")
        ap.add_argument("--budget", type=float, default=None,
                        help="byte budget [MB, 1e6]; searches quality down to --min-quality, then fps 24/20/15, "
                             "then smaller sizes. Default: none (one encode at --quality)")
        ap.add_argument("--min-quality", type=int, default=50, help="lowest quality tried with --budget")
    else:
        ap.add_argument("--max-mb", type=float, default=3.0, help="byte budget [MB, 1e6]")
    a = ap.parse_args(argv)
    cmd = " ".join(["python -m metalsim.render.film.encode", kind] + [
        _display_path(x) if os.path.exists(x) else x for x in argv])
    kw = dict(start=a.start, seconds=a.seconds or None, width=a.width, height=a.height or None, fps=a.fps,
              crop_x=parse_crop_x(a.crop_x), src_fps=a.src_fps, command=cmd)
    if kind == "webp":
        res = encode_webp(a.src, a.out, quality=a.quality, min_quality=a.min_quality,
                          max_bytes=int(a.budget * 1e6) if a.budget else None, **kw)
    else:
        res = encode_mp4_clip(a.src, a.out, max_bytes=int(a.max_mb * 1e6), **kw)
    return 0 if res["fits"] else 3


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("webp", "mp4"):
        return _clip_main(argv[0], argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--stills", type=float, nargs="*", default=())
    a = ap.parse_args(argv)
    encode(a.shot, a.out, a.fps, a.crf, a.width, a.stills)


if __name__ == "__main__":
    sys.exit(main())
