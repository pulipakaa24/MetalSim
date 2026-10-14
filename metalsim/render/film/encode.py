"""Encode stage: PNG frames -> H.264 mp4 (a full-resolution master next to the frames and a smaller copy for the
docs) and JPEG stills. ffmpeg from ``imageio_ffmpeg`` (the main venv) or ``FFMPEG``."""
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


def main(argv=None):
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
