"""Checks before a film starts: everything the later stages need is installed, so a rollout is never recorded for a
render that cannot run. The render, encode and doc stages run in the main venv (.venv); the check runs there in a
subprocess whatever venv the caller is in (the record stage of a Newton 1.5.2 task runs in .venv-newton152)."""
from __future__ import annotations

import os
import subprocess
import sys

FIX = 'pip install -e ".[full]"   (in the main venv, from the repo root; docs/GUIDE.md section 1)'

# (what, import / call to try, fix)
CHECKS_RENDER = [
    ("imageio (PNG frames)", "import imageio.v3", FIX),
    ("Metal tier-2 renderer", "import metalsim.render.tier2", FIX),
]
CHECK_OIDN = ("Open Image Denoise (tier 2's denoiser)", "from metalsim.render.denoise import _find_lib; _find_lib()",
              FIX + "  [pyoidn: macOS arm64 wheel, bundles OIDN]  or: brew install open-image-denoise  "
              "or: set OIDN_LIB  or: film without it, --no-denoise")
CHECKS_ENCODE = [("ffmpeg (imageio-ffmpeg)", "import imageio_ffmpeg; imageio_ffmpeg.get_ffmpeg_exe()",
                  FIX + "  or: set FFMPEG to an ffmpeg binary")]


def _main_python() -> str:
    from metalsim.render.film.__main__ import venv_python

    return venv_python("main")


def missing(task, stages) -> list:
    """[(what, fix, error)] for every requirement of ``stages`` that the main venv lacks."""
    checks = []
    if "render" in stages:
        checks += CHECKS_RENDER + ([CHECK_OIDN] if task.film.denoise == "oidn" else [])
    if "encode" in stages and not os.environ.get("FFMPEG"):
        checks += CHECKS_ENCODE
    if not checks:
        return []
    try:
        py = _main_python()
    except FileNotFoundError:
        py = sys.executable              # no .venv next to the repo (an install elsewhere): the caller's interpreter
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    out = []
    for what, code, fix in checks:
        r = subprocess.run([py, "-c", code], cwd=root, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            out.append((what, fix, err))
    return out


def check(task, stages) -> None:
    """Exit with the fix commands if a later stage could not run."""
    miss = missing(task, stages)
    if miss:
        lines = [f"[film] cannot film {task.name}: the {'/'.join(s for s in stages if s != 'record')} stage(s) need:"]
        for what, fix, err in miss:
            lines.append(f"  - {what}: {err}\n      fix: {fix}")
        print("\n".join(lines), file=sys.stderr, flush=True)
        raise SystemExit(3)
