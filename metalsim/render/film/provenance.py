"""Hardware, software pins and asset provenance for the film's JSON and doc (no absolute machine paths are kept)."""
from __future__ import annotations

import importlib
import os
import platform
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _run(cmd, cwd=None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_sha(path: str) -> str:
    """HEAD of the git checkout containing ``path`` (with "+dirty" for local changes), or ""."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    sha = _run(["git", "rev-parse", "--short=10", "HEAD"], cwd=d)
    if sha and _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=d):
        sha += "+dirty"
    return sha


def hardware() -> dict:
    hw = dict(machine=platform.machine(), python=platform.python_version())
    if platform.system() == "Darwin":
        hw.update(model=_run(["sysctl", "-n", "hw.model"]), cpu=_run(["sysctl", "-n", "machdep.cpu.brand_string"]),
                  memory_gb=round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30),
                  macos=_run(["sw_vers", "-productVersion"]), macos_build=_run(["sw_vers", "-buildVersion"]))
        sp = _run(["system_profiler", "SPDisplaysDataType"])
        for line in sp.splitlines():
            line = line.strip()
            if line.startswith("Total Number of Cores:"):
                hw["gpu_cores"] = line.split(":", 1)[1].strip()
            if line.startswith("Chipset Model:"):
                hw["gpu"] = line.split(":", 1)[1].strip()
    return hw


def software(modules=("warp", "newton", "mujoco", "mujoco_warp", "torch", "numpy")) -> dict:
    """Version and source checkout (git SHA when the module is imported from a git checkout) of each module."""
    out = {"metalsim": git_sha(ROOT)}
    for name in modules:
        try:
            m = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        f = getattr(m, "__file__", None) or ""
        ent = {"version": str(getattr(m, "__version__", ""))}
        if f and "site-packages" not in f:
            ent["git"] = git_sha(os.path.dirname(f))
            ent["checkout"] = os.path.relpath(os.path.dirname(os.path.dirname(f)), ROOT) if f.startswith(ROOT) else \
                os.path.basename(os.path.dirname(os.path.dirname(f)))
        out[name] = ent
    return out


def relpath(p: str | None) -> str | None:
    """Repo-relative form of a path (docs and JSON never carry machine paths)."""
    if not p:
        return p
    ap = os.path.abspath(p)
    if ap.startswith(ROOT + os.sep):
        return os.path.relpath(ap, ROOT)
    for marker in ("/assets/", "/runs/", "/upstream/"):
        if marker in ap:
            return ap[ap.index(marker) + 1:]
    return os.path.basename(ap)
