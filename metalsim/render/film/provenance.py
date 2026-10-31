"""Hardware, software pins and asset provenance for the film's JSON and doc (no absolute machine paths are kept)."""
from __future__ import annotations

import importlib
import os
import platform
import re
import subprocess
import time

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


# pmset's power-mode setting, as ``pmset -g`` prints it: "powermode" (macOS releases with High Power Mode), else
# "lowpowermode". Values: 0 Automatic, 1 Low Power, 2 High Power. pmset's man page does not document the key. The
# sources: the System Settings choice (Automatic / Low Power / High Power, https://support.apple.com/101613); 2 =
# High Power as the original post of https://developer.apple.com/forums/thread/835131 reads it ("`powermode 2` (the
# High Power equivalent)"; no reply contradicts or confirms the value), where Apple DTS states that pmset returns "the
# kernel's current configuration from IOKit", while system_profiler's Low Power Mode status is the machine's current
# state ("even in automatic or high power mode, ... idle machines show as 'Low = Yes'"); 1 = Low Power as for the older
# two-state ``lowpowermode``. On the 14-inch (Mac16,6), 2026-10-31, pmset showed "powermode 2" while system_profiler
# reported "Low Power Mode: Yes", so system_profiler is not used here. This is the configured setting, not proof that
# the machine ran in that mode: High Power engages only near the thermal limit (DTS, same thread).
POWER_MODES = {0: "Automatic", 1: "Low Power", 2: "High Power"}


def parse_power(batt: str, settings: str) -> dict:
    """Power source, battery and power-mode setting from the text of ``pmset -g batt`` and ``pmset -g``. Every value is
    None when its line is absent (another OS, or a Mac without a battery for the battery fields)."""
    out = dict(source=None, battery_percent=None, battery_state=None, mode=None, mode_setting=None)
    m = re.search(r"Now drawing from '([^']+)'", batt or "")
    if m:
        out["source"] = m.group(1)                 # "AC Power", "Battery Power" or "UPS Power"
    m = re.search(r"InternalBattery\S*[^\n]*?\t\s*(\d+)%;\s*([^;\n]+)", batt or "")
    if m:
        out["battery_percent"], out["battery_state"] = int(m.group(1)), m.group(2).strip()
    for key in ("powermode", "lowpowermode"):      # the newer key wins when both are printed
        m = re.search(rf"^\s*{key}\s+(\d+)\s*$", settings or "", re.M)
        if m:
            out["mode_setting"] = f"{key} {m.group(1)}"
            out["mode"] = POWER_MODES.get(int(m.group(1)), f"unknown ({m.group(1)})")
            break
    return out


def power() -> dict:
    """The power state now (``parse_power`` of ``pmset -g batt`` and ``pmset -g``) and when it was read; all None off
    macOS. The stages stamp it into their JSON (``hardware()``), so a doc states the power a stage ran under
    instead of guessing."""
    darwin = platform.system() == "Darwin"
    out = parse_power(_run(["pmset", "-g", "batt"]) if darwin else "", _run(["pmset", "-g"]) if darwin else "")
    out["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out


def power_text(pw: dict | None) -> str:
    """One line for a doc: "on battery (77 %, discharging), power-mode setting High Power (pmset `powermode 2`)"."""
    if not pw:
        return "not recorded"
    src = {"AC Power": "on AC power", "Battery Power": "on battery", "UPS Power": "on UPS power"}.get(
        pw.get("source"), f"power source {pw['source']}" if pw.get("source") else "power source not recorded")
    if pw.get("battery_percent") is not None:
        src += f" ({pw['battery_percent']} %" + (f", {pw['battery_state']}" if pw.get("battery_state") else "") + ")"
    mode = (f"power-mode setting {pw['mode']} (pmset `{pw['mode_setting']}`)" if pw.get("mode")
            else "power-mode setting not recorded")
    return f"{src}, {mode}"


def hardware() -> dict:
    """The machine (model, CPU, GPU, memory, macOS) and its power state when called (``power()``)."""
    hw = dict(machine=platform.machine(), python=platform.python_version(), power=power())
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
