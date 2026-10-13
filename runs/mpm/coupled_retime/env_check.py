"""Which Warp / Newton / metalsim / MuJoCo Warp the re-timing jobs import (run with run.sh's PYTHONPATH).
Reads each package checkout's .git/HEAD (and the ref it names) directly."""
import os

import metalsim
import mujoco_warp
import newton
import warp


def head(mod):
    d = os.path.dirname(os.path.dirname(mod.__file__))
    g = os.path.join(d, ".git")
    if os.path.isfile(g):
        g = open(g).read().split("gitdir:")[1].strip()
    try:
        h = open(os.path.join(g, "HEAD")).read().strip()
    except OSError:
        return "not a git checkout"
    if h.startswith("ref: "):
        ref = h[5:]
        p = os.path.join(g, ref)
        if os.path.exists(p):
            return f"{ref} {open(p).read().strip()}"
        return f"{ref} (packed or worktree ref)"
    return h


for m in (warp, newton, mujoco_warp, metalsim):
    print(f"{m.__name__:12s} {m.__file__}  HEAD {head(m)}")
import importlib.metadata as md  # noqa: E402

print("warp", warp.__version__, "newton", newton.__version__, "mujoco-warp", md.version("mujoco-warp"),
      "mujoco", md.version("mujoco"))
