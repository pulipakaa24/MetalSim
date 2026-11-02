"""Import this first to run this stream's metalsim.render.film.so101 (so101_patched.py: the live so101.py + the so101-sag
hunks + the GPU friction wiring) in place of the tree's, without touching the tree (so101-bam-gpu). The draft of
actuation.py that this once loaded under BAM_DEV_ACTUATION=1 is now the tree's file (removed from scratch)."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
import metalsim.physics  # noqa: E402,F401
import metalsim.render.film  # noqa: E402,F401

MODS = [("metalsim.render.film.so101", "so101_patched.py")]
for name, fn in MODS:
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "ROOT"):            # so101.py derives the repo root from its own path (used at call time only)
        mod.ROOT = ROOT
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[parent], child, mod)
