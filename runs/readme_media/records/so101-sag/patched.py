"""Import this first to run the PATCHED metalsim.physics.actuation and metalsim.render.film.so101 (patch/new/) in place
of the working tree's, without touching the tree (so101-sag)."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
import metalsim.physics  # noqa: E402,F401
import metalsim.render.film  # noqa: E402,F401

for name, fn in (("metalsim.physics.actuation", "actuation.py"), ("metalsim.render.film.so101", "so101.py")):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "patch", "new", fn))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
