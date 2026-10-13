"""speed_bench.py with newton_compat.install(device_args=<env MPM_DA>) forced (1: device, 0: constant, keep: Warp's
"auto" policy), printing which Warp / Newton / metalsim trees were imported."""
import os, runpy, sys
from metalsim.physics.mpm import newton_compat
import warp, newton, metalsim
print("TREES", warp.__file__, newton.__file__, metalsim.__file__, flush=True)
da = os.environ.get("MPM_DA", "keep")
_install = newton_compat.install


def install(*a, **kw):
    kw["device_args"] = None if da == "keep" else da == "1"
    return _install(*a, **kw)


newton_compat.install = install
sys.argv = ["speed_bench.py"] + sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(here, "../../../metalsim/physics/mpm/probes/speed_bench.py"), run_name="__main__")
