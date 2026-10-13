"""speed_bench.py with newton_compat.install(device_args=<env MPM_DA>) forced (1 device, 0 constant, keep: Warp's policy)."""
import os, runpy, sys
from metalsim.physics.mpm import newton_compat
da = os.environ.get("MPM_DA", "1")
_install = newton_compat.install
def install(*a, **kw):
    kw["device_args"] = None if da == "keep" else da == "1"
    return _install(*a, **kw)
newton_compat.install = install
sys.argv = ["speed_bench.py"] + sys.argv[1:]
runpy.run_path("/Users/aditya/robosim/metalsim/physics/mpm/probes/speed_bench.py", run_name="__main__")
