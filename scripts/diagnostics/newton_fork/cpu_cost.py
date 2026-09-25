"""CPU cost proxy for the fork's solver changes (not a GPU timing): NewtonSim, N envs on the CPU device, wall time
of S control steps (16 XPBD substeps each at 1.25 ms), after a warm-up step. Either venv.
usage: python scripts/diagnostics/newton_fork/cpu_cost.py [N] [S] [k=v solver kwargs ...]"""
import sys, time, inspect, warp as wp, newton
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb
import metalsim.interop.warp_metal as wm

class _E:
    def __init__(self, *x): self.v = 0
    def next_value(self): self.v += 1; return self.v
wm.SharedEvent = _E
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256; S = int(sys.argv[2]) if len(sys.argv) > 2 else 10
skw = {}
if "joint_armature_inertia" in inspect.signature(newton.solvers.SolverXPBD.__init__).parameters:
    skw["joint_armature_inertia"] = "none"
for kv in sys.argv[3:]:
    k, v = kv.split("="); skw[k] = eval(v)
m = build_g1_model("flat", physics_dt=0.0025)[0]
sim = nb.NewtonSim(m, N, iterations=4, dt=0.00125, device="cpu", solver_kw=skw or None)
sim.launch_step(); t0 = time.perf_counter()
for _ in range(S): sim.launch_step()
el = time.perf_counter() - t0
print(f"| cpu cost | {newton.__file__.split('/')[-3]} {skw} | {N} envs x {S} steps | {el / S * 1e3:.1f} ms per control step |")
