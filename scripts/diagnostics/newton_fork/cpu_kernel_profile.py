"""Per-kernel CPU time of one NewtonSim control step (CPU device launches are synchronous), for comparing the fork
with pinned upstream (cost proxy, not a GPU timing). usage: python cpu_kernel_profile.py [N] [S] [k=v ...]"""
import sys, time, inspect, collections, warp as wp, newton
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb
import metalsim.interop.warp_metal as wm

class _E:
    def __init__(self, *x): self.v = 0
    def next_value(self): self.v += 1; return self.v
wm.SharedEvent = _E
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256; S = int(sys.argv[2]) if len(sys.argv) > 2 else 5
skw = {}
if "joint_armature_inertia" in inspect.signature(newton.solvers.SolverXPBD.__init__).parameters:
    skw["joint_armature_inertia"] = "none"
for kv in sys.argv[3:]:
    k, v = kv.split("="); skw[k] = eval(v)
m = build_g1_model("flat", physics_dt=0.0025)[0]
sim = nb.NewtonSim(m, N, iterations=4, dt=0.00125, device="cpu", solver_kw=skw or None)
sim.launch_step()
T = collections.defaultdict(float); C = collections.Counter()
_launch = wp.launch
def timed(kernel, *a, **k):
    t0 = time.perf_counter(); r = _launch(kernel, *a, **k); T[kernel.key] += time.perf_counter() - t0; C[kernel.key] += 1; return r
wp.launch = timed
t0 = time.perf_counter()
for _ in range(S): sim.launch_step()
tot = time.perf_counter() - t0
print(f"newton {newton.__file__.split('/')[-3]} {skw}: {tot / S * 1e3:.1f} ms per control step; kernels {sum(T.values()) / S * 1e3:.1f} ms")
for k, v in sorted(T.items(), key=lambda x: -x[1])[:14]:
    print(f"  {k:60s} {v / S * 1e3:8.2f} ms  ({C[k] // S} launches/step)")
