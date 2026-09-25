"""Physics-only throughput (graph replay of one 50 Hz control step, 4096 envs, env-steps/s) of the drift remedies next
to the XPBD settings: XPBD 4 it 1.25 ms / 0.625 ms, XPBD + joint projection, Featherstone at 0.3125 ms.

usage: python scripts/diagnostics/newton_remedy_throughput.py [N]"""
import sys, time, warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics.newton_backend import NewtonSim

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
m = build_g1_model("flat", physics_dt=0.0025)[0]
for tag, kw in (("XPBD 4 it 1.25 ms", dict(iterations=4, dt=0.00125)), ("XPBD 4 it 0.625 ms", dict(iterations=4, dt=0.000625)),
                ("XPBD 4 it 1.25 ms + projection", dict(iterations=4, dt=0.00125, project=True)),
                ("XPBD 4 it 1.25 ms recentred", dict(iterations=4, dt=0.00125, recenter=True)),
                ("Featherstone 0.3125 ms", dict(dt=0.0003125, solver="featherstone"))):
    try:
        sim = NewtonSim(m, N, **kw)
        sim.step(); sim.synchronize(); t0 = time.perf_counter()
        for _ in range(20): sim.step()
        sim.synchronize(); print(f"{tag}: {N * 20 / (time.perf_counter() - t0):,.0f} env-steps/s physics only (N={N})", flush=True)
        del sim
    except Exception as e:
        print(f"{tag}: failed {e!r:.150}", flush=True)
