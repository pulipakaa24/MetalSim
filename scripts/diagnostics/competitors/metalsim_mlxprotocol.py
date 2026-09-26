"""MuJoCo-MLX-Cpp's bench_batched_step_file protocol on MetalSim: model file as-is, Euler integrator, reset to
qpos0, zero ctrl, 60 steps per sample (reset included), 1 warm-up + 3 measured samples; solver budget from the XML."""
import sys, time, json, mujoco, torch
from metalsim.physics.batch import BatchSim, BatchSimOptions
path, name, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = mujoco.MjSpec.from_file(path)
spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
for g in spec.geoms: g.margin = 0.0          # MuJoCo Warp: non-zero margin unsupported with NATIVECCD (Go2 has 1 mm)
m = spec.compile()
sim = BatchSim(m, N, options=BatchSimOptions())
sim.t.ctrl.zero_(); torch.mps.synchronize()
def sample():
    sim.reset(); sim.synchronize()
    for _ in range(60): sim.step()
    sim.synchronize()
sample()
ts = []
for _ in range(3):
    t0 = time.perf_counter(); sample(); ts.append(time.perf_counter() - t0)
best = min(ts); mean = sum(ts) / 3
print("RESULT", json.dumps(dict(engine="metalsim", model=name, N=N, iters=int(sim.m.opt.iterations), ls=int(sim.m.opt.ls_iterations),
      sps_mean=round(N * 60 / mean), sps_best=round(N * 60 / best), finite=bool(torch.isfinite(sim.t.qpos).all().item()), overflow=sim.overflow_flags())))
