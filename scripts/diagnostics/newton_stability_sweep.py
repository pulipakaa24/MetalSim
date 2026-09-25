"""Newton XPBD on the G1 flat task: 3-sigma stability (g1_preflight.stability: 1024 envs, 400 control steps of
N(0, 3) actions; pass = at most 1 blown-up episode) vs physics throughput (task.sim.step, graph, 4096 envs,
env-steps/s at 50 Hz), for solver iterations x substep, joint relaxation, and the actuator's leg branch
(stiff_implicit: stiffness integrated implicitly too, bias 1/(1 + kp dt^2 / I) at rest).

usage: python scripts/diagnostics/newton_stability_sweep.py [config ...]   config = IT:DT_MS[:relax=R][:stiff][:rc][:nolim]"""
import sys, io, time, contextlib, subprocess, numpy as np, warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn import g1_preflight

DEFAULT = ["4:1.25", "6:1.25", "8:1.25", "4:0.8333", "4:0.625", "6:0.8333", "8:0.625",
           "4:1.25:stiff", "6:1.25:stiff", "4:0.8333:stiff", "4:1.25:relax=0.3", "4:1.25:relax=0.5"]


def parse(c):
    p = c.split(":"); it, dt = int(p[0]), float(p[1]) * 1e-3; kw = {}
    for x in p[2:]:
        if x == "stiff": kw.setdefault("actuator_kw", {})["stiff_implicit"] = True
        elif x.startswith("relax="): kw["relaxation"] = float(x.split("=")[1])
        elif x == "nolim": kw["limit_margin"] = None        # USD limits as authored (no +-(pi - 0.15) clamp)
        elif x == "rc": kw["recenter"] = True          # revolute zeros at the middle of the limit range
    return it, dt, kw


print("install:", subprocess.run([sys.executable, "scripts/diagnostics/newton_stamp.py"], capture_output=True, text=True).stdout.strip())
print("| setting | blown of 1024 (3 sigma, 400 steps) | peak joint speed [rad/s] | preflight | physics env-steps/s (4096) |")
print("|---|---|---|---|---|", flush=True)
for c in sys.argv[1:] or DEFAULT:
    it, dt, kw = parse(c)
    task = G1VelocityTask(1024, terrain="flat", seed=0, engine="newton", newton_iterations=it, newton_dt=dt, newton_kw=kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = g1_preflight.stability(task)
    line = buf.getvalue(); peak = line.split("peak joint speed ")[-1].split(" rad/s")[0] if "peak" in line else "?"
    blown = task.blown_up_episodes
    del task
    big = G1VelocityTask(4096, terrain="flat", seed=0, engine="newton", newton_iterations=it, newton_dt=dt, newton_kw=kw)
    big.reset_all(); big.sim.step(); big.sim.synchronize(); t0 = time.perf_counter()
    for _ in range(30): big.sim.step()
    big.sim.synchronize(); r = 4096 * 30 / (time.perf_counter() - t0)
    del big
    print(f"| {c} | {blown} | {peak} | {'PASS' if ok else 'FAIL'} | {r:,.0f} |", flush=True)
