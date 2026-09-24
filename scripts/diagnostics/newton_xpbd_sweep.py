"""Which XPBD setting holds Isaac's G1 with Isaac's drives on Metal, and what it costs: sweep solver
iterations and substep size on a 2 s PD hold (4 envs), then throughput at 4096 envs for each stable setting."""
import sys, time, numpy as np, warp as wp, newton
wp.config.quiet = True
sys.path.insert(0, "scripts/diagnostics"); from newton_xpbd_g1 import robot_builder
def make(N, iters, **kw):
    rb = robot_builder(); s = newton.ModelBuilder(); s.gravity = -9.81; s.add_ground_plane(); s.replicate(rb, N, spacing=(2.5, 2.5, 0.0)); s.gravity = -9.81
    m = s.finalize(); solver = newton.solvers.SolverXPBD(m, iterations=iters, **kw); s0, s1 = m.state(), m.state(); ctrl = m.control(); newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    return m, solver, s0, s1, ctrl
with wp.ScopedDevice("metal:0"):
    stable = []
    for iters, h, kw in ((4, 0.0025, {}), (8, 0.0025, {}), (16, 0.0025, {}), (4, 0.00125, {}), (8, 0.00125, {}), (4, 0.0025, {"joint_angular_relaxation": 0.9, "joint_linear_relaxation": 0.9}), (8, 0.000625, {})):
        m, solver, s0, s1, ctrl = make(4, iters, **kw); contacts = None; vmax = 0.0
        for k in range(int(2.0 / h)):
            s0.clear_forces(); contacts = m.collide(s0, contacts); solver.step(s0, s1, ctrl, contacts, h); s0, s1 = s1, s0
            if k % int(0.25 / h) == 0: wp.synchronize(); vmax = max(vmax, float(np.abs(s0.body_qd.numpy()[:, 3:]).max()))
        wp.synchronize(); bq = s0.body_q.numpy().reshape(4, -1, 7); z = bq[:, 0, 2]
        ok = bool(np.all(z > 0.6) and vmax < 5.0)
        print(f"iters {iters:2d} h {h*1000:.3f} ms {kw}: pelvis z after 2 s {np.round(z, 3)}, max |v| {vmax:.1f} m/s -> {'STANDS' if ok else 'falls/unstable'}", flush=True)
        if ok: stable.append((iters, h, kw))
    for iters, h, kw in stable[:3]:
        N = 4096; m, solver, s0, s1, ctrl = make(N, iters, **kw); contacts = m.collide(s0)
        def step():
            global s0, s1, contacts
            s0.clear_forces(); contacts = m.collide(s0, contacts); solver.step(s0, s1, ctrl, contacts, h); s0, s1 = s1, s0
        for _ in range(3): step()
        wp.synchronize(); t0 = time.perf_counter(); steps = int(0.5 / h)
        for _ in range(steps): step()
        wp.synchronize(); r = N * steps / (time.perf_counter() - t0)
        print(f"N=4096 iters {iters} h {h*1000:.3f} ms {kw}: {r:,.0f} physics-steps/s -> {r * h / 0.02:,.0f} env-steps/s at 50 Hz control (eager)", flush=True)
