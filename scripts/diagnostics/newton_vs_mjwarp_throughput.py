"""Physics-only throughput on the same G1 asset at the same 2.5 ms step: MuJoCo Warp (our BatchSim, graph
replay, 8 substeps per call) vs Newton XPBD on Metal (eager launches and graph capture), at 256 / 1024 / 4096 envs."""
import sys, time, numpy as np, warp as wp, newton, mujoco
wp.config.quiet = True
sys.path.insert(0, "scripts/diagnostics"); from newton_xpbd_g1 import robot_builder
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics.batch import BatchSim, BatchSimOptions
H = 0.0025; STEPS = 200
for N in (256, 1024, 4096):
    # MuJoCo Warp
    m, _ = build_g1_model("flat", physics_dt=H)
    sim = BatchSim(m, N, options=BatchSimOptions(substeps=8, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20)); sim.synchronize()
    for _ in range(3): sim.step()
    sim.synchronize(); t0 = time.perf_counter()
    for _ in range(STEPS // 8): sim.step()
    sim.synchronize(); mj = N * STEPS / (time.perf_counter() - t0)
    del sim
    # Newton XPBD
    with wp.ScopedDevice("metal:0"):
        rb = robot_builder(); s = newton.ModelBuilder(); s.gravity = -9.81; s.add_ground_plane(); s.replicate(rb, N, spacing=(2.5, 2.5, 0.0)); s.gravity = -9.81
        model = s.finalize(); solver = newton.solvers.SolverXPBD(model, iterations=4); s0, s1 = model.state(), model.state(); ctrl = model.control(); newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        contacts = model.collide(s0)
        def step():
            global s0, s1, contacts
            s0.clear_forces(); contacts = model.collide(s0, contacts); solver.step(s0, s1, ctrl, contacts, H); s0, s1 = s1, s0
        for _ in range(3): step()
        wp.synchronize(); t0 = time.perf_counter()
        for _ in range(STEPS): step()
        wp.synchronize(); nw_eager = N * STEPS / (time.perf_counter() - t0)
        nw_graph = float("nan")
        try:
            with wp.ScopedCapture() as cap:
                for _ in range(8): step()
            g = cap.graph
            wp.capture_launch(g); wp.synchronize(); t0 = time.perf_counter()
            for _ in range(STEPS // 8): wp.capture_launch(g)
            wp.synchronize(); nw_graph = N * STEPS / (time.perf_counter() - t0)
        except Exception as e:
            print(f"  (graph capture failed: {repr(e)[:120]})")
    print(f"N={N}: MuJoCo Warp {mj:,.0f} physics-steps/s | Newton XPBD eager {nw_eager:,.0f} | Newton XPBD graph {nw_graph:,.0f}  (env-steps/s at decimation 8: MJ {mj/8:,.0f}, Newton {nw_graph/8 if nw_graph == nw_graph else nw_eager/8:,.0f})", flush=True)
