"""SolverKamino (Newton 1.5.2 or the 1.7.0.dev fork) on the closed-loop test mechanisms (mechanisms.py): hard loop
closure via a loop-closing revolute joint. Records world-0 body angles and per-step loop-closure error (max over
worlds), and optionally times graph-replayed steps.

    .venv-newtonfork/bin/python scripts/diagnostics/closed_loops/kamino_probe.py --mech fourbar --device cpu --dt 0.0025
    scripts/gpu_run.sh kamino_probe low 20 -- .venv-newtonfork/bin/python .../kamino_probe.py --device metal:0 ...

--torque none | sigma3: per-world Gaussian joint torques 3 x SIGMA_TAU (mechanisms.py), held for 20 ms (50 Hz
control, the policy rate), seed per world; the torque table is identical to mjw_loops.py's for the same seed.
"""
import argparse, json, math, os, sys, time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanisms as M  # noqa: E402
from mechanisms import torque_table  # noqa: E402,F401


def body_angles(body_q):
    """direction angle of each rod's local +x in the x-z plane, from Newton transforms (..., nb, 7)."""
    x, y, z, w = body_q[..., 3], body_q[..., 4], body_q[..., 5], body_q[..., 6]
    dx = 1 - 2 * (y * y + z * z); dz = 2 * (x * z - w * y)
    return np.arctan2(dz, dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="fourbar", choices=list(M.GEOMS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--torque", default="none", choices=("none", "sigma3"))
    ap.add_argument("--solver", default="padmm", choices=("padmm", "dvi"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--alpha", type=float, default=None, help="Baumgarte alpha for bilateral joints (default 0.01)")
    ap.add_argument("--graph_conditionals", type=int, default=1)
    ap.add_argument("--capture", type=int, default=0, help="replay one captured step graph")
    ap.add_argument("--bench_steps", type=int, default=0, help="time this many extra steps after the run")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    wp.config.quiet = True
    import newton
    dev = wp.get_device(a.device)
    wp.set_device(dev)

    sub, ids, g = M.newton_builder(a.mech, damped=a.torque == "sigma3")
    newton.solvers.SolverKamino.register_custom_attributes(sub)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-M.G)
    for _ in range(a.worlds):
        b.add_world(sub)
    model = b.finalize(device=dev, skip_validation_joints=True)
    cfg = newton.solvers.SolverKamino.Config.from_model(model)
    if hasattr(cfg, "dynamics_solver"):
        cfg = newton.solvers.SolverKamino.Config.from_model(model, dynamics_solver=a.solver) if a.solver != "padmm" else cfg
    cfg.use_collision_detector = False
    cfg.padmm.max_iterations = a.iters
    cfg.padmm.primal_tolerance = cfg.padmm.dual_tolerance = cfg.padmm.compl_tolerance = a.tol
    if hasattr(cfg.padmm, "use_graph_conditionals"):
        cfg.padmm.use_graph_conditionals = bool(a.graph_conditionals)
    if a.alpha is not None:
        cfg.constraints.alpha = a.alpha
    t0 = time.time()
    solver = newton.solvers.SolverKamino(model=model, config=cfg)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    solver.reset(state=s0)
    nb = len(ids)
    # actuated dof indices within one world (joint order = builder order; one dof per revolute)
    jnames = [j for j, _, _ in g["joints"]] + [f"loop{k}" for k in range(len(g["loops"]))]
    act_idx = [jnames.index(j) for j in g["actuated"]]
    ndof_w = len(jnames)
    tau_tab = torque_table(a.mech, a.worlds, a.T) if a.torque == "sigma3" else None
    jf = np.zeros((a.worlds, ndof_w), np.float32)

    nsteps = int(round(a.T / a.dt))
    hold = int(round(0.02 / a.dt))
    ang = np.zeros((nsteps + 1, nb)); clos = np.zeros(nsteps + 1); clos_p99 = np.zeros(nsteps + 1)
    bq = s0.body_q.numpy().reshape(a.worlds, nb, 7)
    ang[0] = body_angles(bq[0]); clos[0] = M.newton_closure_error(a.mech, bq).max()
    blown_at = None
    graph = None
    first_step_s = None
    peak = 0.0
    for k in range(nsteps):
        if tau_tab is not None and k % hold == 0:
            jf[:, act_idx] = tau_tab[k // hold]
            ctrl.joint_f.assign(jf.reshape(-1))
        ts = time.time()
        solver.step(s0, s1, ctrl, None, a.dt)
        s0, s1 = s1, s0
        bq = s0.body_q.numpy().reshape(a.worlds, nb, 7)
        if first_step_s is None:
            first_step_s = time.time() - ts
        ang[k + 1] = body_angles(bq[0])
        ce = M.newton_closure_error(a.mech, bq)
        fin = np.isfinite(ce)
        clos[k + 1] = ce[fin].max() if fin.any() else np.nan
        clos_p99[k + 1] = np.percentile(ce[fin], 99) if fin.any() else np.nan
        if blown_at is None and not fin.all():
            blown_at = k
        if k % 400 == 399:
            print(f"[kamino_probe] step {k + 1}/{nsteps} {time.time() - t0:.0f} s", flush=True)
        if k % 20 == 19:
            w = s0.body_qd.numpy().reshape(a.worlds, nb, 6)[..., 3:]
            ok = np.isfinite(w).all(axis=(1, 2))
            if ok.any():
                peak = max(peak, float(np.abs(w[ok]).max()))
    run_s = time.time() - t0
    qd = s0.body_qd.numpy().reshape(a.worlds, nb, 6)
    res = dict(mech=a.mech, device=str(dev), dt=a.dt, T=a.T, worlds=a.worlds, torque=a.torque, solver=a.solver,
               iters=a.iters, tol=a.tol, alpha=a.alpha if a.alpha is not None else 0.01, newton=newton.__version__,
               closure_max=float(np.nanmax(clos)), closure_mean=float(np.nanmean(clos)), closure_final=float(clos[-1]),
               nonfinite_worlds=int((~np.isfinite(qd).all(axis=(1, 2))).sum()), blown_at_step=blown_at,
               peak_body_ang_speed=peak, wall_s=run_s, first_step_s=first_step_s)
    st = getattr(solver, "status", None)
    if st is not None:
        try:
            res["status_sample"] = str(st.numpy()[:4])
        except Exception:  # noqa: BLE001
            pass
    if a.bench_steps:
        if a.capture:
            with wp.ScopedCapture(device=dev) as cap:
                solver.step(s0, s1, ctrl, None, a.dt)
                solver.step(s1, s0, ctrl, None, a.dt)
            graph = cap.graph
        wp.synchronize_device(dev)
        tb = time.time()
        n = a.bench_steps
        for _ in range(n // 2):
            if graph is not None:
                wp.capture_launch(graph)
            else:
                solver.step(s0, s1, ctrl, None, a.dt); solver.step(s1, s0, ctrl, None, a.dt)
        wp.synchronize_device(dev)
        el = time.time() - tb
        res["bench"] = dict(steps=n, s=el, env_steps_per_s=a.worlds * n / el, captured=bool(graph is not None))
    print(json.dumps(res), flush=True)
    if a.out:
        np.savez(a.out, t=np.arange(nsteps + 1) * a.dt, ang=ang, closure=clos, closure_p99=clos_p99, meta=json.dumps(res))


if __name__ == "__main__":
    main()
