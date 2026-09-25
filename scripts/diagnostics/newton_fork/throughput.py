"""Physics-only throughput of the Newton fork's options on Metal: graph replay of one 50 Hz control step
(NewtonSim.step), N envs (default 4096), env-steps/s, median of 3 blocks of 20 steps after warm-up.
Configs: IT:DT_MS[:relax=L/A][:color][:extra=K][:solverdrive][:mode=pd|implicit|compliance], e.g. 4:1.25  4:0.625  4:1.25:relax=0.8/0.8:color
Run through the GPU queue: scripts/gpu_run.sh NAME timing 15 -- .venv-newtonfork/bin/python scripts/diagnostics/newton_fork/throughput.py ..."""
import sys, time, inspect, numpy as np, warp as wp, newton
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb

args = sys.argv[1:]
N = 4096
if args and args[0].startswith("N="):
    N = int(args.pop(0)[2:])
FORK = "joint_coloring" in inspect.signature(newton.solvers.SolverXPBD.__init__).parameters
m = build_g1_model("flat", physics_dt=0.0025)[0]
print(f"newton at {newton.__file__}, fork {FORK}, N {N}", flush=True)


@wp.kernel
def _target_to_coord(target: wp.array[float], dof: wp.array[int], coord: wp.array[int], out: wp.array[float]):
    i = wp.tid()
    out[coord[i]] = target[dof[i]]


for cfg in args:
    p = cfg.split(":"); it, dt = int(p[0]), float(p[1]) * 1e-3; relax = (0.4, 0.4); skw = {}; solverdrive = False
    if FORK:
        skw["joint_armature_inertia"] = "none"
    for x in p[2:]:
        if x.startswith("relax="): relax = tuple(float(v) for v in x[6:].split("/"))
        elif x == "color": skw["joint_coloring"] = True
        elif x.startswith("extra="): skw["joint_extra_iterations"] = int(x[6:])
        elif x == "solverdrive": solverdrive = True; skw["joint_drive_mode"] = "pd"
        elif x.startswith("mode="): skw["joint_drive_mode"] = x[5:]
    _X = newton.solvers.SolverXPBD
    class _XR(_X):
        def __init__(self, mm, **k):
            k["joint_linear_relaxation"], k["joint_angular_relaxation"] = relax
            super().__init__(mm, **k)
    newton.solvers.SolverXPBD = _XR
    try:
        sim = nb.NewtonSim(m, N, iterations=it, dt=dt, solver_kw=skw or None)
        if solverdrive:
            M, act = sim.model, sim.actuator
            M.joint_target_ke.assign(act.kp.numpy()); M.joint_target_kd.assign(act.kd.numpy()); act.kp.zero_(); act.kd.zero_()
            sim.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            qs, qds = M.joint_q_start.numpy(), M.joint_qd_start.numpy()
            rev = [j for j in range(M.joint_count) if qds[j + 1] - qds[j] == 1]
            dof = wp.array([int(qds[j]) for j in rev], dtype=int, device=sim.device); coord = wp.array([int(qs[j]) for j in rev], dtype=int, device=sim.device)
            _apply = act.apply
            def apply(state, control, _apply=_apply, dof=dof, coord=coord, n=len(rev)):
                _apply(state, control)
                wp.launch(_target_to_coord, dim=n, inputs=[act.target, dof, coord], outputs=[control.joint_target_q], device=sim.device)
            act.apply = apply
        sim.step(); sim.synchronize()
        rates = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(20): sim.step()
            sim.synchronize(); rates.append(N * 20 / (time.perf_counter() - t0))
        print(f"| {cfg} | {np.median(rates):,.0f} env-steps/s | blocks {', '.join(f'{r:,.0f}' for r in rates)} |", flush=True)
        del sim
    except Exception as e:
        print(f"| {cfg} | failed {e!r:.200} |", flush=True)
    newton.solvers.SolverXPBD = _X
