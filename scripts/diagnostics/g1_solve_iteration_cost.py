"""Marginal graph-mode cost of each Newton iteration of the G1 task's constraint solve at 4096 envs: the solve
(`solver.solve`) is captured as a graph with the iteration cap set to k = 0..10 and replayed 20 times on a fixed
state (the solve writes qacc / qfrc_constraint only, so replays are idempotent); cost(k) - cost(k-1) is what
iteration k costs in graph mode, where converged worlds have exited. The state is the one after 20 random-action
control steps (contact-rich, limits driven). Also prints the cost of the pieces before the loop (k = 0).

usage: MJW_TP_VARIANT='{"contact_cfg":"recommended"}' python scripts/diagnostics/g1_solve_iteration_cost.py [N=4096] [KMAX=10]"""
import json, os, sys, time
import numpy as np, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver as S
import metalsim.learn.g1_velocity as g1v
from metalsim.physics.batch import BatchSimOptions

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 4096
KMAX = int(args[1]) if len(args) > 1 else 10
VAR = json.loads(os.environ.get("MJW_TP_VARIANT", "{}") or "{}")
_orig = BatchSimOptions
g1v.BatchSimOptions = lambda **kw: _orig(**{**kw, **{k: VAR[k] for k in ("njmax", "nconmax", "jacobian", "m_dense_max") if k in VAR}})
task = g1v.G1VelocityTask(N, terrain="flat", physics_dt=0.0025, seed=0, **{k: VAR[k] for k in ("contact_cfg", "solver_cfg") if k in VAR})
task.reset_all()
dev = task.device; m, d = task.sim.m, task.sim.d
print(f"variant {VAR}; iterations cap {m.opt.iterations}, ls {m.opt.ls_iterations}, njmax {d.njmax}, naconmax {d.naconmax}", flush=True)
from metalsim.learn.warp_policy import RolloutBuffers, bump


class _Pol:
    step_idx = wp.zeros(1, dtype=int, device=dev)


pol = _Pol(); bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device=dev); bufs.done = wp.zeros((1, N), dtype=float, device=dev)
rng = np.random.default_rng(0)
with wp.ScopedDevice(dev):
    for k in range(20):
        a = wp.array(rng.uniform(-1, 1, (N, task.act_dim)).astype(np.float32), dtype=float, device=dev)
        wp.launch(bump, dim=1, inputs=[pol.step_idx], device=dev)
        task.launch_apply_action(a); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
    # everything up to the solve of the next substep, so the solve sees its real inputs
    import mujoco_warp._src.forward as F, mujoco_warp._src.sensor as SE
    F.fwd_position(m, d, factorize=False); SE.sensor_pos(m, d); F.fwd_velocity(m, d); SE.sensor_vel(m, d)
    F.fwd_actuation(m, d); F.fwd_acceleration(m, d, factorize=True)
    task.sim.synchronize()
    nefc = d.nefc.numpy(); print(f"state: nefc mean {nefc.mean():.1f} max {nefc.max()}, nacon {int(d.nacon.numpy()[0]) / N:.2f} per world", flush=True)
    cap = int(m.opt.iterations)
    prev = None
    rows = []
    for k in range(0, min(KMAX, cap) + 1):
        m.opt.iterations = k
        with wp.ScopedCapture(device=dev) as c:
            for _ in range(20):
                S.solve(m, d)
        wp.capture_launch(c.graph); task.sim.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); wp.capture_launch(c.graph); task.sim.synchronize(); ts.append((time.perf_counter() - t0) / 20)
        t = sorted(ts)[1] * 1e3
        ov = task.sim.overflow_flags()
        niter = d.solver_niter.numpy()
        rows.append((k, t, t - prev if prev is not None else t, int((niter < k).sum()) if k else 0))
        prev = t
    m.opt.iterations = cap
print("  k | solve ms (graph, per substep) | marginal ms of iteration k | worlds converged before iteration k")
for k, t, dt, conv in rows:
    print(f"  {k:2d} | {t:7.3f} | {dt:7.3f} | {conv:5d}")
