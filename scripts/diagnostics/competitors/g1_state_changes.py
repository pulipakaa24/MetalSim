"""Per-Newton-iteration constraint-state statistics of the elliptic-cone solver on dumped G1-task states
(g1_state_dump.py), on Warp's CPU device: rows per world, cone-state contacts, rows flipping QUADRATIC state per
iteration (what an incremental Hessian update must apply), rows changing any state, iterations to convergence.
    python g1_state_changes.py runs/competitors/g1_states/ellip10 [--worlds 64] [--preset ellip10|pyr]"""
import argparse, os, sys, numpy as np, mujoco, warp as wp
ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--worlds", type=int, default=64); ap.add_argument("--cone", default=None)
a = ap.parse_args(); wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver, types
m = mujoco.MjModel.from_binary_path(a.prefix + ".mjb"); z = np.load(a.prefix + ".npz")
if a.cone: m.opt.cone = {"pyramidal": 0, "elliptic": 1}[a.cone]
print(f"model: nv {m.nv} cone {m.opt.cone} impratio {m.opt.impratio} iterations {m.opt.iterations} ls {m.opt.ls_iterations} dt {m.opt.timestep}; mujoco_warp {os.path.dirname(mjw.__file__)}")
N = a.worlds; rec = []
orig_iter = solver._solver_iteration
def hooked(*args, **kw):
    dd = kw.get("d", args[1] if len(args) > 1 else None); ctx = kw.get("ctx", args[2] if len(args) > 2 else None)
    st_before = dd.efc.state.numpy().copy()
    orig_iter(*args, **kw)
    rec.append((st_before, dd.efc.state.numpy().copy(), ctx.done.numpy().copy()))
solver._solver_iteration = hooked
with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); d0 = mujoco.MjData(m)
    dd = mjw.put_data(m, d0, nworld=N, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * N)
    tot = dict(rows=[], cone_con=[], quad_flips=[], any_change=[], niter=[], cone_rows_frac=[])
    for t in z["steps"]:
        k = f"t{t}"
        dd.qpos.assign(z[k + "_qpos"][:N]); dd.qvel.assign(z[k + "_qvel"][:N]); dd.ctrl.assign(z[k + "_ctrl"][:N]); dd.qacc_warmstart.assign(z[k + "_qacc_warmstart"][:N])
        if z[k + "_act"].size: dd.act.assign(z[k + "_act"][:N])
        rec.clear(); mjw.forward(mm, dd)
        nefc = dd.nefc.numpy(); typ = dd.efc.type.numpy(); niter = dd.solver_niter.numpy()
        CONE, QUAD = int(types.ConstraintState.CONE), int(types.ConstraintState.QUADRATIC)
        ELL = int(types.ConstraintType.CONTACT_ELLIPTIC)
        per_it = []
        for it, (sb, sa, done) in enumerate(rec):
            rows = np.arange(sb.shape[1])[None, :] < nefc[:, None]
            active = rows & ~done[:, None]
            flips = ((sb == QUAD) != (sa == QUAD)) & active
            anych = (sb != sa) & active
            cone_rows = (sa == CONE) & active
            per_it.append((flips.sum(1), anych.sum(1), cone_rows.sum(1), (~done).sum()))
        cone_con = np.array([((sa == CONE) & (typ == ELL) & rows).sum() / 3 for _, sa, _ in rec[-1:]])
        f = np.array([p[0] for p in per_it]); c = np.array([p[1] for p in per_it]); cr = np.array([p[2] for p in per_it]); alive = np.array([p[3] for p in per_it])
        print(f"step t={t}: nefc mean {nefc[:N].mean():.1f} max {nefc[:N].max()}, elliptic rows mean {((typ == ELL) & rows).sum(1).mean():.1f}; "
              f"solver_niter mean {niter[:N].mean():.2f} (max {niter[:N].max()}); worlds alive per iteration {alive.tolist()}")
        print("   per iteration (mean over alive worlds): QUAD flips " + " ".join(f"{x:.2f}" for x in (f.sum(1) / np.maximum(alive, 1))) +
              " | any state change " + " ".join(f"{x:.2f}" for x in (c.sum(1) / np.maximum(alive, 1))) +
              " | CONE rows " + " ".join(f"{x:.1f}" for x in (cr.sum(1) / np.maximum(alive, 1))))
        print(f"   worlds with 0 QUAD flips per iteration: " + " ".join(f"{int(((fi == 0) & (al > 0)).sum())}" for fi, al in zip(f, [np.ones(N)] * len(f))) +
              f" of {N}; max flips in one world-iteration {f.max()}; CONE rows at the first iteration: {100 * cr[0].sum() / max(1, ((typ == ELL) & rows).sum()):.0f} % of elliptic rows")
        tot["quad_flips"].append(f.sum(1) / np.maximum(alive, 1)); tot["niter"].append(niter[:N].mean()); tot["cone_rows_frac"].append(cr[0].sum() / max(1, ((typ == ELL) & rows).sum()))
    print(f"SUMMARY cone={m.opt.cone} impratio={m.opt.impratio}: mean solver_niter {np.mean(tot['niter']):.2f}; QUAD flips per world-iteration by iteration index (mean over snapshots): "
          + " ".join(f"{x:.2f}" for x in np.mean([q[:min(len(t) for t in tot['quad_flips'])] for q in tot['quad_flips']], axis=0)) + f"; CONE-state share of elliptic rows at the first iteration {100 * np.mean(tot['cone_rows_frac']):.0f} %")
