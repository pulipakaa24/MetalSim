"""CPU-device check of the elliptic cone term as rank-1 factor updates (MJW_ELLIPTIC_CONE_UPDATE=1) against the
per-entry cone term + refactorization (mode 2) on the G1 task with the ellip10 preset: (1) per Newton iteration,
the search direction of each path against the float64 solve of h + the cone term assembled in float64 from the
same rows (relative error, over all solving worlds); (2) the trajectories of the two paths over a few substeps
(max |dq|), against the two-instance floor of the same path. Each path runs in its own process (the flag is read
at import): this script runs one path and saves; --ref compares.

usage: MJW_FUSE_H_CHOLESKY_CPU=1 MJW_ELLIPTIC_CONE_UPDATE=0|1 python scripts/diagnostics/cone_update_cpu_check.py [N=16] [STEPS=3] --out F.npz [--ref G.npz]"""
import os, sys, time
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver as S, types as T
from metalsim.learn.g1_velocity import build_g1_model, ACTION_SCALE
from metalsim.physics import contact_tuning

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 16
STEPS = int(args[1]) if len(args) > 1 else 3
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
REF = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
print(f"mujoco_warp {mjw.__file__}; MJW_ELLIPTIC_CONE_UPDATE={os.environ.get('MJW_ELLIPTIC_CONE_UPDATE')} KMAX={S._CONE_UPDATE_KMAX} mode {S._ELLIPTIC_INCREMENTAL_MODE}", flush=True)
m, _ = build_g1_model(physics_dt=0.0025); contact_tuning.apply(m, "tau10_impact_hardlimits_ellip10")
mjd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, mjd, 0); mujoco.mj_forward(m, mjd)
nv = m.nv
dev = wp.get_device("cpu")
with wp.ScopedDevice(dev):
    mw = mjw.put_model(m); mw.opt.graph_conditional = False; mw.opt.warn_overflow = 0
    d = mjw.put_data(m, mjd, nworld=N, nconmax=128, njmax=512)
    assert S._elliptic_incremental(mw), "the elliptic incremental path is not active on this device"
    orig = S._solver_iteration
    errs = []

    def cone_hessian_f64(w, ctx):
        """h (float32, QUADRATIC rows) + the cone term assembled in float64 from the cone list of this iteration."""
        H = ctx.h.numpy()[w, :nv, :nv].astype(np.float64)
        H = np.triu(H) + np.triu(H, 1).T
        J = d.efc.J.numpy()[w, :, :nv].astype(np.float64)
        for k in range(int(ctx.cone_count.numpy()[w])):
            e0 = int(ctx.cone_efcid.numpy()[w, k]); t = ctx.cone_terms.numpy()[w, k].astype(np.float64)
            nrows = int(t[0]); mu, dm, mot, mnt, tdg = t[6], t[12], t[13], t[14], t[15]
            z = np.stack([mu * J[e0]] + [t[6 + r] * J[e0 + r] for r in range(1, nrows)])
            Hc = np.zeros((nrows, nrows)); Hc[0, 0] = dm
            for r in range(1, nrows):
                Hc[0, r] = Hc[r, 0] = -dm * mot * t[r]
                for c in range(1, nrows):
                    Hc[r, c] = dm * mnt * t[r] * t[c]
                Hc[r, r] += dm * tdg
            H += z.T @ Hc @ z
        return H

    def patched(m_, d_, ctx, nsolving, compact=False):
        done_before = ctx.done.numpy().copy()
        orig(m_, d_, ctx, nsolving, compact=compact)
        done_after = ctx.done.numpy()
        grad = ctx.grad.numpy()[:, :nv].astype(np.float64); search = ctx.search.numpy()[:, :nv].astype(np.float64)
        nvec = ctx.cone_nvec.numpy() if ctx.cone_nvec is not None and ctx.cone_nvec.shape[0] else None
        for w in range(N):
            if done_before[w]:
                continue
            H = cone_hessian_f64(w, ctx)
            try:
                s64 = -np.linalg.solve(H, grad[w])
            except np.linalg.LinAlgError:
                continue
            e = np.linalg.norm(search[w] - s64) / (np.linalg.norm(s64) + 1e-30)
            errs.append((e, int(nvec[w]) if nvec is not None else -1))
    S._solver_iteration = patched
    rng = np.random.default_rng(0); q0 = m.key_qpos[0]
    snaps = []
    t0 = time.time()
    try:
        for s_ in range(STEPS):
            d.ctrl.assign((q0[7:][None] + ACTION_SCALE * rng.uniform(-1, 1, (N, m.nu))).astype(np.float32))
            for _ in range(8):
                mjw.step(mw, d)
            snaps.append(d.qpos.numpy().copy())
    finally:
        S._solver_iteration = orig
e = np.array([x for x, _ in errs]); k = np.array([y for _, y in errs])
print(f"{len(errs)} iteration solves over {N} worlds x {STEPS * 8} substeps ({time.time() - t0:.0f} s): search-direction error vs float64 of h + cone term: "
      f"median {np.median(e):.2e} p99 {np.percentile(e, 99):.2e} max {e.max():.2e}")
if (k >= 0).any():
    for lo, hi in ((0, 0), (1, 3), (4, 6), (7, 99)):
        sel = (k >= lo) & (k <= hi)
        if sel.any():
            print(f"   cone vectors {lo}-{hi}: {int(sel.sum()):5d} solves, median {np.median(e[sel]):.2e} p99 {np.percentile(e[sel], 99):.2e} max {e[sel].max():.2e}")
if OUT:
    np.savez_compressed(OUT, qpos=np.stack(snaps), err=e, k=k)
if REF:
    z = np.load(REF)
    for i, (a, b) in enumerate(zip(snaps, z["qpos"])):
        dq = np.abs(a - b)
        print(f"   control step {i + 1}: |dq| vs reference max {dq.max():.2e} p99 {np.percentile(dq, 99):.1e} median {np.median(dq):.1e}")
    print(f"   reference search-direction error: median {np.median(z['err']):.2e} p99 {np.percentile(z['err'], 99):.2e} max {z['err'].max():.2e}")
