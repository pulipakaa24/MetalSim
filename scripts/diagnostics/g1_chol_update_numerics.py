"""Numerical prototype (numpy, no GPU): maintaining the Newton Hessian's Cholesky factor by rank-1 updates /
downdates (MuJoCo C's HessianIncremental) instead of refactorizing, in float32, on the real sequence of H
matrices and constraint-state flips of the G1 task (Warp CPU device, random actions).

For every (world, iteration) with 1..K quadratic flips: factor_ref = chol(H_new) in float32 (what MuJoCo Warp
computes), factor_upd = rank-1 updates of the previous iteration's float32 factor with x = sqrt(D) J_row, and
the float64 truth chol(H_new). Reports the error of the Newton search direction s = -H^-1 grad from either
factor against the float64 one, relative to |s|, so the update path can be compared with the refactor path's
own float32 error. Also the diagonal loss ratio of downdates (min over k of r^2 / L[k,k]^2), the predictor
of accuracy loss.

usage: python scripts/diagnostics/g1_chol_update_numerics.py [N=16] [CONTROL_STEPS=4] [preset=recommended] [K=8]"""
import sys, time
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver as S, types as T
from metalsim.learn.g1_velocity import build_g1_model, ACTION_SCALE
from metalsim.physics import contact_tuning

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 16
STEPS = int(args[1]) if len(args) > 1 else 4
PRESET = args[2] if len(args) > 2 else "recommended"
K = int(args[3]) if len(args) > 3 else 8
FORM = args[4] if len(args) > 4 else "rotation"
BUDGETS = [1, 2, 4, 8, 10**9]
DT = 0.0025; SUB = 8
dev = wp.get_device("cpu")


FORM = FORM if "FORM" in dir() else "rotation"   # "linpack": L' = (L + s x)/c; "rotation": Givens rotation for adds (orthogonal), hyperbolic for removes


def chol_update(L, x, sign):
    """Rank-1 update (sign +1) / downdate (sign -1) of lower L (L L^T + sign x x^T), in L's dtype.
    Returns (L, min diagonal ratio r^2 / Lkk^2, ok)."""
    n = L.shape[0]; x = x.copy(); ratio = np.inf
    for k in range(n):
        Lkk = L[k, k]
        r2 = Lkk * Lkk + sign * x[k] * x[k]
        if sign < 0:
            ratio = min(ratio, float(r2 / (Lkk * Lkk)))
        if r2 <= 0:
            return L, ratio, False
        r = np.sqrt(r2)
        if FORM == "rotation":
            c = Lkk / r; s = x[k] / r          # add: [c s; -s c] orthogonal; remove: [1/c -s/c...] hyperbolic
            L[k, k] = r
            if k + 1 < n:
                if sign > 0:
                    Lk = c * L[k + 1:, k] + s * x[k + 1:]
                    x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]
                    L[k + 1:, k] = Lk
                else:
                    Lk = (L[k + 1:, k] - s * x[k + 1:]) / c
                    x[k + 1:] = (x[k + 1:] - s * L[k + 1:, k]) / c
                    L[k + 1:, k] = Lk
        else:
            c = r / Lkk; s = x[k] / Lkk
            L[k, k] = r
            if k + 1 < n:
                L[k + 1:, k] = (L[k + 1:, k] + sign * s * x[k + 1:]) / c
                x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]
    return L, ratio, True


def solve(L, g):
    y = np.linalg.solve(L, g); return -np.linalg.solve(L.T, y)


m, _ = build_g1_model(physics_dt=DT)
if PRESET != "default":
    contact_tuning.apply(m, PRESET)
mjd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, mjd, 0); mujoco.mj_forward(m, mjd)
nv = m.nv
with wp.ScopedDevice(dev):
    mw = mjw.put_model(m); mw.opt.graph_conditional = False; mw.opt.warn_overflow = 0
    d = mjw.put_data(m, mjd, nworld=N, nconmax=128, njmax=512)
    orig = S._solver_iteration
    trace = {"prevH": None, "L32": {U: {} for U in BUDGETS}, "nupd": {U: {} for U in BUDGETS}, "rows": []}
    rows = trace["rows"]

    def patched(m_, d_, ctx, nsolving, compact=False):
        done_before = ctx.done.numpy().copy()
        orig(m_, d_, ctx, nsolving, compact=compact)
        H = ctx.h.numpy()[:, :nv, :nv]; qc = ctx.quad_changed_count.numpy(); ids = ctx.quad_changed_ids.numpy()
        J = d_.efc.J.numpy()[:, :, :nv]; D = d_.efc.D.numpy(); st = d_.efc.state.numpy(); grad = ctx.grad.numpy()[:, :nv]
        sc = ctx.state_changed_count.numpy()
        for w in range(N):
            if done_before[w]:
                continue
            Hs = np.triu(H[w]); Hs = Hs + np.triu(Hs, 1).T          # upper triangle held
            Lref = np.linalg.cholesky(Hs.astype(np.float32))
            L64 = np.linalg.cholesky(Hs.astype(np.float64))
            g = grad[w].astype(np.float64)
            s64 = solve(L64, g); sref = solve(Lref.astype(np.float64), g); nrm = np.linalg.norm(s64) + 1e-30
            errs = {}
            for U in BUDGETS:
                L32, nupd = trace["L32"][U], trace["nupd"][U]
                nf = int(qc[w])
                if w in L32 and 0 < nf <= K and sc[w] > 0 and nupd.get(w, 0) + nf <= U:
                    L = L32[w].copy(); ratio = np.inf; ok = True
                    for e in ids[w, :nf]:
                        x = (np.sqrt(D[w, e]) * J[w, e]).astype(np.float32)
                        sign = 1.0 if st[w, e] == T.ConstraintState.QUADRATIC.value else -1.0
                        L, rt, ok1 = chol_update(L, x, np.float32(sign)); ratio = min(ratio, rt); ok = ok and ok1
                    if ok:
                        L32[w] = L; nupd[w] = nupd.get(w, 0) + nf
                        errs[U] = np.linalg.norm(solve(L.astype(np.float64), g) - s64) / nrm
                        continue
                L32[w] = Lref; nupd[w] = 0
                errs[U] = np.linalg.norm(sref - s64) / nrm
            if any(U in errs for U in BUDGETS):
                rows.append((int(qc[w]), np.linalg.norm(sref - s64) / nrm, [errs[U] for U in BUDGETS], sc[w] > 0 and 0 < qc[w] <= K))
    S._solver_iteration = patched
    rng = np.random.default_rng(0); q0 = m.key_qpos[0]
    t0 = time.time()
    try:
        for s in range(STEPS):
            d.ctrl.assign((q0[7:][None] + ACTION_SCALE * rng.uniform(-1, 1, (N, m.nu))).astype(np.float32))
            for _ in range(SUB):
                for U in BUDGETS: trace["L32"][U].clear(); trace["nupd"][U].clear()   # a new solve starts fresh
                mjw.step(mw, d)
    finally:
        S._solver_iteration = orig
upd = [row for row in rows if row[3]]
print(f"{PRESET}: {len(rows)} (world, iteration) solves, {len(upd)} of them with 1..{K} flips, over {N} worlds x {STEPS*SUB} substeps ({time.time()-t0:.0f} s)")
ref = np.array([row[1] for row in rows])
print(f"  refactor float32 search-direction error vs float64 (relative): median {np.median(ref):.2e} p99 {np.percentile(ref, 99):.2e} max {ref.max():.2e}")
for i, U in enumerate(BUDGETS):
    e = np.array([row[2][i] for row in rows])
    print(f"  update budget U={U if U < 10**9 else 'inf':>4}: over all solves median {np.median(e):.2e} p99 {np.percentile(e, 99):.2e} max {e.max():.2e}; "
          f"share of solves refactored: {100*np.mean(np.isclose(e, ref)):.0f} %")
