"""Newton-solver convergence statistics of the G1 task per contact / solver preset, on Warp's CPU device
(no GPU; the iteration path is the same as on Metal up to float noise).

Per Newton iteration, over all worlds and substeps: worlds still solving (ctx.done false), worlds that
re-factorize the Hessian this iteration (state_changed_count > 0, MuJoCo Warp's rule), of which worlds with
no quadratic flip (quad_changed_count == 0: the factor is unchanged, only the solve is needed), the histogram
of quadratic flips per refactoring world, and adds vs removes (a removed row is a Cholesky downdate in
MuJoCo C's HessianIncremental). At the end: solver_niter distribution and iteration-cap hits.

usage: python scripts/diagnostics/g1_solver_iters.py [N=64] [CONTROL_STEPS=8] [presets...]
  presets: contact_cfg names (metalsim.physics.contact_tuning) or "solver:<solver_presets name>"; default:
  default hardlimits recommended solver:isaaclab3_every_substep_cap20"""
import os, sys, time
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver as S, types as T
from metalsim.learn.g1_velocity import build_g1_model, ACTION_SCALE
from metalsim.physics import contact_tuning, solver_presets

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 64
STEPS = int(args[1]) if len(args) > 1 else 8
PRESETS = args[2:] or ["default", "hardlimits", "recommended", "solver:isaaclab3_every_substep_cap20"]
DT = 0.0025
SUB = int(round(0.02 / DT))
dev = wp.get_device("cpu")


class Stats:
    def __init__(self, iters, cone=False):
        self.iters = iters; self.cone = cone
        self.undone = np.zeros(iters, np.int64); self.refactor = np.zeros(iters, np.int64)
        self.solve_only = np.zeros(iters, np.int64); self.adds = np.zeros(iters, np.int64); self.removes = np.zeros(iters, np.int64)
        self.hist = np.zeros((iters, 6), np.int64)   # flips per refactoring world: 1, 2, 3-4, 5-8, 9-16, >16
        self.substeps = 0; self.niter = []; self.cap = 0
        self.ls_exh = np.zeros(iters, np.int64)
        self.cone_worlds = np.zeros(iters, np.int64); self.cone_rows_hist = np.zeros((iters, 8), np.int64)   # k = cone rows per cone world: 1-2,3-4,5-6,7-8,9-12,13-16,17-24,>24
        self.cone_and_flip = np.zeros(iters, np.int64)

    def record(self, k, ctx, d):
        done = ctx.done.numpy(); sc = ctx.state_changed_count.numpy(); qc = ctx.quad_changed_count.numpy()
        if sc.size == 0:      # no incremental tracking on this path (elliptic cones on the installed fork): count solving worlds only
            self.undone[k] += (~done).sum(); return
        ids = ctx.quad_changed_ids.numpy(); st = d.efc.state.numpy()
        und = ~done
        self.undone[k] += und.sum()
        rf = und & (sc > 0)
        self.refactor[k] += rf.sum()
        self.solve_only[k] += (rf & (qc == 0)).sum()
        self.ls_exh[k] += (und & ctx.ls_exhausted.numpy()).sum()
        if self.cone:
            cone = (st == T.ConstraintState.CONE.value)
            nrow = np.array([cone[w, :int(n)].sum() for w, n in enumerate(d.nefc.numpy())])
            cw = und & (nrow > 0)
            self.cone_worlds[k] += cw.sum(); self.cone_and_flip[k] += (cw & (qc > 0)).sum()
            for n in nrow[cw]:
                self.cone_rows_hist[k, 0 if n <= 2 else 1 if n <= 4 else 2 if n <= 6 else 3 if n <= 8 else 4 if n <= 12 else 5 if n <= 16 else 6 if n <= 24 else 7] += 1
        for w in np.nonzero(rf & (qc > 0))[0]:
            n = int(qc[w])
            self.hist[k, 0 if n == 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 8 else 4 if n <= 16 else 5] += 1
            for e in ids[w, :n]:
                if st[w, e] == T.ConstraintState.QUADRATIC.value: self.adds[k] += 1
                else: self.removes[k] += 1


def run(name):
    m, _ = build_g1_model(physics_dt=DT)
    if name.startswith("solver:"):
        contact_tuning.apply(m, "recommended"); solver_presets.apply(m, name[7:])
    elif name != "default":
        contact_tuning.apply(m, name)
    mjd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, mjd, 0); mujoco.mj_forward(m, mjd)
    with wp.ScopedDevice(dev):
        mw = mjw.put_model(m); mw.opt.graph_conditional = False; mw.opt.warn_overflow = 0
        d = mjw.put_data(m, mjd, nworld=N, nconmax=128, njmax=512)
        iters = int(mw.opt.iterations)
        stats = Stats(iters, cone=int(m.opt.cone) == 1)
        orig = S._solver_iteration
        state = {"k": 0}

        def patched(m_, d_, ctx, nsolving, compact=False):
            orig(m_, d_, ctx, nsolving, compact=compact)
            stats.record(state["k"], ctx, d_); state["k"] += 1
        S._solver_iteration = patched
        rng = np.random.default_rng(0)
        q0 = m.key_qpos[0]
        t0 = time.time()
        try:
            for s in range(STEPS):
                ctrl = q0[7:][None] + ACTION_SCALE * rng.uniform(-1, 1, (N, m.nu))
                d.ctrl.assign(ctrl.astype(np.float32))
                for _ in range(SUB):
                    state["k"] = 0
                    mjw.step(mw, d)
                    stats.substeps += 1
                    ni = d.solver_niter.numpy(); stats.niter.append(ni.copy())
                    stats.cap += int((ni >= iters).sum())
        finally:
            S._solver_iteration = orig
        ov = d.overflow.numpy()
        el = time.time() - t0
    W = N * stats.substeps
    ni = np.concatenate(stats.niter)
    print(f"\n=== {name}: N={N}, {stats.substeps} substeps ({el:.0f} s), iterations cap {iters}; solver_niter mean {ni.mean():.2f} "
          f"p50 {np.percentile(ni, 50):.0f} p90 {np.percentile(ni, 90):.0f} max {ni.max()}; at cap {stats.cap}/{W} "
          f"({100*stats.cap/W:.1f} %); overflow ITERATIONS {int(((ov & int(mjw.OverflowType.ITERATIONS)) != 0).sum())} "
          f"LS {int(((ov & int(mjw.OverflowType.LS_ITERATIONS)) != 0).sum())} worlds; nefc mean {d.nefc.numpy().mean():.1f}")
    print("  it | solving % | refactor % | solve-only % | ls_exh % | flips/refactor world: 1 | 2 | 3-4 | 5-8 | 9-16 | >16 | adds | removes")
    for k in range(iters):
        h = stats.hist[k]
        print(f"  {k+1:2d} | {100*stats.undone[k]/W:8.1f} | {100*stats.refactor[k]/W:9.1f} | {100*stats.solve_only[k]/W:11.1f} | "
              f"{100*stats.ls_exh[k]/W:7.1f} | " + " | ".join(f"{v:5d}" for v in h) + f" | {stats.adds[k]:5d} | {stats.removes[k]:5d}")
    if stats.cone:
        print("  elliptic: it | cone worlds (still solving) | of which with a quad flip too | cone rows per cone world: <=2 | 3-4 | 5-6 | 7-8 | 9-12 | 13-16 | 17-24 | >24")
        for k in range(iters):
            print(f"  {k+1:2d} | {stats.cone_worlds[k]:6d} | {stats.cone_and_flip[k]:6d} | " + " | ".join(f"{v:5d}" for v in stats.cone_rows_hist[k]))
        print(f"  cone-world iterations total {stats.cone_worlds.sum()} of {stats.undone.sum()} solving world-iterations; without a quad flip {stats.cone_worlds.sum() - stats.cone_and_flip.sum()}")
    tot_ref = stats.refactor.sum(); tot_flip = stats.hist.sum()
    print(f"  total refactorizations {tot_ref} = {tot_ref/W:.2f} per world-substep (initial factorization excluded); "
          f"solve-only {stats.solve_only.sum()}; flips {stats.adds.sum()} adds / {stats.removes.sum()} removes over {tot_flip} worlds "
          f"(mean {(stats.adds.sum()+stats.removes.sum())/max(tot_flip,1):.2f} per refactoring world)")


for p in PRESETS:
    run(p)
